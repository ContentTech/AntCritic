import os
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset.second_stage import SecondStageDataset
from dataset.first_stage import FirstStageDataset
from loss import calc_final_loss
from metrics import calc_final_metric
import models
from runner.optimizer import GradualWarmupScheduler
from utils.container import metricsContainer
from utils.helper import move_to_cuda, move_to_cpu
from utils.processor import tuple2dict
from utils.timer import Timer

class MainRunner:
    def __init__(self, config):
        print("Initialization Start.")
        self.config = config
        self._init_misc()
        self._init_dataset(config.dataset)
        self._init_model(config.model)
        self._init_optimizer(config.optimizer)
        print("Initialization End.")

    def _init_misc(self):
        seed = 8
        random.seed(seed)
        np.random.seed(seed + 1)
        torch.manual_seed(seed + 2)
        torch.cuda.manual_seed(seed + 3)
        torch.cuda.manual_seed_all(seed + 4)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print(self.config)
        self.model_saved_path = self.config["saved_path"]
        os.makedirs(self.model_saved_path, mode=0o755, exist_ok=True)
        self.device_ids = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
        print('GPU: {}'.format(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
        self.initial_epoch = 0

    def _init_dataset(self, config):
        train_dataset = FirstStageDataset("train", config)
        eval_dataset = FirstStageDataset("val", config)
        test_dataset = FirstStageDataset("test", config)
        self.train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                                       num_workers=8, pin_memory=False)
        self.eval_loader = DataLoader(eval_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=8,
                                      pin_memory=False)
        self.test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=8,
                                      pin_memory=False)

    def _init_model(self, model_config):
        self.model = getattr(models, model_config["name"])(**model_config).cuda()
        self.model = nn.DataParallel(self.model, device_ids=self.device_ids)

    def _init_optimizer(self, config):
        backbone_param = self.model.module.extractor.parameters()
        other_param = [param for param in self.model.parameters() if (param not in backbone_param)]
        self.optimizer = torch.optim.AdamW([{'params': other_param},
                                            {'params': list(backbone_param), 'lr': config["lr"] / 2.0}],
                                           lr=config["lr"],
                                           weight_decay=config["weight_decay"])

        self.sub_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config["T_max"])
        self.main_scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1,
                                                     total_epoch=config["warmup_epoch"],
                                                     after_scheduler=self.sub_scheduler)

    def _train_one_epoch(self, epoch, last_total_step):
        self.model.train()
        timer = Timer()
        batch_idx = 0
        total_time = []
        for batch_idx, (inputs, targets) in enumerate(self.train_loader, 1):
            timer.reset()
            self.optimizer.zero_grad()
            batch_input = move_to_cuda(inputs)
            target = move_to_cuda(targets)
            batch_input = tuple2dict(batch_input, ["char_id", "word_id", "char_mask", "word_mask"])
            target = tuple2dict(target, ["label"])
            label = target["label"]
            start_time = time.time()
            output = self.model(**batch_input)
            loss, loss_items = calc_final_loss(outputs=output, targets=label, mask=None)
            loss.backward()
            # update
            self.optimizer.step()
            self.main_scheduler.step(epoch + batch_idx / len(self.train_loader))
            end_time = time.time()
            total_time.append(end_time - start_time)
            curr_lr = self.main_scheduler.get_last_lr()[0]
            time_interval = timer.elapsed_interval
            metricsContainer.update("loss", loss_items)
            metricsContainer.update("train_time", time_interval)

            if batch_idx % self.config.display_interval == 0:
                self._export_log(epoch, last_total_step + batch_idx, batch_idx, curr_lr,
                                 metricsContainer.calculate_average("loss"),
                                 metricsContainer.calculate_average("train_time"))

        if batch_idx % self.config.display_interval != 0:
            self._export_log(epoch, last_total_step + batch_idx, batch_idx,
                             self.main_scheduler.get_last_lr()[0],
                             metricsContainer.calculate_average("loss"),
                             metricsContainer.calculate_average("train_time"))
        print("Train Avg. Time: ", sum(total_time) / len(total_time))
        return batch_idx + last_total_step

    def _export_log(self, epoch, total_step, batch_idx, lr, loss_meter, time_meter):
        msg = 'Epoch {}, Batch ({} / {}), lr = {:.5f}, '.format(epoch, batch_idx,
                                                                len(self.train_loader), lr)
        for k, v in loss_meter.items():
            msg += '{} = {:.4f}, '.format(k, v)
        remaining = len(self.train_loader) - batch_idx
        msg += '{:.3f} s/batch ({}s left)'.format(time_meter, int(time_meter * remaining))
        print(msg + "\n")
        sys.stdout.flush()
        loss_meter.update({"epoch": epoch, "batch": total_step, "lr": lr})

    def save_model(self, path, epoch):
        state_dict = {
            'epoch': epoch,
            'config': self.config,
            'model_parameters': self.model.module.state_dict(),
        }
        torch.save(state_dict, path)
        print('save models to {}, epoch {}.'.format(path, epoch))

    def load_model(self, path):
        state_dict = torch.load(path)
        self.initial_epoch = state_dict['epoch']
        self.main_scheduler.step(self.initial_epoch)
        parameters = state_dict['model_parameters']
        self.model.module.load_state_dict(parameters)
        print('load models from {}, epoch {}.'.format(path, self.initial_epoch))

    def train(self, start_epoch=0):
        best_result, best_criterion, best_epoch = (), -float('inf'), -1
        total_step = start_epoch * len(self.train_loader)
        for epoch in range(start_epoch, self.config["max_epoch"] + 1):
            saved_path = os.path.join(self.model_saved_path, 'models-{}.pt'.format(epoch))
            total_step = self._train_one_epoch(epoch, total_step)
            self.save_model(saved_path, epoch)
            print("Eval for Eval dataset.")
            eval_results = self.eval(epoch, "eval")
            print("Eval for Test dataset.")
            test_results = self.eval(epoch, "test")
            if eval_results is None or test_results is None:
                continue
            if eval_results[0] > best_criterion:
                best_result = test_results[1:]
                best_criterion = eval_results[0]
                best_epoch = epoch
            print('=' * 60)
        print('-' * 120)
        print('Done.')
        print("Best Result:")
        self._print_metrics(best_epoch, best_result, "Result")


    def _print_metrics(self, epoch, metrics, action):
        msg = "{} Epoch {}: ".format(action, epoch)
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                msg += '{} = {:.4f} | '.format(k, v)
        else:
            msg += str(metrics)
        print(msg)
        sys.stdout.flush()

    def eval(self, epoch, data):
        if data == "eval":
            data_loader = self.eval_loader
        elif data == "test":
            data_loader = self.test_loader
        else:
            raise NotImplementedError
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader, 1):
                batch_input = move_to_cuda(inputs)
                target = move_to_cuda(targets)
                batch_input = tuple2dict(batch_input, ["char_id", "word_id", "char_mask", "word_mask"])
                target = tuple2dict(target, ["label"])
                output = self.model(**batch_input)
                mask = None
                label = target["label"]
                loss, _ = calc_final_loss(outputs=output, targets=label, mask=mask)
                metrics = calc_final_metric(output=move_to_cpu(output), target=move_to_cpu(label), mask=mask)
                metrics["neg_loss"] = -loss.item()
                for key, metric in metrics.items():
                    metricsContainer.update(key, metric)
                if batch_idx % (self.config["display_interval"] * 10) == 0:
                    print("Reflection:", targets[-1][0])
                    print("Trgs: ", output[0].max(-1)[1].cpu().numpy())
                    print("Real: ", target[0].cpu().numpy())
        metric = metricsContainer.calculate_average("item", reset=True)
        neg_loss = metricsContainer.calculate_average("neg_loss", reset=True)
        self._print_metrics(epoch, metric, data)
        return neg_loss, metric
