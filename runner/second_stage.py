import math
import os
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import models
from dataset.second_stage import SecondStageDataset
from loss import calc_second_loss
from metrics import calc_second_metric
from runner.optimizer import GradualWarmupScheduler
from utils.container import metricsContainer
from utils.helper import move_to_cuda, move_to_cpu
from utils.processor import tuple2dict
from utils.timer import Timer
import pandas as pd
torch_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'torch_device: {torch_device}')
if 'cuda' not in torch_device:
    os.environ['OMP_NUM_THREADS'] = '1'

# 显示所有列
# pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 30)


class SecondStageRunner:
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
        if 'cuda' in torch_device:
            self.device_ids = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
            print('GPU: {}'.format(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
        self.initial_epoch = 0

    def _init_dataset(self, config):
        train_dataset = SecondStageDataset("train", config)
        eval_dataset = SecondStageDataset("val", config)
        test_dataset = SecondStageDataset("test", config)
        self.train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                                       num_workers=8, pin_memory=False)
        self.eval_loader = DataLoader(eval_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=1,
                                      pin_memory=False)
        self.test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=1,
                                      pin_memory=False)

    def _init_model(self, model_config):
        self.model = getattr(models, model_config["name"])(**model_config).to(torch_device)
        if 'cuda' in torch_device:
            self.device_ids = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
            self.model = nn.DataParallel(self.model, device_ids=self.device_ids)
        else:
            self.model = nn.DataParallel(self.model)

    def _init_optimizer(self, config):
        self.optimizer = torch.optim.AdamW(list(self.model.parameters()),
                                           lr=config["lr"], weight_decay=config["weight_decay"])

        self.sub_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config["T_max"])
        self.main_scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1,
                                                     total_epoch=config["warmup_epoch"],
                                                     after_scheduler=self.sub_scheduler)
        self.loss_config = config["loss"]

    def _train_one_epoch(self, epoch, last_total_step):
        self.model.train()
        timer = Timer()
        batch_idx = 0
        total_time = []
        for batch_idx, (inputs, targets) in enumerate(self.train_loader, 1):
            self.optimizer.zero_grad()
            if 'cuda' in torch_device:
                batch_input = move_to_cuda(inputs)
                target = move_to_cuda(targets)
            else:
                batch_input = move_to_cpu(inputs)
                target = move_to_cpu(targets)
            batch_input = tuple2dict(batch_input, ["sentence_embedding", "sentence_mask", "paragraph_order",
                                                   "sentence_order", "font_size", "style_mark", "coarse_logit"])
            target = tuple2dict(target, ["grid", "reflection", "label", "is_major"])
            start_time = time.time()
            output = self.model(**batch_input)
            # print('_train_one_epoch output: ', output['grid_logit'].max(-1)[1])
            # print('_train_one_epoch target: ', target['grid'])
            loss, loss_items = calc_second_loss(outputs=output, targets=target,
                                                mask=batch_input["sentence_mask"], config=self.loss_config)
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
        state_dict = torch.load(path, map_location=torch.device(torch_device))
        self.initial_epoch = state_dict['epoch']
        self.main_scheduler.step(self.initial_epoch)
        parameters = state_dict['model_parameters']
        self.model.module.load_state_dict(parameters)
        print('load models from {}, epoch {}.'.format(path, self.initial_epoch))

    def train(self, start_epoch=0):
        best_result, best_criterion, best_epoch = (), -float('inf'), -1
        total_step = start_epoch * len(self.train_loader)
        for epoch in range(start_epoch, self.config["max_epoch"] + 1):
            print("Epoch {} Start! ".format(epoch))
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


    def _print_metrics(self, epoch, metrics, action, namespace=''):
        msg = "{} {} Epoch {}: ".format(namespace, action, epoch)
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                msg += '{} = {:.4f} | '.format(k, v)
        else:
            msg += str(metrics)
        print(msg)
        sys.stdout.flush()

    def eval(self, epoch, data):
        self.model.eval()
        all_pred_sentence, all_real_sentence = [], []
        all_pred_grid, all_real_grid = [], []
        all_major_logits = []
        all_is_major = []
        all_full_label = []
        all_loss, all_num = 0, 0
        if data == "eval":
            data_loader = self.eval_loader
        elif data == "test":
            data_loader = self.test_loader
        else:
            raise NotImplementedError
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader, 1):
                if 'cuda' in torch_device:
                    batch_input = move_to_cuda(inputs)
                    targets = move_to_cuda(targets)
                else:
                    batch_input = move_to_cpu(inputs)
                    targets = move_to_cpu(targets)
                batch_input = tuple2dict(batch_input, ["sentence_embedding", "sentence_mask", "paragraph_order",
                                                       "sentence_order", "font_size", "style_mark", "coarse_logit"])
                targets = tuple2dict(targets, ["grid", "reflection", "label", "is_major"])
                output = self.model(**batch_input)
                mask_1d = batch_input["sentence_mask"]
                mask_2d = mask_1d.unsqueeze(1) * mask_1d.unsqueeze(-1)
                loss, _ = calc_second_loss(outputs=output, targets=targets, mask=mask_1d, config=self.loss_config)
                # all_sentence_first.extend(batch_input["coarse_logit"].max(-1)[1].masked_select(mask_1d == 1))
                all_major_logits.extend(output["major_logit"])
                all_is_major.extend(targets["is_major"])
                all_full_label.extend(targets["label"])
                all_pred_sentence.extend(output["label_logit"].max(-1)[1].masked_select(mask_1d == 1))
                all_real_sentence.extend(targets["label"].masked_select(mask_1d == 1))
                all_pred_grid.extend(output["grid_logit"].max(-1)[1].masked_select(mask_2d == 1))
                all_real_grid.extend(targets["grid"].masked_select(mask_2d == 1))
                batch_size = len(mask_1d)
                all_loss += (-loss.item()) * batch_size
                all_num += batch_size
                if batch_idx % (self.config["display_interval"]) == 0:
                    chosen_idx = random.randint(0, batch_size - 1)
                    length = batch_input["sentence_mask"][chosen_idx].sum()
                    print("Reflection: ", targets["reflection"][chosen_idx].cpu().numpy())
                    # first = (batch_input["coarse_logit"][chosen_idx].max(-1)[1]).cpu().numpy()[:length]
                    second = (output["label_logit"][chosen_idx].max(-1)[1]).cpu().numpy()[:length]
                    real = (targets["label"][chosen_idx]).cpu().numpy()[:length]
                    # print("First:  ", first)
                    print("Pred:", second)
                    print("Real:", real)
                    # print("Pred grid:", output["grid_logit"].max(-1)[1].shape)
                    # print(output["grid_logit"].max(-1))
                    # g = output["grid_logit"].max(-1)[1][chosen_idx]
                    # for i, item in enumerate(g.tolist()):
                    #     print(i, item)
                    # print("Real grid:")
                    # rg = targets["grid"][chosen_idx]
                    # for i, item in enumerate(rg.tolist()):
                    #     print(i, item)
        # first_pred, second_pred = torch.tensor(all_sentence_first), torch.tensor(all_sentence_second)
        pred_label = torch.tensor(all_pred_sentence)
        real_label = torch.tensor(all_real_sentence)
        pred_grid = torch.tensor(all_pred_grid)
        real_grid = torch.tensor(all_real_grid)
        major_logits = torch.stack(all_major_logits, dim=0)
        full_label = torch.stack(all_full_label, dim=0)
        is_major = torch.stack(all_is_major, dim=0)
        # first_metric = calc_second_metric(first_pred, real_label)
        # pred_label, real_label, pred_grid, real_grid
        second_metric = calc_second_metric(pred_label, real_label, pred_grid, real_grid, is_major, full_label, major_logits)
        neg_loss = all_loss / all_num
        # self._print_metrics(epoch, first_metric["item"], data, "First")
        self._print_metrics(epoch, second_metric["label"], data, "Label")
        self._print_metrics(epoch, second_metric["grid"], data, "Grid")
        self._print_metrics(epoch, second_metric["major"], data, "Major")
        return neg_loss, second_metric

    def test(self, epoch, test_file):
        self.model.eval()
        test_dataset = SecondStageDataset("test", self.config, test_data_file=test_file)
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1,
                                      pin_memory=False)
        all_pred_sentence, all_real_sentence = [], []
        all_pred_grid = []
        all_major_logits = []
        all_loss, all_num = 0, 0
        data_loader = self.test_loader
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(data_loader, 1):
                if 'cuda' in torch_device:
                    batch_input = move_to_cuda(inputs)
                else:
                    batch_input = move_to_cpu(inputs)
                batch_input = tuple2dict(batch_input, ["sentence_embedding", "sentence_mask", "paragraph_order",
                                                       "sentence_order", "font_size", "style_mark", "coarse_logit"])
                output = self.model(**batch_input)
                mask_1d = batch_input["sentence_mask"]
                mask_2d = mask_1d.unsqueeze(1) * mask_1d.unsqueeze(-1)
                all_major_logits.extend(output["major_logit"].masked_select(mask_1d == 1))
                # print('major_logit: ', output["major_logit"].masked_select(mask_1d == 1).shape)
                all_pred_sentence.extend(output["label_logit"].max(-1)[1].masked_select(mask_1d == 1))
                n = int(math.sqrt(output["grid_logit"].max(-1)[1].masked_select(mask_2d == 1).shape[0]))
                all_pred_grid.append(output["grid_logit"].max(-1)[1].masked_select(mask_2d == 1).reshape(n, n))
                batch_size = len(mask_1d)
                all_num += batch_size
                # if batch_idx % (self.config["display_interval"]) == 0:
                #     chosen_idx = random.randint(0, batch_size - 1)
                #     length = batch_input["sentence_mask"][chosen_idx].sum()
                #     second = (output["label_logit"][chosen_idx].max(-1)[1]).cpu().numpy()[:length]
        pred_label = torch.tensor(all_pred_sentence)
        major_logits = torch.tensor(all_major_logits)
        pred_grid = []
        grid_map = {0: 'No-Relation', 1: 'Co-occurence', 2: 'Co-reference', 3: 'Affiliation'}
        for g in all_pred_grid:
            for i, item in enumerate(g.tolist()):
                # print(i, item)
                pred_grid.append([f'{i}-{j}-{grid_map[re]}' for j, re in enumerate(item)])
        # print("pred_label:", pred_label, pred_label.shape)
        # print("major_logits:", major_logits, major_logits.shape)
        # print("pred_grid:", len(pred_grid))
        map = {0: 'Others', 1: 'Claim', 2: 'Premise', 3: 'Major'}
        d = {'major': major_logits.tolist(),
             'preds': [map[p] for p in pred_label.tolist()],
             'grid': pred_grid}
        df = pd.DataFrame(data=d)
        df.to_csv('.'.join(test_file.split('.')[:-1] + ['prediction', 'csv']), index=False)
        return pred_label, all_pred_grid, major_logits

 # Result Epoch 22: ({'label': {'macro_f1': 0.304600917597834, 'micro_f1': 0.5161394274665664, 'weighted_f1': 0.45237791409285505, 'all_macro': 0.4204792546703584, 'all_micro': 0.6374749387391401, 'all_weighted': 0.5963196480978951}, 'grid': {'macro_f1': 0.10814810654381485, 'micro_f1': 0.11226900634091057, 'weighted_f1': 0.11451091495378628, 'all_macro': 0.10734466885875847, 'all_micro': 0.1088465578332228, 'all_weighted': 0.10604271813758946}},)