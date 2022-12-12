import argparse

from config.config import get_cfg_defaults
from runner import SecondStageRunner

import warnings

# from runner.test_runner import TestRunner

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--start-from', type=int, default=-1)
    parser.add_argument('--eval-epoch', type=int, default=-1)
    parser.add_argument('--test-file', type=str, default=None)
    return parser.parse_args()

import os
BASEDIR = os.path.dirname(os.path.realpath(__file__))
print('BASEDIR: ', BASEDIR)
if __name__ == "__main__":
    """
    CUDA_VISIBLE_DEVICES=0,1 /mnt/fengyao.hjj/miniconda3/envs/antmmf/bin/python \
    /mnt/fengyao.hjj/argument_mining/second_main.py \
    --config use_gru \
    --eval-epoch 7 \
    --test-file /mnt/fengyao.hjj/transformers/data/topic_pgc/data/test_2.0428_2022042702000000348001.hdf5
    --test-file /mnt/fengyao.hjj/transformers/data/pgc/0506/test_2.hdf5

    --test-file /mnt/fengyao.hjj/transformers/data/topic_pgc/data/test_2.0427_2022042702000000348601.hdf5
    --test-file /mnt/fengyao.hjj/transformers/data/topic_pgc/data/test_2.0425_2022042402000000344601.hdf5
    --test-file /mnt/fengyao.hjj/transformers/data/topic_pgc/data/test_2.hdf5
    """
    args = parse_args()
    cfg = get_cfg_defaults()
    if args.config is not None:
        cfg.merge_from_file(os.path.join(BASEDIR, 'config', args.config + ".yaml"))
    cfg.freeze()
    runner = SecondStageRunner(cfg)
    if args.eval_epoch != -1:
        runner.load_model(cfg.saved_path + "/models-{}.pt".format(args.eval_epoch))
        print('load model: ', cfg.saved_path + "/models-{}.pt".format(args.eval_epoch))
        # runner.eval(args.eval_epoch, "eval")
        if args.test_file:
            runner.test(args.eval_epoch, args.test_file)
    else:
        if args.start_from >= 0:
            runner.load_model(cfg.saved_path + "/models-{}.pt".format(args.start_from))
        runner.train()
