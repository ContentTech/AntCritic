import argparse

from config.config import get_cfg_defaults
from runner import FirstStageRunner

import warnings

# from runner.test_runner import TestRunner

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--start-from', type=int, default=-1)
    parser.add_argument('--eval-epoch', type=int, default=-1)
    return parser.parse_args()


BASEDIR = '/mnt/fengyao.hjj/argument_mining/'
if __name__ == "__main__":
    args = parse_args()
    cfg = get_cfg_defaults()
    if args.config is not None:
        cfg.merge_from_file(BASEDIR + "config/" + args.config + ".yaml")
    cfg.freeze()
    runner = FirstStageRunner(cfg)
    # runner.load_model("checkpoints/new_char/best.pt")
    # runner.eval(args.eval_epoch, "eval")
    # runner.eval(args.eval_epoch, "test")
    if args.eval_epoch != -1:
        runner.load_model(BASEDIR + cfg.saved_path + "/models-{}.pt".format(args.eval_epoch))
        runner.eval(args.eval_epoch, "eval")
        runner.eval(args.eval_epoch, "test")
    else:
        if args.start_from >= 0:
            runner.load_model(cfg.saved_path + "/models-{}.pt".format(args.start_from))
        runner.train()
