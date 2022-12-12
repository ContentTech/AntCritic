import math
import os
from yacs.config import CfgNode as CN

_c = CN()
BASEDIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print('BASEDIR: ', BASEDIR)
_c.display_interval = 10
# _c.saved_path = os.path.join(BASEDIR, "checkpoints/char")
# _c.saved_path = os.path.join(BASEDIR, "checkpoints/word")
_c.saved_path = os.path.join(BASEDIR, "checkpoints/second_gru_nofl_8")

_c.max_epoch = 12
_c.text_embd_dims = 768
_c.max_sentence_num = 400
_c.model_dim = 384
_c.root = BASEDIR


_c.dataset = CN()
_c.dataset.batch_size = 8
_c.dataset.neighbor_size = 5
_c.dataset.max_len = _c.max_sentence_num

_c.model = CN()
_c.model.name = "SecondStageModel"  # CharBasedModel
# _c.model.name = "FirstStageModel"  # CharBasedModel
_c.model.input_dim = _c.text_embd_dims
_c.model.dim = _c.model_dim
_c.model.mark_num = 5
_c.model.dropout = 0.4
_c.model.max_len = _c.max_sentence_num
_c.model.use_word = False
_c.model.mode = "all"  # "text"

_c.model.module_cfg = CN()
_c.model.module_cfg.char_path = _c.root + "pretrained_model/FinBERT_L-12_H-768_A-12_pytorch"
_c.model.module_cfg.word_path = _c.root + "pretrained_model/paraphrase-xlm-r-multilingual-v1"
_c.model.module_cfg.name = "GRU"  # ["Transformer", "Graph", "GRU"]
_c.model.module_cfg.module = CN()
_c.model.module_cfg.module.head_num = 4
_c.model.module_cfg.module.layer_num = 3


_c.metric = CN()

_c.optimizer = CN()
_c.optimizer.lr = 1e-4
_c.optimizer.weight_decay = 5e-5
_c.optimizer.T_max = _c.max_epoch
_c.optimizer.warmup_epoch = 1

_c.optimizer.loss = CN()
_c.optimizer.loss.label = 1
_c.optimizer.loss.grid = 1
_c.optimizer.loss.major = 1
_c.optimizer.loss.margin = 0.2


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _c.clone()
