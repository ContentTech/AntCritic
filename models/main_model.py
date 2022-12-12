import torch
from sentence_transformers import SentenceTransformer
from torch import nn
from transformers import BertModel

from models import FirstStageModel
from models.encoder import TrmBasedEncoder #, GraphBasedEncoder
from utils.helper import masked_operation


class MainModel(nn.Module):
    def __init__(self, init_dim, dim, mark_num, dropout, module_cfg, use_word, **kwargs):
        super().__init__()
        if module_cfg["name"] == "Transformer":
            self.encoder = TrmBasedEncoder(init_dim, dim, mark_num, dropout, module_cfg["transformer"])
        # elif module_cfg["name"] == "Graph":
        #     self.encoder = GraphBasedEncoder(init_dim, dim, mark_num, dropout, module_cfg["graph"])
        else:
            raise NotImplementedError

    def forward(self, sentence_embedding, sentence_mask, coarse_logit, paragraph_order,
                sentence_order, font_size, style_mark):
        """
        :param sentence_embedding: Float, in (B, L, D)
        :param sentence_mask: Int, in (B, L), 1: valid, 0: invalid
        :param coarse_logit: Float, in (B, L, 3)
        :param paragraph_order: Int, in (B, L)
        :param sentence_order: Int, in (B, L)
        :param font_size: Int, in (B, L)
        :param style_mark: Int, in (B, L, M), 1: dressed by i-th mark, 0: not dressed
        :return:
        """
        features, results = self.sentence_encoder(sentence_emb, passage_mask, para_order, sent_order,
                                                  font_size, style_mark, init_pred)
        return results  # (B, L, 3)