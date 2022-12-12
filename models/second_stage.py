import copy

import torch
from sentence_transformers import SentenceTransformer
from torch import nn
from transformers import BertModel

from models import FirstStageModel
from models.encoder import TrmBasedEncoder, GRUBasedEncoder, LinearBasedEncoder  # , GraphBasedEncoder
from models.modules.layers import CrossGate
from utils.helper import masked_operation
import torch.nn.functional as F

class Biaffine(nn.Module):
    def __init__(self, dim, target_dim):
        super().__init__()
        self.start_linear = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.end_linear = copy.deepcopy(self.start_linear)
        self.kernel = nn.Parameter(torch.randn(dim + 1, target_dim, dim + 1))

    def forward(self, inputs):
        start_inputs, end_inputs = self.start_linear(inputs), self.end_linear(inputs)
        biased_start = torch.cat((start_inputs, torch.ones_like(start_inputs[..., :1])), dim=-1)
        biased_end = torch.cat((end_inputs, torch.ones_like(end_inputs[..., :1])), dim=-1)
        return torch.einsum('bxi,ioj,byj->bxyo', biased_start, self.kernel, biased_end)
        # [B, L, L, 4]

class SimpleFusion(nn.Module):
    def __init__(self, dim, target_dim):
        super().__init__()
        self.start_linear = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.end_linear = copy.deepcopy(self.start_linear)
        self.final_linear = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.ReLU(),
            nn.Linear(dim, target_dim)
        )

    def forward(self, inputs):
        # in [B, L, D]
        # batch_size, max_len, dim = inputs.size()
        start = self.start_linear(inputs).unsqueeze(2)
        end = self.end_linear(inputs).unsqueeze(1)
        return self.final_linear(torch.cat((
            start + end, start * end
        ), dim=-1))


class SecondStageModel(nn.Module):
    def __init__(self, input_dim, dim, mark_num, dropout, max_len, mode, module_cfg, **kwargs):
        super().__init__()
        self.char_input = nn.Sequential(
            nn.Linear(input_dim, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        self.word_input = copy.deepcopy(self.char_input)
        if module_cfg["name"] == "Transformer":
            self.encoder = TrmBasedEncoder(dim, 3, mark_num, max_len, dropout, mode, module_cfg["module"])
        elif module_cfg["name"] == "GRU":
            self.encoder = GRUBasedEncoder(dim, 3, mark_num, max_len, dropout, mode, module_cfg["module"])
        elif module_cfg["name"] == "Linear":
            self.encoder = LinearBasedEncoder(dim, 3, mark_num, max_len, dropout, mode, module_cfg["module"])
        # elif module_cfg["name"] == "Graph":
        #     self.encoder = GraphBasedEncoder(dim, mark_num, dropout, module_cfg["graph"])
        else:
            raise NotImplementedError
        self.major_predictor = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, 1)
        )
        self.biaffine = Biaffine(dim, 4)
        self.cross_gate = CrossGate(dim)
        self.fusion = SimpleFusion(dim, 4)

    def forward(self, sentence_embedding, sentence_mask, coarse_logit, paragraph_order,
                sentence_order, font_size, style_mark):
        """
        :param sentence_embedding: Float, in (B, L, 2 * D)
        :param sentence_mask: Int, in (B, L), 1: valid, 0: invalid
        :param coarse_logit: Float, in (B, L, 3)
        :param paragraph_order: Int, in (B, L)
        :param sentence_order: Int, in (B, L)
        :param font_size: Int, in (B, L)
        :param style_mark: Int, in (B, L, M), 1: dressed by i-th mark, 0: not dressed
        :return:
        """
        char_embedding, word_embedding = sentence_embedding.chunk(chunks=2, dim=-1)
        char_feature, word_feature = self.char_input(char_embedding), self.word_input(word_embedding)
        # fused_feature = torch.cat((char_feature, word_feature), dim=-1)
        fused_feature = self.cross_gate(char_feature, word_feature)
        semantic_feature, pos_feature, label_logits = self.encoder(fused_feature, sentence_mask, paragraph_order,
                                                                   sentence_order, font_size, style_mark, coarse_logit)
        grid_logits = self.biaffine(semantic_feature + pos_feature)
        major_logit = self.major_predictor(semantic_feature).squeeze(-1)
        return {
            "grid_logit": grid_logits,
            "label_logit": label_logits,
            "major_logit": major_logit
        }  # (B, L, L, 4), (B, L, 3)
