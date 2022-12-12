from functools import reduce

import torch
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_clones
# from torch_geometric import nn as graph_nn
# from torch_geometric.nn import SAGEConv
from torch import nn
# from torch_geometric.data import Data

# from models.graph_model import GraphComponent
from models.modules.dynamic_rnn import DynamicGRU
from models.modules.layers import CrossGate
from models.modules.transformer import ArbitraryPositionEncoder, EncoderLayer
from utils.calculator import max_min_norm


class InitialEncoder(nn.Module):
    def __init__(self, dim, mark_num, mode, max_len, dropout):
        super().__init__()
        self.mark_num = mark_num
        self.mode = mode
        # self.paragraph_pos_encoder = ArbitraryPositionEncoder(dim // 3)
        # self.sentence_pos_encoder = ArbitraryPositionEncoder(dim // 3)
        self.all_pos_encoder = ArbitraryPositionEncoder(dim)
        self.paragraph_pos_encoder = nn.Embedding(max_len + 1, dim)
        self.sentence_pos_encoder = nn.Embedding(max_len + 1, dim)
        # self.all_pos_encoder = nn.Embedding(max_len + 1, dim)
        self.font_embedding = nn.Embedding(2, dim // 2)
        print('mark_num: ', mark_num)
        self.style_embedding = nn.Embedding(mark_num, dim // 2)
        self.input_linear = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        self.style_gate = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.Sigmoid()
        )

    def forward(self, sentence_data, para_order, sent_order, font_size, style_mark):
        batch_size, max_len = para_order.size()
        para_pe = self.paragraph_pos_encoder(para_order.long())
        sent_pe = self.sentence_pos_encoder(sent_order.long())
        if self.mode == "all":
            overall_pe = para_pe + sent_pe
        else:
            overall_pe = self.all_pos_encoder(torch.arange(max_len).long().unsqueeze(0).to(para_pe.device))
        sentence_emb = self.input_linear(sentence_data)
        style_emb = torch.mm(style_mark.reshape(-1, self.mark_num).float(),
                             self.style_embedding.weight)
        # (B * L, M) * (M, D) -> (B * L, D) -> (B, L, D)
        font_size = torch.cat((font_size == 1, font_size == 2), dim=-1)
        font_emb = torch.mm(font_size.reshape(-1, 2).float(), self.font_embedding.weight)

        # (B, L) -> (B, L, 2) -> (B * L, 2)
        overall_style = torch.cat((style_emb, font_emb), dim=-1).reshape_as(sentence_emb)
        if self.mode == "text":
            overall_style = torch.zeros_like(overall_style)
        else:
            sentence_emb = (1 + self.style_gate(torch.cat((overall_style, sentence_emb), dim=-1))) * sentence_emb
        return sentence_emb, overall_pe, overall_style

def pessimistic_combine(*inputs):
    return reduce(lambda x, y: (x + y) if x is not None and y is not None else None, inputs)

def optimistic_combine(*inputs):
    return reduce(lambda x, y: (x + y), [item for item in inputs if item is not None])


class LinearBasedEncoder(nn.Module):
    def __init__(self, dim, target_dim, mark_num, max_len, dropout, mode, mlp_cfg):
        super().__init__()
        self.target_dim = target_dim
        self.initial_encoder = InitialEncoder(dim, mark_num, mode, max_len, dropout)
        self.out_norm = nn.LayerNorm(dim)
        self.out_linear = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, target_dim)
        )

    def forward(self, sentence_data, sentence_mask, para_order, sent_order, font_size, style_mark, coarse_logit):
        sent_emb, pos_emb, style_emb = self.initial_encoder(sentence_data, para_order, sent_order, font_size, style_mark)
        prediction = self.out_linear(self.out_norm(sent_emb))
        return sent_emb, pos_emb, prediction  # (B, L, D) / (B, L, T)


class GRUBasedEncoder(nn.Module):
    def __init__(self, dim, target_dim, mark_num, max_len, dropout, mode, gru_cfg):
        super().__init__()
        self.target_dim = target_dim
        self.initial_encoder = InitialEncoder(dim, mark_num, mode, max_len, dropout)
        self.encoder = DynamicGRU(dim, dim // 2, gru_cfg["layer_num"], batch_first=True,
                                  dropout=dropout, bidirectional=True)
        self.out_norm = nn.LayerNorm(dim)
        self.out_linear = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, target_dim)
        )

    def forward(self, sentence_data, sentence_mask, para_order, sent_order, font_size, style_mark, coarse_logit):
        sent_emb, pos_emb, style_emb = self.initial_encoder(sentence_data, para_order, sent_order, font_size, style_mark)
        final_emb = self.encoder(sent_emb, sentence_mask.sum(-1))[0]
        prediction = self.out_linear(self.out_norm(final_emb))
        return final_emb, pos_emb, prediction  # (B, L, D) / (B, L, T)


class TrmBasedEncoder(nn.Module):
    def __init__(self, dim, target_dim, mark_num, max_len, dropout, mode, trm_cfg):
        super().__init__()
        self.target_dim = target_dim
        self.mode = mode
        self.initial_encoder = InitialEncoder(dim, mark_num, mode, max_len, dropout)
        self.encoder = _get_clones(EncoderLayer(dim, trm_cfg["head_num"], dim * 4, dropout), trm_cfg["layer_num"])
        self.projection = nn.Embedding(target_dim, dim)
        self.layer_num = trm_cfg["layer_num"]
        self.out_norm = nn.LayerNorm(dim)
        self.out_linear = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, target_dim)
        )

    def iterative_forward(self, sentence_embedding, sentence_mask, pos_embedding, init_prediction):
        prediction = F.gumbel_softmax(init_prediction, hard=False)
        for layer_idx in range(self.layer_num):
            pred_embedding = torch.mm(prediction.reshape(-1, self.target_dim),
                                      self.projection.weight).reshape_as(sentence_embedding)
            # Use predictions of each stage to act as a labelled token
            sentence_embedding, _ = self.encoder[layer_idx](inputs=sentence_embedding,
                                                            padding_mask=sentence_mask == 0,
                                                            pos=pessimistic_combine(pos_embedding,
                                                                                    pred_embedding))
            prediction = F.gumbel_softmax(self.out_linear(self.out_norm(sentence_embedding)), hard=False)
        return sentence_embedding

    def straight_forward(self, sentence_embedding, sent_mask, pos_embedding, init_prediction):
        # sentence_embedding = optimistic_combine(sentence_embedding, style_embedding)
        for layer_idx in range(self.layer_num):
            # if self.mode == "weighted":
            #     weight = 1 - prediction.softmax(-1).unbind(-1)[0].unsqueeze(-1)  # (B, L, 1)
            #     normed_weight = max_min_norm(weight, dim=1)
            #     sent_emb = sent_emb * (1 + normed_weight)
            sentence_embedding, _ = self.encoder[layer_idx](inputs=sentence_embedding, padding_mask=sent_mask == 0,
                                                            pos=pos_embedding)
        return sentence_embedding

    def forward(self, sentence_data, sentence_mask, para_order, sent_order, font_size, style_mark, coarse_logit):
        sent_emb, pos_emb, style_emb = self.initial_encoder(sentence_data, para_order, sent_order, font_size, style_mark)
        final_emb = self.straight_forward(sent_emb, sentence_mask, pos_emb, coarse_logit)
        # if self.mode in ["weighted", "straight"]:
        #     final_emb = self.straight_forward(sent_emb, sentence_mask, other_emb, coarse_logit)
        # else:
        #     final_emb = self.iterative_forward(sent_emb, sentence_mask, other_emb, coarse_logit)
        prediction = self.out_linear(self.out_norm(final_emb))
        return final_emb, pos_emb, prediction  # (B, L, D) / (B, L, T)

# class GraphBasedEncoder(nn.Module):
#     def __init__(self, input_dim, dim, mark_num, dropout, graph_cfg, **kwargs):
#         super().__init__()
#         self.initial_encoder = InitialEncoder(input_dim, dim, mark_num, dropout)
#         self.graph_encoder = GraphComponent(graph_cfg["name"], graph_cfg["layer_num"], dropout, graph_cfg["component_cfg"])
#         self.out_linear = nn.Sequential(
#             nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout),
#             nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout),
#             nn.Linear(dim, T)
#         )
#
#     def insert_embedding(self, sentence_data, other_embedding, init_prediction):
#         batch_size = sentence_data.size(0)
#         sentence_data[:, -2] = self.global_embedding.reshape(1, 1, -1).repeat_interleave(batch_size, dim=0)
#         sentence_data[:, -1] = self.stop_embedding.reshape(1, 1, -1).repeat_interleave(batch_size, dim=0)
#         other_embedding[:, -2:] = 0
#         init_prediction[:, -2:] = 0
#         return sentence_data + other_embedding, init_prediction
#
#     def forward(self, sentence_data, sentence_mask, para_order, sent_order,
#                 font_size, style_mark, init_prediction, sentence_edges):
#         batch_size, max_len = sentence_mask.size()
#         if init_prediction is None:
#             init_prediction = torch.zeros(batch_size, max_len, T).to(sentence_data.device)
#         sent_emb, other_emb = self.initial_encoder(sentence_data, para_order, sent_order, font_size, style_mark)
#         sent_emb, init_prediction = self.insert_embedding(sent_emb, other_emb, init_prediction)
#         for layer_idx in range(self.graph_encoder.layer_num):
#             sent_emb = self.graph_encoder(sent_emb, sentence_edges, layer_idx)
#         prediction = self.out_linear(self.out_norm(sent_emb))  # Read Out
#         return sent_emb, prediction  # (B, L, D) / (B, L)
#
#
