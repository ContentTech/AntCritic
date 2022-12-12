import torch
from torch import nn

from models.transformer import TransEncoder, PositionEncoder


class FirstStage(nn.Module):
    def __init__(self, dim, mark_num, head_num, num_layers, dropout):
        super().__init__()
        self.mark_num = mark_num
        self.encoder = TransEncoder(dim, head_num, num_layers, dim * 4, dropout)
        self.paragraph_pos_encoder = PositionEncoder(dim)
        self.sentence_pos_encoder = PositionEncoder(dim)
        self.font_embedding = nn.Embedding(3, dim)  # Larger, Common, Smaller
        self.style_embedding = nn.Parameter(torch.randn(mark_num, dim))
        # Font, FG-Color, BG-Color, Bold, Quote

    def forward(self, sentence_data, sentence_mask, para_order, sent_order, font_size, style_mark):
        """
        :param sentence_data: Float, in (B, L, D)
        :param sentence_mask: Int, in (B, L), 1: valid, 0: invalid
        :param para_order: Int, in (B, L)
        :param sent_order: Int, in (B, L)
        :param font_size: Int, in (B, L)
        :param style_mark: Int, in (B, L, M), 1: dressed by i-th mark, 0: not dressed
        :return:
        """
        para_pe = self.paragraph_pos_encoder(para_order)
        sent_pe = self.sentence_pos_encoder(sent_order)
        style_emb = torch.mm(style_mark.reshape(-1, self.mark_num), self.style_embedding).reshape_as(sentence_data)
        # (B * L, M) * (M, D) -> (B * L, D)
        font_emb = self.font_embedding(font_size)
        overall_emb = para_pe + sent_pe + style_emb + font_emb
        result = self.encoder(inputs=sentence_data, padding_mask=sentence_mask == 0, pos=overall_emb)
        return result