import torch
from torch import nn
import torch.nn.functional as F

class TanhAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # self.dropout = nn.Dropout(dropout)
        self.ws1 = nn.Linear(d_model, d_model, bias=True)
        self.ws2 = nn.Linear(d_model, d_model, bias=False)
        self.wst = nn.Linear(d_model, 1, bias=False)

    def reset_parameters(self):
        self.ws1.reset_parameters()
        self.ws2.reset_parameters()
        self.wst.reset_parameters()

    def forward(self, x, memory, memory_mask=None):
        item1 = self.ws1(x)  # [nb, len1, d]
        item2 = self.ws2(memory)  # [nb, len2, d]
        # print(item1.shape, item2.shape)
        item = item1.unsqueeze(2) + item2.unsqueeze(1)  # [nb, len1, len2, d]
        S_logit = self.wst(torch.tanh(item)).squeeze(-1)  # [nb, len1, len2]
        if memory_mask is not None:
            memory_mask = memory_mask.unsqueeze(1)  # [nb, 1, len2]
            S_logit = S_logit.masked_fill(memory_mask == 0, -1e4)
        return S_logit  # [nb, len1, len2]


class CrossGate(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear_transformation1 = nn.Linear(input_dim, input_dim, bias=True)
        self.linear_transformation2 = nn.Linear(input_dim, input_dim, bias=True)
        self.final = nn.Linear(input_dim * 2, input_dim, bias=True)

    def forward(self, x1, x2):
        g1 = torch.sigmoid(self.linear_transformation1(x1))
        h2 = g1 * x2
        g2 = torch.sigmoid(self.linear_transformation2(x2))
        h1 = g2 * x1
        return self.final(torch.cat([h1, h2], dim=-1))
        # return torch.cat([h1, h2], dim=-1)
        # return h1, h2
