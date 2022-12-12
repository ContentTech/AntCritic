import torch
from torch import nn

from models.modules.layers import TanhAttention
from utils.helper import masked_operation


class PointerNetwork(nn.Module):
    def __init__(self, dim, max_iter):
        super().__init__()
        self.attention = TanhAttention(dim)
        self.bilinear = nn.Bilinear(dim, dim, dim)
        self.memory_bank = nn.GRUCell(dim, dim)
        self.max_iter = max_iter

    def forward(self, parent_embedding, sibling_embedding, all_embeddings, all_masks, gold_sequence=None):
        """
        :param parent_embedding: FloatTensor, in (B, D)
        :param sibling_embedding: FloatTensor, in (B, D)
        :param all_embeddings: in (B, L, D)
        :param all_masks: in (B, L)
        :param gold_sequence: IntTensor, in (B, T)
        :return:
        """
        assert gold_sequence is not None or not self.training, "Invalid Inputs!"
        batch_size = parent_embedding.size(0)
        results = []
        for iter in self.max_iter:
            query_embedding = self.bilinear(parent_embedding, sibling_embedding).unsqueeze(1)  # (B, 1, D)
            logits = self.attention(query_embedding, all_embeddings, all_masks).squeeze(1)  # [B, L]
            batch_idx = torch.arange(batch_size)
            prob = logits.softmax(-1)
            if self.is_training:
                target_idx = gold_sequence[:, iter]
                target_prob = prob[batch_idx, target_idx]
                results.append(target_prob)
            else:
                target_idx = prob.max(-1)[1]
                results.append(target_idx)
            sibling_embedding = self.memory_bank(all_embeddings[batch_idx, target_idx], sibling_embedding)
        return torch.stack(results, dim=1)  # (B, T)

def selective_pooling(features, select_idx):
    # features: in (B, L, D), select_idx: in (B, T)
    batch_idx = torch.arange(features.size(0)).unsqueeze(-1)
    all_features = features[batch_idx, select_idx]
    all_masks = select_idx != 1
    return masked_operation(all_features, all_masks, 1, "mean")

class TwoStagePointerReasoning(nn.Module):
    def __init__(self, dim, max_claim_num, max_premise_num):
        super().__init__()
        self.global2claim = PointerNetwork(dim, max_claim_num)
        self.claim2premise = PointerNetwork(dim, max_premise_num)
        self.query_claim = nn.Parameter(torch.randn(1, 1, dim))
        self.query_premise = nn.Parameter(torch.randn(1, 1, dim))

    def extract_features(self, feature, index):
        # feature: (B, L, D), index: (B, N, T)
        batch_size = feature.size(0)
        batch_idx = torch.arange(batch_size).reshape(-1, 1, 1)
        aggregation = masked_operation(feature=feature[batch_idx, index],
                                       mask=index != -1, dim=2, operation="mean")
        return aggregation
        # (B, L, D) & (B, N, T) -> (B, N, T, D) -> (B, N, D)

    def forward(self, sentence_feat, sentence_mask, gold_claim=None, gold_claim_mask=None,
                gold_premise=None, gold_premise_mask=None):
        """
        :param sentence_feat: Encoded Feature, Float, in (B, L, D)
        :param sentence_mask: Sentence Mask, Int, in (B, L)
        :param gold_claim: int, in (B, N_claim, T_claim)
        :param gold_premise: int, in (B, N_claim, N_premise, T_premise)
        :return:
        """
        global_feat = sentence_feat[:, -2]
        if gold_claim is not None and gold_premise is not None:
            gold_claim_head, gold_premise_head = gold_claim[:, :, 0], gold_premise[:, :, 0]
        else:
            gold_claim_head, gold_premise_head = None, None
        claim_pointer = self.global2claim(global_feat, self.query_claim, sentence_feat, sentence_mask, gold_claim_head)
        premise_pointer = self.claim2premise(global_feat, self.query_premise, sentence_feat, sentence_mask, gold_premise_head)
        pass






class OneStagePointerReasoning(nn.Module):
    def __init__(self, dim, max_claim_num, max_premise_num, max_claim_len, max_premise_len):
        super().__init__()
        self.claim_ext = PointerNetwork(dim, max_claim_len)
        self.premise_ext = PointerNetwork(dim, max_premise_len)
        self.global2claim = PointerNetwork(dim, max_claim_num)
        self.claim2premise = PointerNetwork(dim, max_premise_num)
        self.query_claim = nn.Parameter(torch.randn(1, 1, dim))
        self.query_premise = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, sentence_feat, sentence_mask, gold_claim=None, gold_premise=None):
        """
        :param sentence_feat: Encoded Feature, float, in (B, L + 2, D), first: <CLS>, end: <EOP>
        :param sentence_mask: Sentence Mask, in (B, L)
        :param gold_claim: int, in (B, N_claim, T_claim)
        :param gold_premise: int, in (B, N_premise, T_premise)
        :return:
        """
        batch_size = sentence_feat.size(0)
        global_feat = sentence_feat[:, 0]
        gold_claim_head, gold_premise_head = gold_claim[:, :, 0], gold_premise[:, :, 0]
        initial_claim = self.initial_sub.repeat_interleave(batch_size, dim=0)
        initial_premise = self.query_premise.repeat_interleave(batch_size, dim=0)
        pass






