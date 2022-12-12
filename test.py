# import torch
#
#
# def selective_pooling(features, select_idx):
#     # features: in (B, L, D), select_idx: in (B, T)
#     batch_idx = torch.arange(features.size(0)).unsqueeze(-1)
#     all_features = features[batch_idx, select_idx]
#     return all_features
#
# def trivial(features, select_idx):
#     # features: in (B, L, D), select_idx: in (B, T)
#     batch_size = features.size(0)
#     dim = features.size(-1)
#     selection_num = select_idx.size(-1)
#     results = torch.zeros(batch_size, selection_num, dim)
#     for batch_idx in range(batch_size):
#         for idx, tgt in enumerate(select_idx[batch_idx]):
#             results[batch_idx, idx] = features[batch_idx, tgt]
#     return results
#
#
# def auto_test():
#     batch_size, len, dim = 8, 20, 16
#     for _ in range(10):
#         features = torch.randn(batch_size, len, dim)
#         _, idx = torch.randn(batch_size, len).topk(dim=-1, k=5)
#         a = selective_pooling(features, idx)
#         b = trivial(features, idx)
#         assert (a == b).all(), "Not the same!"
#     print("Over!")
#
# auto_test()


import time
import pyperclip

while True:
    time.sleep(1)
    data = pyperclip.paste()
    if data is not None:
        pyperclip.copy(data.replace("\n", " ").replace("- ", ""))
