import os.path

import h5py
# import torch_geometric.transforms as T
from torch.utils.data import Dataset
import numpy as np

#
# def construct_graph(paragraph_order, sentence_order, neighbor_size, max_len):
#     # para_order & sent_order in L
#     cnt_len = len(paragraph_order)
#     outer_edge, inner_edge = [], []
#     for sent_idx in range(cnt_len):
#         # 1. Construct inner graph
#         inner_pointer = sent_idx + 1
#         while (inner_pointer < cnt_len and paragraph_order[inner_pointer] == paragraph_order[sent_idx]):
#             inner_edge.append((sent_idx, inner_pointer))
#             inner_pointer += 1
#         # 2. Construct outer graph
#         if sentence_order[sent_idx] != 1:
#             continue
#         outer_pointer = inner_pointer
#         while (outer_pointer < cnt_len and paragraph_order[outer_pointer] - paragraph_order[sent_idx] <= neighbor_size):
#             outer_edge.append((sent_idx, outer_pointer))
#             outer_group = paragraph_order[outer_pointer]
#             while (outer_pointer < cnt_len and paragraph_order[outer_pointer] == outer_group):
#                 outer_pointer += 1
#     return [*outer_edge, *inner_edge]  # [2, edge_num]

def adjust_label(label, grid):
    max_len = grid.shape[-1]
    major_sentence = np.where(grid[max_len - 2] != 0)[0]
    label[major_sentence] = 3
    return label
# 0: Others, 1: Claim, 2: Premise, 3: Major

BASEDIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_root = os.path.join(BASEDIR, "antcritic")

class SecondStageDataset(Dataset):
    def __init__(self, split, config, test_data_file=None):
        super().__init__()
        self.config = config
        if split == 'test' and test_data_file is not None:
            self.embedding_path = test_data_file
            self.annotation_path = ''
        else:
            self.embedding_path = data_root + f"/{split}_2.hdf5"
            self.annotation_path = data_root + f"/{split}_2.hdf5"
        # self.raw_embedding_path = data_root + f"/{split}_4.hdf5"
        with h5py.File(self.embedding_path, "r") as f:
            self.length = f.attrs["size"]

    def _open_hdf5(self):
        print('_open_hdf5: ', self.embedding_path)
        self.embedding = h5py.File(self.embedding_path, 'r')
        if os.path.exists(self.annotation_path):
            self.annotation = h5py.File(self.annotation_path, 'r')
        else:
            self.annotation = None
        # self.raw_embedding = h5py.File(self.raw_embedding_path, 'r')



    def __getitem__(self, item):
        if not hasattr(self, 'embedding'):
            self._open_hdf5()
        inputs, targets = [], []
        input_keys = ["embedding", "sentence_mask", "paragraph_order", "sentence_order",
                      "font_size", "style_mark", "coarse_logit"]
        target_keys = ["grid", "reflection", "label", "is_major"]
        for input_key in input_keys:
            inputs.append(self.embedding[input_key][item])
        # inputs[0] = self.raw_embedding["embedding"][item]

        for target_key in target_keys:
            if self.annotation:
                targets.append(self.annotation[target_key][item])
            else:
                targets.append('')

        # targets[-1] = adjust_label(targets[-1], targets[0])
        # edges = construct_graph(inputs[2], inputs[3], self.config["neighbor_size"], self.config["max_len"])
        # inputs.append(edges)
        return inputs, targets  #, edges

    def __len__(self):
        return self.length
