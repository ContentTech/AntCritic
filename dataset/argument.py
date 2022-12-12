import h5py
# import torch_geometric.transforms as T
from torch.utils.data import Dataset

# transform = T.Compose([T.ToUndirected(), T.AddSelfLoops()])

# def construct_graph(para_order, sent_order, neighbor_size, max_len):
#     # para_order & sent_order in L
#     cnt_len = len(para_order)
#     outer_edge, inner_edge = [], []
#     for sent_idx in range(cnt_len):
#         # 1. Construct inner graph
#         inner_pointer = sent_idx + 1
#         while (inner_pointer < cnt_len and para_order[inner_pointer] == para_order[sent_idx]):
#             inner_edge.append((sent_idx, inner_pointer))
#             inner_pointer += 1
#         # 2. Construct outer graph
#         if sent_order[sent_idx] != 1:
#             continue
#         outer_edge.append((max_len - 2, sent_idx))
#         outer_edge.append((max_len - 1, sent_idx))
#         outer_pointer = inner_pointer
#         while (outer_pointer < cnt_len and para_order[outer_pointer] - para_order[sent_idx] <= neighbor_size):
#             outer_edge.append((sent_idx, outer_pointer))
#             outer_group = para_order[outer_pointer]
#             while (outer_pointer < cnt_len and para_order[outer_pointer] == outer_group):
#                 outer_pointer += 1
#     return [*outer_edge, *inner_edge]


class ArgumentDataset(Dataset):
    def __init__(self, split, config):
        super().__init__()
        self.config = config
        self.data_path = f"/mnt/workspace/wenyu.zy/argument_mining/data/{split}_data.hdf5"
        with h5py.File(self.data_path, "r") as f:
            self.length = f.attrs["size"]

    def _open_hdf5(self):
        self.data = h5py.File(self.data_path, 'r')

    def __getitem__(self, item):
        if not hasattr(self, 'data'):
            self._open_hdf5()
        inputs, targets = [], []
        input_keys = ["char_id", "word_id", "char_mask", "word_mask", "passage_mask",
                      "para_order", "sent_order", "font_size", "style_mark"]
        target_keys = ["major_idx", "claim_order", "premise_order", "target_relation", "reflection"]
        for input_key in input_keys:
            inputs.append(self.data[input_key][item])
        for target_key in target_keys:
            targets.append(self.data[target_key][item])
        # edges = construct_graph(inputs[2], inputs[3], self.config["neighbor_size"], self.config["max_len"])
        return inputs, targets  #, edges

    def __len__(self):
        return self.length



if __name__ == "__main__":
    index =      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    para_order = [1, 2, 3, 3, 3, 3, 4, 5, 6, 7, 7, 8]
    sent_order = [1, 1, 1, 2, 3, 4, 1, 1, 1, 1, 2, 1]
    neighbor_size = 3
    outer, inner = construct_graph(para_order, sent_order, neighbor_size)
    print(outer)
    print(inner)