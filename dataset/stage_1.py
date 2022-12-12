import h5py
from torch.utils.data import Dataset
import numpy as np


class FirstStageDataset(Dataset):
    def __init__(self, split, config):
        super().__init__()
        self.config = config
        self.data_path = f"/mnt/workspace/wenyu.zy/argument_mining/data/{split}_1.hdf5"
        with h5py.File(self.data_path, "r") as f:
            self.length = f.attrs["size"]

    def _open_hdf5(self):
        self.data = h5py.File(self.data_path, 'r')

    def __getitem__(self, item):
        if not hasattr(self, 'data'):
            self._open_hdf5()
        inputs, targets = [], []
        input_keys = ["char_id", "word_id", "char_mask", "word_mask"]
        target_keys = ["label"]
        for input_key in input_keys:
            inputs.append(self.data[input_key][item].astype(np.int64))
        for target_key in target_keys:
            targets.append(self.data[target_key][item].astype(np.int64))
        return inputs, targets  #, edges

    def __len__(self):
        return self.length