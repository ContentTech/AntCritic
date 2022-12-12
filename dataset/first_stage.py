import random

import h5py
from torch.utils.data import Dataset
import numpy as np

class RandomAugment:
    def __init__(self, bos, eos, mask, probability=0.3, ratio=0.1):
        self.ratio = ratio
        self.probability = probability
        self.bos, self.eos, self.mask = bos, eos, mask

    def __call__(self, token_id, token_mask):
        methods = [self.random_mask, self.random_repeat, self.random_swap]
        if random.random() < self.probability:
            return random.choice(methods)(token_id, token_mask)
        else:
            return token_id, token_mask

    def random_mask(self, token_id, token_mask):
        # print("Before (mask): ", token_id, token_mask)
        valid_mask = (token_id != self.bos) * (token_id != self.eos) * token_mask
        slot_mask = ((np.random.rand(*(token_mask.shape)) < self.ratio) * valid_mask)
        # print("After (mask): ", token_id, token_mask * (1 - slot_mask))
        return token_id, token_mask * (1 - slot_mask)

    def random_swap(self, token_id, token_mask):
        # print("Before (swap): ", token_id, token_mask)
        valid_mask = (token_id != self.bos) * (token_id != self.eos) * token_mask
        next_pos = np.where((np.random.rand(*(token_mask.shape)) < self.ratio) * valid_mask)[0].clip(min=2)
        token_id[next_pos - 1], token_id[next_pos] = token_id[next_pos], token_id[next_pos - 1]
        # print("After (swap): ", token_id, token_mask)
        return token_id, token_mask

    def random_repeat(self, token_id, token_mask):
        # print("Before (repeat): ", token_id, token_mask)
        result_id, result_mask = [], []
        for token, mask in zip(token_id, token_mask):
            if token == self.bos or token == self.eos or mask == 0 or random.random() >= self.ratio:
                result_id.append(token)
                result_mask.append(mask)
            else:
                result_id.extend([token] * 2)
                result_mask.extend([mask] * 2)
        # print("After (repeat): ", result_id, result_mask)
        max_len = len(token_id)
        return np.array(result_id)[:max_len], np.array(result_mask)[:max_len]

data_root = "/mnt/fengyao.hjj/transformers/data/pgc/0506"

class FirstStageDataset(Dataset):
    def __init__(self, split, config):
        super().__init__()
        self.config = config
        print('config --- : ', config)
        self.split = split
        self.word_augment = RandomAugment(0, 2, 250001)
        self.char_augment = RandomAugment(101, 102, 103)
        self.data_path = data_root + f"/{split}_1.hdf5"
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
        if self.split == "train":
            inputs[0], inputs[2] = self.char_augment(inputs[0], inputs[2])
            inputs[1], inputs[3] = self.word_augment(inputs[1], inputs[3])
        for target_key in target_keys:
            targets.append(self.data[target_key][item].astype(np.int64))
        return inputs, targets  #, edges

    def __len__(self):
        return self.length