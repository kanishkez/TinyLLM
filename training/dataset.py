import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TokenDataset(Dataset):

    def __init__(self, path, seq_len):
        self.data    = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, i):
        s = i * self.seq_len
        x = torch.from_numpy(self.data[s     : s + self.seq_len    ].astype(np.int64))
        y = torch.from_numpy(self.data[s + 1 : s + self.seq_len + 1].astype(np.int64))
        return x, y


def get_loader(path, seq_len, batch_size, num_workers=4):
    ds = TokenDataset(path, seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True, drop_last=True)
