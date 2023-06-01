import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random
import os

import param
from param import DEVICE

class base_dataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

# ---------------------- set random seed --------------------

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

# ---------------------- Load Datasets ----------------------

def load_dataset(dataset: str, folder: str, names: list, idx: int = -1):
    res = []
    for name, is_client_unique in names:
        if is_client_unique:
            assert idx >= 1
            res.append(_load(dataset, folder, "{}_{}".format(name, idx)))
        else:
            res.append(_load(dataset, folder, name))
    return res

def _load(dataset: str, folder: str, name: str):
    Name = "./dataset/{}/{}/{}".format(dataset, folder, name)
    if os.path.exists(Name + ".pkl"):
        with open(Name + ".pkl", "rb") as F:
            dataset = pickle.load(F)
    elif os.path.exists(Name + "_samples.npy") and os.path.exists(Name + "_labels.npy"):
        samples = torch.from_numpy(np.load("./dataset/{}/{}/{}_samples.npy".format(dataset, folder, name)))
        labels  = torch.from_numpy(np.load("./dataset/{}/{}/{}_labels.npy".format(dataset, folder, name)))
        dataset = base_dataset(samples, labels)
    else:
        raise ValueError("Unknown dataset format! - {}".format(Name))

    loader = DataLoader(dataset, batch_size=param.BATCH_SIZE_TRAIN, shuffle=True)
    return loader
