import pickle
import torch
import param

import random

from param import DEVICE
from model import *

from torch.utils.data import DataLoader

# ---------------------- set random seed --------------------

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    # np.random.seed(seed)

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
    with open("./dataset/{}/{}/{}.pkl".format(dataset, folder, name), "rb") as F:
        dataset = pickle.load(F)
    loader = DataLoader(dataset, batch_size=param.BATCH_SIZE_TRAIN, shuffle=True)
    return loader
