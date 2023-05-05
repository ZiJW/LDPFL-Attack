import pickle
import torch
import param

import math
import random

from param import DEVICE
from model import *

from torch.utils.data import DataLoader

def log(info: str):
    if param.DEBUG:
        print(info)
        
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

# ---------------------- Functions ----------------------

def data_pertubation(W, c: float, r: float, eps: float):
    # print("--> {}, c={}, r={}".format(W, c, r))
    try:
        temp = iter(W)
        L = len(W)
        for idx in range(L):
            W[idx] = data_pertubation((float)(W[idx]), c, r, eps)
        return None
    except TypeError:
        # assert (c - r <= W) and (W <= c + r)
        coff = (math.exp(eps) + 1) / (math.exp(eps) - 1)
        Pb = (((W - c) / coff) + r) / (2.0 * r)
        # print('--> > {}, {} ({}, {})'.format(coff, Pb, c, r))
        if random.random() < Pb:
            res = c + r * coff
        else:
            res = c - r * coff
        return res