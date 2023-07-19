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
    
def ExpM(V, eps):
    d = len(V)
    pr = torch.exp(eps*torch.abs(V)/(d-1))
    pr = np.asarray(pr.div(torch.sum(pr)).cpu().numpy()).astype('float64')
    pr = pr/np.sum(pr)
    index = np.random.choice(range(d), p=pr)
    return index

def PE(V: torch.Tensor, eps):
    N = len(V)
    k = int(N/10)
    ind = torch.topk(V, k).indices
    arr = torch.Tensor(np.zeros(N))
    arr[ind] = 1
    flip_pr = 1/(1+torch.exp(eps))
    arr2 = torch.Tensor(np.random.binomial(1, flip_pr, N))
    arr.logical_xor(arr2)
    non_zeros = arr.nonzero()[0]
    if len(non_zeros) == 0:
        return None
    else:
        return np.random.choice(non_zeros)

    

def pm_perturbation(query_result, clip, epsilon):
    if query_result>clip:
        query_result = clip
    if query_result<-clip:
        query_result = -clip

    if type(query_result) is torch.Tensor:
        query_result = query_result.cpu().numpy()
    tran_x = query_result / clip
    noisy_query = []
    
    ee2 = np.exp(epsilon/2)
    ee = np.exp(epsilon)
    s = (ee2 + 1) / (ee2 - 1)
    
    l = (ee2 * tran_x - 1) / (ee2 - 1)
    r = (ee2 * tran_x + 1) / (ee2 - 1)
    r1 = np.random.uniform(0, 1)
    if r1 < ee2 / (ee2 + 1):
        noisy_query = np.random.uniform(l, r)
    else:
        len1 = l + s
        len2 = s - r

        if np.random.random() < len1 / (len1 + len2):
            noisy_query = np.random.uniform(-s, l)
        else:
            noisy_query = np.random.uniform(r, s)

    return noisy_query


def pm_aggregation(noisy_reports, clip):
    result = []
    for x in noisy_reports:
        result.append(x * clip)
    return np.mean(result)
