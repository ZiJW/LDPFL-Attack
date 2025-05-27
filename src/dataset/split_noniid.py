import pickle
import random
import numpy as np
import torch

import os
import time
from baseDataset import *
from torch.utils.data import random_split
from torchvision import datasets, transforms

def _handle_folder_name(dataset_name: str, folder_name: str):
    if folder_name == None:
        folder_name = time.strftime("%Y-%m-%d_%H:%M", time.localtime())
    folder_path = "{}/{}/{}".format(DEFAULT_DATASET_PATH, dataset_name, folder_name)
    os.system("mkdir -p {}".format(folder_path))
    return folder_name, folder_path

def _handle_dataset(dataset):
    samples, labels = [], []
    for x, y in dataset:
        samples.append(x)
        if type(y) == int:
            labels.append(y)
        elif type(y) == torch.Tensor:
            labels.append(int(y.numpy()))
        else:
            raise TypeError("Unknown label type: {}".format(type(y)))
    return samples, labels

# ===============================================================================================================

def _split_dirichlet_on_label(dataset_info, dist = None):
    dataset_name, folder_name, file_name = dataset_info["name"], dataset_info["folder_name"], dataset_info["file_name"]
    N_clients, Label_num = dataset_info["client_num"], dataset_info["label_num"]
    dataset, transform = dataset_info["dataset"], dataset_info["transform"]

    folder_name, folder_path = _handle_folder_name(dataset_name, folder_name)
    samples, labels = _handle_dataset(dataset)

    if type(dist) != np.ndarray: 
        distr = np.random.dirichlet([ALPHA] * N_clients, size=Label_num)
    else:
        distr = dist
        assert distr.shape == (Label_num, N_clients)
    original_distr = distr.copy()

    Label_sample = [[] for idx in range(Label_num)]
    for idx, label in enumerate(labels):
        Label_sample[label].append(samples[idx])
    Label_count = [len(sample) for sample in Label_sample]

    # print(Label_count)

    for idx in range(Label_num):
        distr[idx] *= Label_count[idx]
    distr = np.rint(distr).astype(np.int32)
    
    for idx in range(Label_num):
        if sum(distr[idx]) > Label_count[idx]:
            ind = np.argmax(distr[idx])
            distr[idx][ind] -= sum(distr[idx]) - Label_count[idx]
            assert distr[idx][ind] >= 0

        elif sum(distr[idx]) < Label_count[idx]:
            left = Label_count[idx] - sum(distr[idx]) 
            for _ in range(left):
                distr[idx][np.random.randint(Label_num)] += 1

    # print(distr)
    print(np.sum(distr, axis=0))
    client_samples = [[] for idx in range(N_clients)]
    client_labels = [[] for idx in range(N_clients)]
    for label in range(Label_num):
        cnt = 0
        for idx in range(N_clients):
            client_samples[idx] += Label_sample[label][cnt: cnt + distr[label][idx]]
            client_labels[idx] += [label] * distr[label][idx]
            cnt += distr[label][idx]

    for idx in range(N_clients):
        if file_name == "train" :
            np.save("{}/{}_{}_samples.npy".format(folder_path, file_name, idx + 1), np.array(client_samples[idx]))
            np.save("{}/{}_{}_labels.npy".format(folder_path, file_name, idx + 1), np.array(client_labels[idx]))

    return original_distr

def _split_pathological_on_label(dataset_info, dist = None):
    dataset_name, folder_name, file_name = dataset_info["name"], dataset_info["folder_name"], dataset_info["file_name"]
    N_clients, Label_num = dataset_info["client_num"], dataset_info["label_num"]
    dataset, transform = dataset_info["dataset"], dataset_info["transform"]
    Label_num_each = dataset_info["label_num_each"]

    folder_name, folder_path = _handle_folder_name(dataset_name, folder_name)
    samples, labels = _handle_dataset(dataset)

    distr = np.zeros((Label_num, N_clients))
    if type(dist) != np.ndarray:
        assert N_clients * Label_num_each % Label_num == 0
        Avg = N_clients * Label_num_each // Label_num
        cnt = np.array([Avg] * Label_num, dtype=np.int32)

        for idx in range(N_clients):
            for label in range(Label_num_each):
                pool = np.where(cnt == np.max(cnt))[0]
                choice = random.choice(pool)
                distr[choice][idx] = random.uniform(PROB_LOW, PROB_HIGH)
                cnt[choice] -= 1

        for label in range(Label_num):
            distr[label] = distr[label] / np.sum(distr[label])
    else:
        distr = dist
        assert distr.shape == (Label_num, N_clients)
    
    # print(distr)
    original_distr = distr.copy()

    Label_sample = [[] for idx in range(Label_num)]
    for idx, label in enumerate(labels):
        Label_sample[label].append(samples[idx])
    Label_count = [len(sample) for sample in Label_sample]

    # print(Label_count)

    for idx in range(Label_num):
        distr[idx] *= Label_count[idx]
    distr = np.rint(distr).astype(np.int32)
    
    for idx in range(Label_num):
        if sum(distr[idx]) > Label_count[idx]:
            ind = np.argmax(distr[idx])
            distr[idx][ind] -= sum(distr[idx]) - Label_count[idx]
            assert distr[idx][ind] >= 0

        elif sum(distr[idx]) < Label_count[idx]:
            left = Label_count[idx] - sum(distr[idx]) 
            for _ in range(left):
                distr[idx][np.random.randint(Label_num)] += 1

    # print(distr)
    # print(np.sum(distr, axis=0))
    client_samples = [[] for idx in range(N_clients)]
    client_labels = [[] for idx in range(N_clients)]
    for label in range(Label_num):
        cnt = 0
        for idx in range(N_clients):
            client_samples[idx] += Label_sample[label][cnt: cnt + distr[label][idx]]
            client_labels[idx] += [label] * distr[label][idx]
            cnt += distr[label][idx]

    for idx in range(N_clients):
        cur_dataset = base_dataset(client_samples[idx], client_labels[idx], transform=transform)
        torch.save(cur_dataset, ("{}/{}_{}.pth".format(folder_path, file_name, idx + 1)))

    return original_distr

def _split_iid(dataset_name: str, dataset, N_clients: int, folder_name: str = None, file_name : str = "train", transform=None):
    folder_name, folder_path = _handle_folder_name(dataset_name, folder_name)
    
    size = len(dataset) // N_clients
    ind = list(range(len(dataset)))
    
    random.shuffle(ind)

    samples, labels = _handle_dataset(dataset)
    for idx in range(N_clients):
        client_samples = [samples[ind[idx * size + i]] for i in range(size)]
        client_labels  = [labels[ind[idx * size + i]] for i in range(size)]

        cur_dataset = base_dataset(client_samples, client_labels, transform=transform)
        torch.save(cur_dataset, ("{}/{}_{}.pth".format(folder_path, file_name, idx + 1)))

# def split_cifar100(folder_name: str):
#     dataset_name = "CIFAR100"
#     train_dataset, test_dataset = get_dataset(dataset_name)
#     folder_name = _handle_folder_name(dataset_name, folder_name)

#     samples, labels = _handle_dataset(train_dataset)
#     with open("cifar-100-python/labelMap.pkl", "rb") as F:
#         labelMap = pickle.load(F)

#     print(labels[:10])

#     for i in range(len(labels)):
#         for j in range(len(labelMap)):
#             if labels[i] in labelMap[j]:
#                 labels[i] = j
#                 break

#     print(labels[:10])

#     np.save("./{}/{}/train_samples.npy".format(dataset_name, folder_name), samples)
#     np.save("./{}/{}/train_labels.npy".format(dataset_name, folder_name), labels)

#     samples, labels = _handle_dataset(test_dataset)
#     print(labels[:10])
#     for i in range(len(labels)):
#         for j in range(len(labelMap)):
#             if labels[i] in labelMap[j]:
#                 labels[i] = j
#                 break
        
#     print(labels[:10])

#     np.save("./{}/{}/test_samples.npy".format(dataset_name, folder_name), samples)
#     np.save("./{}/{}/test_labels.npy".format(dataset_name, folder_name), labels)


SEED = 98
CLIENTS_NUM = 20
NAME = "FashionMNIST"
#TYPE = "pathological"
TYPE = "dirichlet"

# Public dataset
PUBLIC = True
PUBLIC_RATIO = 0.05

# pathological
ALPHA = 1
PROB_LOW = 0.4
PROB_HIGH = 0.6

TRANSFORM_TRAIN = None
TRANSFORM_TEST = None

if __name__ == "__main__":
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    if NAME == "MNIST":
        #TRANSFORM_TRAIN = transform_MNIST_train
        #TRANSFORM_TEST = transform_MNIST_test
        TRANSFORM_TRAIN = transforms.ToTensor()
        TRANSFORM_TEST = transforms.ToTensor()
        LABEL_NUM = 10
        LABEL_NUM_PER_USER = 2
    elif NAME == "FashionMNIST":
        TRANSFORM_TRAIN = transforms.ToTensor()
        TRANSFORM_TEST = transforms.ToTensor()
        LABEL_NUM = 10
        LABEL_NUM_PER_USER = 2
    elif NAME == "CIFAR10":
        #TRANSFORM_TRAIN = transform_CIFAR10_train
        #TRANSFORM_TEST = transform_CIFAR10_test
        TRANSFORM_TRAIN = transforms.ToTensor()
        TRANSFORM_TEST = transforms.ToTensor()
        LABEL_NUM = 10
        LABEL_NUM_PER_USER = 2
    elif NAME == "CIFAR100":
        TRANSFORM_TRAIN = transform_CIFAR100_train
        TRANSFORM_TEST = transform_CIFAR100_test
        LABEL_NUM = 100
        LABEL_NUM_PER_USER = 10
    else:
        raise ValueError("Unknown dataset: {}".format(NAME))
    
    train_dataset, test_dataset = get_dataset(NAME)
    if PUBLIC:
        size = int(len(train_dataset) * PUBLIC_RATIO)
        # samples, labels = _handle_dataset(train_dataset)
        public_dataset, train_dataset = random_split(train_dataset, [size, len(train_dataset) - size])
        #public_samples, public_labels = _handle_dataset(public_dataset)
        #public_dataset = base_dataset(public_samples, public_labels, transform=TRANSFORM_TRAIN)

    train_dataset_info = {"name": NAME,
                    "client_num": CLIENTS_NUM,
                    "label_num": LABEL_NUM,
                    "label_num_each": LABEL_NUM_PER_USER,
                    "file_name": "train",
                    "dataset": train_dataset,
                    "transform": TRANSFORM_TRAIN
                }
    test_dataset_info = {"name": NAME,
                    "client_num": 1,
                    "label_num": LABEL_NUM,
                    "label_num_each": LABEL_NUM_PER_USER,
                    "file_name": "test",
                    "dataset": test_dataset,
                    "transform": TRANSFORM_TEST
                }


    if TYPE == "iid":
        folder_name = "iid_{}_{}".format(CLIENTS_NUM, SEED)
        if PUBLIC:
            folder_name += "_public{}".format(PUBLIC_RATIO)

        train_dataset_info["folder_name"] = folder_name
        test_dataset_info["folder_name"] = folder_name
        _split_iid(train_dataset_info)
        _split_iid(test_dataset_info)

    elif TYPE == "dirichlet":
        folder_name = "dirichlet_{}users_a{}_seed{}".format(CLIENTS_NUM, ALPHA, SEED)
        if PUBLIC:
            folder_name += "_public{}".format(PUBLIC_RATIO)

        train_dataset_info["folder_name"] = folder_name
        test_dataset_info["folder_name"] = folder_name
        distr = _split_dirichlet_on_label(train_dataset_info)
    
    elif TYPE == "pathological":
        folder_name = "pathological_{}users_seed{}".format(CLIENTS_NUM, SEED)
        if PUBLIC:
            folder_name += "_public{}".format(PUBLIC_RATIO)
        train_dataset_info["folder_name"] = folder_name
        test_dataset_info["folder_name"] = folder_name
        distr = _split_pathological_on_label(train_dataset_info)
        _split_pathological_on_label(test_dataset_info, dist=distr)

    else:
        raise ValueError("Unknown split type: {}".format(TYPE))
    
    if PUBLIC:
        folder_name, folder_path = _handle_folder_name(NAME, folder_name)
        #torch.save(public_dataset, ("{}/public.pth".format(folder_path)))
        with open("{}/test.pkl".format(folder_path), "wb") as F:
            pickle.dump(test_dataset, F)
        with open("{}/public.pkl".format(folder_path), "wb") as F:
            pickle.dump(public_dataset, F)