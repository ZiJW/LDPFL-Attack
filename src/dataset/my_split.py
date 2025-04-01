import pickle
import random
import numpy as np
import torch

import os
import time
from baseDataset import get_dataset
from torch.utils.data import random_split
from collections import defaultdict

def _handle_folder_name(dataset_name: str, folder_name: str):
    if folder_name == None:
        folder_name = time.strftime("%Y-%m-%d_%H:%M", time.localtime())
    if not os.path.exists("./{}/{}".format(dataset_name, folder_name)):
        os.mkdir("./{}/{}".format(dataset_name, folder_name))
    return folder_name

def _handle_dataset(dataset):
    samples, labels = [], []
    for x, y in dataset:
        if type(x) == torch.Tensor:
            samples.append(x.numpy())
        else:
            raise TypeError("Unknown sample type: {}".format(type(x)))

        if type(y) == int:
            labels.append(y)
        elif type(y) == torch.Tensor:
            labels.append(y.numpy())
        else:
            raise TypeError("Unknown label type: {}".format(type(y)))
    return samples, labels

# ===============================================================================================================

def split_iid(dataset_name: str, dataset, N_clients: int, folder_name: str = None, file_name : str = "train"):
    folder_name = _handle_folder_name(dataset_name, folder_name)

    size = len(dataset) // N_clients
    ind = list(range(len(dataset)))
    
    random.shuffle(ind)

    samples, labels = _handle_dataset(dataset)
    for idx in range(N_clients):
        client_samples = np.stack([samples[ind[idx * size + i]] for i in range(size)])
        if type(labels[0]) == int:
            client_labels  = np.array([labels[ind[idx * size + i]] for i in range(size)])
        elif type(labels[0]) == np.ndarray:
            client_labels  = np.stack([labels[ind[idx * size + i]] for i in range(size)])
        else:
            raise TypeError("Unknown label type: {}".format(type(labels[0])))
    
        np.save("./{}/{}/{}_{}_samples.npy".format(dataset_name, folder_name, file_name, idx + 1), client_samples)
        np.save("./{}/{}/{}_{}_labels.npy".format(dataset_name, folder_name, file_name, idx + 1), client_labels)

        # break
    # clients_dataset_torch = [base_dataset(clients_dataset[i]["sample"], clients_dataset[i]["label"]) for i in range(N_clients)]
    # for idx in range(N_clients):
    #     with open("./{}/{}/{}_{}.pkl".format(dataset_name, folder_name, file_name, idx + 1), "wb") as F:
    #         pickle.dump(clients_dataset_torch[idx], F)
        
    #     st = pickle.dumps(clients_dataset_torch[idx])
    #     print(st)
    #     break

def split_on_label(dataset_name: str, dataset, N_labels: int, slices: list, folder_name: str = None, file_name : str = "train", prob : float = 1.0):
    folder_name = _handle_folder_name(dataset_name, folder_name)

    N_clients = len(slices)
    idxes = []
    for idx in slices:
        idxes += idx
    
    assert sorted(idxes) == list(range(N_labels))

    samples, labels = _handle_dataset(dataset)
    clients_dataset = [{"sample": [], "label": []} for i in range(N_clients)]
    
    N = len(dataset)
    cnt = [0] * N_clients
    for i in range(N):
        mark = True
        for j, idx in enumerate(slices):
            if labels[i] in idx and random.random() < prob:
                clients_dataset[j]["sample"].append(samples[i])
                clients_dataset[j]["label"].append(labels[i])
                cnt[j] += 1
                mark = False
        
        if mark:
            idx = random.randint(0, N_clients - 1)
            clients_dataset[idx]["sample"].append(samples[i])
            clients_dataset[idx]["label"].append(labels[i])
            cnt[idx] += 1

    mini_cnt = min(cnt)
    print("--> original size: {}, reduce to {}".format(cnt, mini_cnt))

    for idx in range(N_clients):
        client_samples = np.stack(clients_dataset[i]["sample"][:mini_cnt])
        if type(labels[0]) == int:
            client_labels  = np.array(clients_dataset[i]["label"][:mini_cnt])
        elif type(labels[0]) == np.ndarray:
            client_labels  = np.stack(clients_dataset[i]["label"][:mini_cnt])
        else:
            raise TypeError("Unknown label type: {}".format(type(labels[0])))
        
        np.save("./{}/{}/{}_{}_samples.npy".format(dataset_name, folder_name, file_name, idx + 1), client_samples)
        np.save("./{}/{}/{}_{}_labels.npy".format(dataset_name, folder_name, file_name, idx + 1), client_labels)

    # clients_dataset_torch = [base_dataset(clients_dataset[i]["sample"][:mini_cnt], clients_dataset[i]["label"][:mini_cnt]) for i in range(N_clients)]
    # for i in range(N_clients):
    #     with open("./{}/{}/{}_{}.pkl".format(dataset_name, folder_name, file_name, i + 1), "wb") as F:
    #         pickle.dump(clients_dataset_torch[i], F)

def generate_noniid_data(dataset, client_num=100, num_classes=10, p=0.5):
    """
    生成符合描述的 Non-IID 数据划分，每类 p% 的数据分给所属客户端，剩下的均匀分配。

    参数:
    - dataset: torchvision 数据集
    - client_num: 客户端数量
    - num_classes: 数据类别数量 (MNIST=10, CH-MNIST=8)
    - p: Non-IID 强度 (0.0 = IID, 1.0 = 完全 Non-IID)

    返回:
    - dict: {client_id: (samples, labels)}，每个客户端的数据及对应的标签
    """
    samples, labels = _handle_dataset(dataset)
    assert len(samples) == len(labels), "samples 和 labels 长度必须相同！"

    # 转换为 NumPy 数组
    samples = np.array(samples)
    labels = np.array(labels)
    data_indices = np.arange(len(labels))  # 所有数据的索引

    # 按类别存储数据索引
    class_indices = {i: data_indices[labels == i] for i in range(num_classes)}

    # 客户端数据存储
    client_data = {i: ([], []) for i in range(client_num)}

    # 1. 每个类别 p% 的数据分配给专属客户端
    clients_per_class = client_num // num_classes  # 每个类别对应多少个客户端
    for cls in range(num_classes):
        np.random.shuffle(class_indices[cls])  # 打乱该类别的数据索引

        # 计算 p% 的数据数量
        p_count = int(len(class_indices[cls]) * p)
        remaining_count = len(class_indices[cls]) - p_count

        # 按类别索引将 p% 数据分给固定客户端
        exclusive_clients = list(range(cls * clients_per_class, (cls + 1) * clients_per_class))
        split_indices = np.array_split(class_indices[cls][:p_count], clients_per_class)

        for i, indices in enumerate(split_indices):
            client_id = exclusive_clients[i]
            if client_id < client_num:
                client_data[client_id][0].extend([samples[idx] for idx in indices])  # 添加样本
                client_data[client_id][1].extend([labels[idx] for idx in indices])  # 添加标签

        # 2. 剩下的 (1-p)% 数据均匀分配给所有客户端
        remaining_data = class_indices[cls][p_count:]
        np.random.shuffle(remaining_data)
        split_remaining = np.array_split(remaining_data, client_num)

        for client_id, indices in enumerate(split_remaining):
            client_data[client_id][0].extend([samples[idx] for idx in indices])
            client_data[client_id][1].extend([labels[idx] for idx in indices])

    # 3. 转换为 NumPy 数组
    for client_id in client_data:
        indices = np.random.permutation(len(client_data[client_id][1]))  # 生成随机排列索引
        client_data[client_id] = (np.array(client_data[client_id][0])[indices], np.array(client_data[client_id][1])[indices])  # 重新排序

    return client_data

def split_noniid(dataset,  num_clients=100, num_classes=10, p=0.5):
    """
    生成 Non-IID 数据划分，并返回直接的 (samples, labels)。

    参数:
    - samples: list, 所有样本数据
    - labels: list, 对应样本的标签
    - num_clients: 客户端数量
    - num_classes: 数据类别数量 (MNIST=10, CH-MNIST=8)
    - p: Non-IID 度 (0.0 = IID, 1.0 = 完全 Non-IID)

    返回:
    - dict: {client_id: (samples, labels)}，每个客户端的数据及标签
    """
    samples, labels = _handle_dataset(dataset)
    assert len(samples) == len(labels), "samples 和 labels 长度必须相同！"

    # 转换为 NumPy 数组
    samples = np.array(samples)
    labels = np.array(labels)
    data_indices = np.arange(len(labels))  # 所有数据的索引

    # 按类别存储数据索引
    class_indices = {i: data_indices[labels == i] for i in range(num_classes)}

    # 客户端数据存储
    clients_per_class = num_clients // num_classes  # 每个类别的客户端组数
    client_data = {i: ([], []) for i in range(num_clients)}

    # 1. 分配类别偏向数据 (p% 数据给特定客户端)
    for cls in range(num_classes):
        np.random.shuffle(class_indices[cls])  # 打乱该类别的数据索引

        # 按类别分组，均分给 clients_per_class 个客户端
        split_indices = np.array_split(class_indices[cls], clients_per_class)

        for i, indices in enumerate(split_indices):
            client_id = cls * clients_per_class + i
            if client_id < num_clients:
                # 以 p 概率直接分配该类别数据
                if random.random() < p:
                    client_data[client_id][0].extend([samples[idx] for idx in indices])
                    client_data[client_id][1].extend([labels[idx] for idx in indices])

    # 2. 以 (1-p) 概率随机分配数据，保证所有客户端都有不同类别
    all_indices = np.concatenate(list(class_indices.values()))  # 获取所有数据索引
    np.random.shuffle(all_indices)  # 打乱索引

    for client_id in range(num_clients):
        missing_samples = len(all_indices) // num_clients - len(client_data[client_id][1])
        if missing_samples > 0:
            client_data[client_id][0].extend([samples[idx] for idx in all_indices[:missing_samples]])
            client_data[client_id][1].extend([labels[idx] for idx in all_indices[:missing_samples]])
            all_indices = all_indices[missing_samples:]

    # 3. 随机打乱每个客户端的数据
    for client_id in client_data:
        indices = np.random.permutation(len(client_data[client_id][1]))  # 生成随机排列索引
        client_data[client_id] = (np.array(client_data[client_id][0])[indices], np.array(client_data[client_id][1])[indices])

    return client_data

def split_public_train_test(NAME: str, folder_name: str):
    train_dataset, test_dataset = get_dataset(NAME)
    length = len(train_dataset)
    public_dataset, train_dataset = random_split(train_dataset, [int(0.1 * length), int(0.9 * length)])
    split_on_label(NAME, train_dataset, 10, [[0,1],[2,3],[4,5],[6,7],[8,9]], folder_name=folder_name)
    with open("./{}/{}/test.pkl".format(NAME, folder_name), "wb") as F:
        pickle.dump(test_dataset, F)
    with open("./{}/{}/public.pkl".format(NAME, folder_name), "wb") as F:
        pickle.dump(public_dataset, F)

def split_train_test_non_iid(NAME: str, folder_name: str):
    train_dataset, test_dataset = get_dataset(NAME)
    split_on_label(NAME, train_dataset, 10, [[0,1],[2,3],[4,5],[6,7],[8,9]], folder_name=folder_name, prob=0.9)
    with open("./{}/{}/test.pkl".format(NAME, folder_name), "wb") as F:
        pickle.dump(test_dataset, F)

def split_train_test_iid(NAME: str, N_clients: int, folder_name: str):
    train_dataset, test_dataset = get_dataset(NAME)
    split_iid(NAME, train_dataset, N_clients, folder_name)
    with open("./{}/{}/test.pkl".format(NAME, folder_name), "wb") as F:
        pickle.dump(test_dataset, F)

def split_public_train_test_iid(NAME: str, N_clients: int, folder_name: str):
    train_dataset, test_dataset = get_dataset(NAME)
    length = len(train_dataset)
    public_dataset, train_dataset = random_split(train_dataset, [int(0.1 * length), int(0.9 * length)])
    split_iid(NAME, train_dataset, N_clients, folder_name=folder_name)
    with open("./{}/{}/test.pkl".format(NAME, folder_name), "wb") as F:
        pickle.dump(test_dataset, F)
    with open("./{}/{}/public.pkl".format(NAME, folder_name), "wb") as F:
        pickle.dump(public_dataset, F)

def split_public_train_test_noniid(NAME: str, N_clients: int, folder_name: str, p:float, public_rate:float):
    train_dataset, test_dataset = get_dataset(NAME)
    length = len(train_dataset)
    public_dataset, train_dataset = random_split(train_dataset, [int(public_rate * length), int(length - public_rate * length)])
    NAME2class_num = {"MNIST":10}
    folder_name = _handle_folder_name(NAME, folder_name)
    result = split_noniid(train_dataset, N_clients, NAME2class_num[NAME], p)
    lengths = []
    for idx in range(0, N_clients):
        lengths.append(len(result[idx][1]))
        np.save("./{}/{}/{}_{}_samples.npy".format(NAME, folder_name, "train", idx + 1), result[idx][0])
        np.save("./{}/{}/{}_{}_labels.npy".format(NAME, folder_name, "train", idx + 1), result[idx][1])
    print(lengths)
    with open("./{}/{}/test.pkl".format(NAME, folder_name), "wb") as F:
        pickle.dump(test_dataset, F)
    with open("./{}/{}/public.pkl".format(NAME, folder_name), "wb") as F:
        pickle.dump(public_dataset, F)

if __name__ == "__main__":
    # split_train_test_non_iid("MNIST", "non_iid_5_p=0.9")
    #split_public_train_test_iid("MNIST", 20, "iid_20_with_public")
    #split_train_test_iid("CIFAR100", 10, "iid_10")
    #split_train_test_iid("MNIST", 5, "iid_5")
    #split_train_test_iid("adult", 20, "iid_20")
    p = 0.0
    split_public_train_test_noniid("MNIST", 20, "noniid_20_p{}_public".format(p), p, 0.05)
