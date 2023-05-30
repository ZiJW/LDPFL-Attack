import pickle
import random
import numpy as np

import os
import time
from baseDataset import base_dataset, get_dataset
from torch.utils.data import random_split

def handle_folder_name(dataset_name: str, folder_name: str):
    if folder_name == None:
        folder_name = time.strftime("%Y-%m-%d_%H:%M", time.localtime())
    if not os.path.exists("./{}/{}".format(dataset_name, folder_name)):
        os.mkdir("./{}/{}".format(dataset_name, folder_name))
    return folder_name

def split_iid(dataset_name: str, dataset, N_clients: int, folder_name: str = None, file_name : str = "train"):
    folder_name = handle_folder_name(dataset_name, folder_name)

    size = len(dataset) // N_clients
    ind = list(range(len(dataset)))
    
    random.shuffle(ind)

    samples, labels = [], []
    for x, y in dataset:
        samples.append(x.numpy())
        if type(y) == int:
            labels.append(y)
        else:
            raise TypeError("Unknown label type: {}".format(type(y)))
    
    for idx in range(N_clients):
        client_samples = np.stack([samples[ind[idx * size + i]] for i in range(size)])
        if type(labels[0]) == int:
            client_labels  = np.array([labels[ind[idx * size + i]] for i in range(size)])
        else:
            raise TypeError("Unknown label type: {}".format(type(labels[0])))
    
        np.save("./{}/{}/{}_samples_{}.pkl".format(dataset_name, folder_name, file_name, idx + 1), client_samples)
        np.save("./{}/{}/{}_labels_{}.pkl".format(dataset_name, folder_name, file_name, idx + 1), client_labels)

        # break
    # clients_dataset_torch = [base_dataset(clients_dataset[i]["sample"], clients_dataset[i]["label"]) for i in range(N_clients)]
    # for idx in range(N_clients):
    #     with open("./{}/{}/{}_{}.pkl".format(dataset_name, folder_name, file_name, idx + 1), "wb") as F:
    #         pickle.dump(clients_dataset_torch[idx], F)
        
    #     st = pickle.dumps(clients_dataset_torch[idx])
    #     print(st)
    #     break

def split_on_label(dataset_name: str, dataset, N_labels: int, slices: list, folder_name: str = None, file_name : str = "train", prob : float = 1.0):
    folder_name = handle_folder_name(dataset_name, folder_name)

    N_clients = len(slices)
    idxes = []
    for idx in slices:
        idxes += idx
    
    assert sorted(idxes) == list(range(N_labels))

    samples, labels = [], []
    clients_dataset = [{"sample": [], "label": []} for i in range(N_clients)]

    for x, y in dataset:
        samples.append(x)
        labels.append(y)
    
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

    clients_dataset_torch = [base_dataset(clients_dataset[i]["sample"][:mini_cnt], clients_dataset[i]["label"][:mini_cnt]) for i in range(N_clients)]
    for i in range(N_clients):
        with open("./{}/{}/{}_{}.pkl".format(dataset_name, folder_name, file_name, i + 1), "wb") as F:
            pickle.dump(clients_dataset_torch[i], F)


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

if __name__ == "__main__":
    # split_train_test_non_iid("MNIST", "non_iid_5_p=0.9")
    # split_public_train_test("MNIST", "non_iid_5")
    split_train_test_iid("MNIST", 10, "test")
