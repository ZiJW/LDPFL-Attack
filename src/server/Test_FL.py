import torch
import threading
import queue
import random
import logging

from network import socket_comm, fake_comm
from util import load_dataset
from model import load_criterion
import param
import numpy as np
from base_server import Base_server

class Test_server(Base_server):
    """
        2.1.2 Section 3 Attack
        1. 将不考虑LDP的攻击机制加到上面
        2. 恶意的攻击者提供假的参数，试图让模型不收敛

        10 clients, Epoch=5, Round=20, lr=0.005
        ----------------------------------------
        Baseline: Acc = 0.979, Loss = 0.0667
    """
    def __init__(self):
        super().__init__(param.N_NODES, param.MODEL, param.MODEL_PARAM, param.N_EPOCH, comm=param.COMM)
        self.round = param.N_ROUND
        self.kap = param.KAP
        
        self.test_loader, = load_dataset(param.DATASET, param.FOLDER, [("test", False)])
        self.criterion = load_criterion(param.CRITERION)

        self.model_size = []
        sd = self.model.state_dict()
        for x in sd:
            self.model_size.append(sd[x].numel())
        logging.info("Server : model shape : {}".format(self.model_size))

        self.last_range = [None for idx in range(len(self.model_size))]
        self.weights_buffer = [queue.Queue(maxsize=self.size - 1) for idx in range(len(self.model_size))]

    def collect_weights(self, idx: int):
        for ln in range(len(self.model_size)):
            msg = self.comm.recv(idx)
            self.weights_buffer[msg["ln"]].put((idx, msg["weight"]))
            if ln < len(self.model_size) - 1: 
                self.comm.send(idx, "ACK")

    def compute_parameter_divergence(self, record_param, weight_range, save_path):
        center_list = [weight_range[idx]["center"] for idx in range(len(weight_range))]
        num_clients = len(record_param)
        num_layers = len(record_param[0])

        # 初始化区分度矩阵
        divergence_matrix = np.zeros((num_layers, num_clients, num_clients), dtype=int)

        # 遍历每一层
        for layer_idx in range(num_layers):
            center = center_list[layer_idx]  # 获取当前层的中心参考值

            # 获取所有客户端在该层的参数
            layer_params = [record_param[i][layer_idx].flatten() for i in range(num_clients)]

            # 遍历每一对客户端
            for i in range(num_clients):
                for j in range(i + 1, num_clients):
                    # 计算两个客户端参数与中心的关系
                    i_less_than_center = layer_params[i] < center
                    j_less_than_center = layer_params[j] < center

                    # 统计它们在不同侧的参数数量
                    divergence_score = torch.sum(i_less_than_center ^ j_less_than_center).item()

                    # 更新区分度矩阵
                    divergence_matrix[layer_idx, i, j] += divergence_score
                    divergence_matrix[layer_idx, j, i] += divergence_score  # 矩阵对称
        if save_path :
            with open(save_path, "w") as f:
                layer_sum = divergence_matrix.sum(axis=0)
                f.write(f"total difference{i}:\n")
                np.savetxt(f, layer_sum, fmt="%d", delimiter=" ")
                f.write("\n")
                for i in range(divergence_matrix.shape[0]):
                    f.write(f"Layer difference{i}:\n")
                    np.savetxt(f, divergence_matrix[i], fmt="%d", delimiter=" ")
                    f.write("\n")
        return divergence_matrix


    def train(self, rn):
        chose = random.sample(list(range(1, self.size)), self.kap[rn])   
        for idx in range(1, self.size):
            self.comm.send(idx, idx in chose)
            assert self.comm.recv(src=idx) == "ACK"
            logging.debug("Server: send invitation {} to Client {} ".format(idx in chose, idx))

        global_model = self.serialize_model(type="raw")
        
        weight_range = []
        for idx, weight in enumerate(global_model):
            mini, maxi = torch.min(weight), torch.max(weight)

            new_range = {"center": (float)((mini + maxi) / 2), "range": (float)((maxi - mini) / 2) if self.last_range[idx] == None else self.last_range[idx]}
            # new_range = {"center": (float)((mini + maxi) / 2), "range": (float)((maxi - mini) / 2)}
            # new_range = (float)((maxi - mini) / 2) if self.last_range[idx] == None else self.last_range[idx]
            self.last_range[idx] = new_range["range"] * 0.995
            weight_range.append(new_range)
        global_model = torch.cat(global_model)

        if param.COMM == "socket":
            Thr = [threading.Thread(target=socket_comm.send, args=(self.comm, idx, (global_model, weight_range))) for idx in chose]
        elif param.COMM == "fake_socket":
            Thr = [threading.Thread(target=fake_comm.send, args=(self.comm, idx, (global_model, weight_range))) for idx in chose]
        else: 
            raise ValueError("Invalid communication type: {}".format(param.COMM))
        
        for tr in Thr:
            tr.start()
        for tr in Thr:
            tr.join()

        logging.debug("Server: send global weights to Client {}".format(chose))

        if param.TAPPING_CLIENTS == []:
            Thr = [threading.Thread(target=Test_server.collect_weights, args=(self, idx)) for idx in chose]
            for tr in Thr:
                tr.start()
            for tr in Thr:
                tr.join()

            record_param = [[] for i in range(self.size - 1)]
            for ln in range(len(self.model_size)):
                for _ in chose:
                    idx, wt = self.weights_buffer[ln].get()
                    record_param[idx - 1].append(wt)
        else :
            Thr = [threading.Thread(target=Test_server.collect_weights, args=(self, idx)) for idx in chose if idx not in param.TAPPING_CLIENTS]
            good_client_num = len(Thr)
            for tr in Thr:
                tr.start()
            for tr in Thr:
                tr.join()
            record_param = [[] for i in range(self.size - 1)]
            for ln in range(len(self.model_size)):
                for _ in range(good_client_num):
                    idx, wt = self.weights_buffer[ln].get()
                    record_param[idx - 1].append(wt)
            
            if param.COMM == "socket":
                Thr = [threading.Thread(target=socket_comm.send, args=(self.comm, idx, record_param)) for idx in chose if idx in param.TAPPING_CLIENTS]
            elif param.COMM == "fake_socket":
                Thr = [threading.Thread(target=fake_comm.send, args=(self.comm, idx, record_param)) for idx in chose if idx in param.TAPPING_CLIENTS]
            else: 
                raise ValueError("Invalid communication type: {}".format(param.COMM))

            for tr in Thr:
                tr.start()
            for tr in Thr:
                tr.join()
            logging.debug("Send other clients' parameter to tapping client")

            Thr = [threading.Thread(target=Test_server.collect_weights, args=(self, idx)) for idx in chose if idx in param.TAPPING_CLIENTS]
            bad_client_num = len(Thr)
            for tr in Thr:
                tr.start()
            for tr in Thr:
                tr.join()
            
            for ln in range(len(self.model_size)):
                for _ in range(bad_client_num):
                    idx, wt = self.weights_buffer[ln].get()
                    record_param[idx - 1].append(wt)

        if param.TAPPING_SAME:
            for i in param.TAPPING_CLIENTS:
                record_param[i - 1] = record_param[param.TAPPING_CLIENTS[0] - 1]

        merged_param = [ torch.cat(record_param[idx]) for idx in range(len(record_param)) ]
        logging.debug("Server merged_param shape : {}".format(merged_param[0].shape))
        if param.MKRUM :
            selected_indices = self.select_multikrum(merged_param, param.MAX_FAILURE, param.KRUM_SELECTED)
        else :
            selected_indices = torch.tensor(range(0, self.size - 1))
        logging.info("Server select aggregating clients : {}".format(selected_indices + 1))
        bad_list_idx = torch.tensor(param.BAD_CLIENTS) - 1
        self.visualize_parameter(merged_param, "round {}".format(rn), "{}fig/round{}.png".format(self.log_path, rn), 
                                 mode = "MDS", red_list=bad_list_idx.tolist(), blue_list=selected_indices.tolist())
        self.compute_parameter_divergence(record_param, weight_range, save_path="{}fig/round{}.txt".format(self.log_path, rn))
        
        res = []
        for ln in range(len(self.model_size)):
            weight = []
            if param.CLIENTS_WEIGHTS == None:
                for idx in selected_indices:
                    weight.append(record_param[idx][ln])
                weight = torch.stack(weight)
                weight = torch.mean(weight, dim=0)
            else:
                sum_client_weights = 0.0
                for idx in selected_indices:
                    weight.append(record_param[idx][ln] * (param.CLIENTS_WEIGHTS[idx + 1]))
                    sum_client_weights += param.CLIENTS_WEIGHTS[idx + 1]
                weight = torch.stack(weight)
                weight = torch.sum(weight, dim=0)
                weight = weight / sum_client_weights
            res.append(weight)
        res = torch.cat(res)
        logging.debug("Server paramter shape : {}".format(res.shape))

        """res = torch.tensor([0.]*len(merged_param[0]), dtype=torch.float32).to(param.DEVICE)
        if param.CLIENTS_WEIGHTS == None: 
            for idx in selected_indices :
                res += merged_param[idx]
            res.div(selected_indices.shape[0])
        else:
            cnt_weights = 0.0
            for idx in selected_indices :
                res += merged_param[idx] * param.CLIENTS_WEIGHTS[idx + 1]
                cnt_weights += param.CLIENTS_WEIGHTS[idx + 1]
            res.div(cnt_weights)"""
            
        self.unserialize_model(res)

    def evaluate(self):
        self.comm.initialize()
        Acc, Loss = [], []
        for rn in range(self.round):
            self.train(rn)
            acc, loss = self.test(rn)
            Acc.append(acc)
            Loss.append(loss)
            #torch.save(self.model, self.log_path + "model/model_{}".format(rn))
        self.draw(Acc, "Acc", "Accurancy")
        self.draw(Loss, "Loss", "Loss")