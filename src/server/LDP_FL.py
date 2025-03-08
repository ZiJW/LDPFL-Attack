import torch
import threading
import queue
import random
import logging

from network import socket_comm, fake_comm
from util import load_dataset
from model import load_criterion
import param
from base_server import Base_server

class LDPFL_server(Base_server):
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

        self.last_range = [None for idx in range(len(self.model_size))]
        self.weights_buffer = [queue.Queue(maxsize=self.size - 1) for idx in range(len(self.model_size))]

    def collect_weights(self, idx: int):
        for ln in range(len(self.model_size)):
            msg = self.comm.recv(idx)
            self.weights_buffer[msg["ln"]].put((idx, msg["weight"]))
            if ln < len(self.model_size) - 1: 
                self.comm.send(idx, "ACK")
            
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

        for idx in chose:
            logging.debug("Server: send global weights to Client {}".format(idx))

        if param.TAPPING_CLIENTS == []:
            Thr = [threading.Thread(target=LDPFL_server.collect_weights, args=(self, idx)) for idx in chose]
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
            Thr = [threading.Thread(target=LDPFL_server.collect_weights, args=(self, idx)) for idx in chose if idx not in param.TAPPING_CLIENTS]
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
            temp_idx2param = {}
            for idx in range(self.size - 1):
                if record_param[idx] != []:
                    temp_idx2param[idx] = torch.cat(record_param[idx])
            
            if param.COMM == "socket":
                Thr = [threading.Thread(target=socket_comm.send, args=(self.comm, idx, temp_idx2param)) for idx in chose if idx in param.TAPPING_CLIENTS]
            elif param.COMM == "fake_socket":
                Thr = [threading.Thread(target=fake_comm.send, args=(self.comm, idx, temp_idx2param)) for idx in chose if idx in param.TAPPING_CLIENTS]
            else: 
                raise ValueError("Invalid communication type: {}".format(param.COMM))

            for tr in Thr:
                tr.start()
            for tr in Thr:
                tr.join()
            logging.debug("Send other clients' parameter to tapping client")

            Thr = [threading.Thread(target=LDPFL_server.collect_weights, args=(self, idx)) for idx in chose if idx in param.TAPPING_CLIENTS]
            bad_client_num = len(Thr)
            for tr in Thr:
                tr.start()
            for tr in Thr:
                tr.join()
            
            for ln in range(len(self.model_size)):
                for _ in range(bad_client_num):
                    idx, wt = self.weights_buffer[ln].get()
                    record_param[idx - 1].append(wt)

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
        for rn in range(self.round):
            self.train(rn)
            self.test(rn)
            #torch.save(self.model, self.log_path + "model/model_{}".format(rn))