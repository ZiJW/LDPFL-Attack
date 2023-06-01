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

        Thr = [threading.Thread(target=LDPFL_server.collect_weights, args=(self, idx)) for idx in chose]
        for tr in Thr:
            tr.start()
        for tr in Thr:
            tr.join()

        res = []
        for ln in range(len(self.model_size)):
            weight = []
            if param.CLIENTS_WEIGHTS == None:
                for _ in chose:
                    idx, wt = self.weights_buffer[ln].get()
                    weight.append(wt)
                weight = torch.stack(weight)
                weight = torch.mean(weight, dim=0)
            else:
                for _ in chose:
                    idx, wt = self.weights_buffer[ln].get()
                    weight.append(wt * (param.CLIENTS_WEIGHTS[idx] / sum(param.CLIENTS_WEIGHTS)))
                weight = torch.stack(weight)
                weight = torch.sum(weight, dim=0)
            res.append(weight)
        
        res = torch.cat(res)
        self.unserialize_model(res)

    def evaluate(self):
        self.comm.initialize()
        for rn in range(self.round):
            self.train(rn)
            self.test(rn)
            torch.save(self.model, self.log_path + "model/model_{}".format(rn))