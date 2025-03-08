import random
import time
import math
import logging
import torch
import queue
import threading

from base_client import Base_client
from geom_median.torch import compute_geometric_median 
from util import load_dataset
import param
from param import DEVICE
    
def data_pertubation(W, c: float, r: float, eps: float, type: str = "normal"):
    sz = len(W)
    with torch.no_grad():
        # torch.clamp(W, min=c-r, max=c+r)
        coff = (math.exp(eps) + 1) / (math.exp(eps) - 1)

        Pb = (((W - c) / coff) + r)
        rnd = (torch.rand(sz) * 2.0 * r).to(DEVICE)
        cmp = torch.gt(Pb, rnd).to(DEVICE)
        if type == "bad":
            cmp = ~cmp

        res = ((cmp) * (c + r * coff)) + ((~cmp) * (c - r * coff))
    return res

class Test_client(Base_client):
    """
        The client in LDP-FL
    """
    def __init__(self, id):
        super().__init__(id, param.N_NODES, param.MODEL, param.MODEL_PARAM, 
                         param.OPTIMIZER, param.LEARNING_RATE, param.CRITERION, comm=param.COMM)
        self.train_loader, = load_dataset(param.DATASET, param.FOLDER, [("train", True)], idx=self.id)
        self.epoch = param.N_EPOCH
        self.round = param.N_ROUND

        if param.CLIENTS_WEIGHTS != None:
            logging.info("Client {} has weight {}".format(self.id, param.CLIENTS_WEIGHTS[self.id]))
        if self.id in param.BAD_CLIENTS:
            logging.info("Client {} is an adversary!".format(self.id))
        self.weights_buffer = [queue.Queue(maxsize=self.size - 1) for idx in range(len(self.model_size))]

    def chose(self):
        logging.debug("Client {}: wait for invitation ...".format(self.id))
        chose = self.comm.recv(0)
        self.comm.send(0, "ACK")
        logging.debug("Client {}: receive invitation {}".format(self.id, chose))
        return chose
    
    def train(self):
        """
            Training for 1 round.
        """        
        # Download the global model parameters
        global_model, weight_range = self.comm.recv(0)
        logging.debug("Client {}: get global weights from server".format(self.id))

        self.unserialize_model(global_model)
        logging.debug("Client {}: training({}) ...".format(self.id, len(self.train_loader)))
        for ep in range(self.epoch):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                # Training
                data, target = data.to(DEVICE), target.to(DEVICE)
                self.optimizer.zero_grad()
                output = self.model(data)
                if self.id in param.BAD_CLIENTS:
                    loss = -self.criterion(output, target)
                else:
                    loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        return weight_range

    def handle_other_parameter(self):
        idx2param = self.comm.recv(0)
        logging.debug("Client {} got other clients' parameter from server".format(self.id))
        param_list = [value for key, value in idx2param.items()]
        #return torch.mean(torch.stack(param_list), dim=0)
        return compute_geometric_median(param_list).median
    
    def handle_weights(self, weight_range, eps):
        """
            Add noise on weights.
        """
        global_model = self.serialize_model(type="raw")
        for idx in range(len(global_model)):
            logging.debug("Client {}: [{}] c={}, r={}".format(self.id, idx, weight_range[idx]["center"], weight_range[idx]["range"]))
            global_model[idx] = data_pertubation(global_model[idx], weight_range[idx]["center"], weight_range[idx]["range"], eps)
        return global_model
    
    def send_weights(self, global_model):
        latency = [(random.uniform(0, param.LATENCY_T), i) for i in range(len(global_model))]
        latency.sort()
        for idx, (lt, ln) in enumerate(latency):
            if idx == 0:
                time.sleep(lt)
            else:
                time.sleep(lt - latency[idx - 1][0])
            logging.debug("Send layer {} = {}".format(ln, global_model[ln][:10]))
            self.comm.send(0, {"ln": ln, "weight": global_model[ln]})
            if idx < len(latency) - 1:
                assert self.comm.recv(0) == "ACK"

        
    def evaluate(self):
        """
            Train the model on all rounds.
        """
        self.comm.initialize()
        for rn in range(self.round):
            if self.chose():
                if self.id in param.BAD_CLIENTS :
                    if self.id in param.TAPPING_CLIENTS:
                        weight_range = self.train()
                        bad_model_param = self.handle_weights(weight_range, param.EPS)
                        bad_model_param = torch.cat(bad_model_param)
                        good_result = self.handle_other_parameter()
                        logging.debug("Client {} bad model parameter shape : {}, other model parameter shape : {}".format(self.id, bad_model_param.shape, good_result.shape))
                        return_param = (1 - param.ATTACK_GAMMA) * good_result + param.ATTACK_GAMMA * bad_model_param
                        self.unserialize_model(return_param)
                        self.send_weights(self.serialize_model(type="raw"))
                    else :
                        weight_range = self.train()
                        global_model = self.handle_weights(weight_range, param.EPS)
                        self.send_weights(global_model)
                else :
                    weight_range = self.train()
                    global_model = self.handle_weights(weight_range, param.EPS)
                    self.send_weights(global_model)
                    
                