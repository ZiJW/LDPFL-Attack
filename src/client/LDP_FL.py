import random
import time

from base_client import Base_client
from util import log, load_dataset, load_model, load_criterion, load_optimizer, data_pertubation
import param
from param import DEVICE

class LDPFL_client(Base_client):
    """
        The client in LDP-FL
    """
    
    def __init__(self, id):
        super().__init__(id, param.N_NODES, param.MODEL, param.MODEL_PARAM, 
                         param.OPTIMIZER, param.LEARNING_RATE, param.CRITERION, comm=param.COMM)
        self.train_loader, = load_dataset(param.DATASET, param.FOLDER, [("train", True)], idx=self.id)
        self.epoch = param.N_EPOCH
        self.round = param.N_ROUND

    def train(self):
        """
            Training for 1 round.
        """
        log("Client {}: wait for invitation ...".format(self.id))

        chose = self.comm.recv(0)
        self.comm.send(0, "ACK")

        log("Client {}: receive invitation {}".format(self.id, chose))
        if not chose:
            # Not chosen this round.
            return
        
        # Download the global model parameters
        global_model, weight_range = self.comm.recv(0)
        log("Client {}: get global weights from server".format(self.id))
        self.unserialize_model(global_model)

        log("Client {}: training ...".format(self.id))
        for ep in range(self.epoch):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                # Training
                data, target = data.to(DEVICE), target.to(DEVICE)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

        global_model = self.serialize_model(type="raw")
        latency = [(random.uniform(0, param.LATENCY_T), i) for i in range(len(global_model))]
        latency.sort()

        for idx, (lt, ln) in enumerate(latency):
            if idx == 0:
                time.sleep(lt)
            else:
                time.sleep(lt - latency[idx - 1][0])
            data_pertubation(global_model[ln], weight_range[ln]["center"], weight_range[ln]["range"], param.EPS)
            self.comm.send(0, {"ln": ln, "weight": global_model[ln]})
            if idx < len(latency) - 1:
                assert self.comm.recv(0) == "ACK"

        log("Client {}: send local weights to server".format(self.id))
        
    def evaluate(self):
        """
            Train the model on all rounds.
        """
        self.comm.initialize()
        for rn in range(self.round):
            self.train()