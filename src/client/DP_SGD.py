import torch
import logging
from base_client import Base_client
from model import load_model, load_criterion, load_optimizer
from util import load_dataset, ExpM, pm_perturbation
from param import DEVICE
import param


class DPSGD_client(Base_client):
    """
        The client in FedSel
    """

    def __init__(self, id):
        super().__init__(id, param.N_NODES, param.MODEL, param.MODEL_PARAM,
                         param.OPTIMIZER, param.LEARNING_RATE, param.CRITERION, comm=param.COMM)
        self.train_loader, = load_dataset(param.DATASET, param.FOLDER, [("train", True)], idx=self.id)
        self.round = param.N_ROUND
        self.privacy1_percent = 0.1

        if self.id in param.BAD_CLIENTS:
            logging.info("Client {} is an adversary!".format(self.id))

    def train(self):
        """
            Training for 1 round.
        """
        logging.debug("Client {}: wait for invitation ...".format(self.id))

        chose = self.comm.recv(0)
        
        logging.debug("Client {} : receive invitation {}".format(self.id, chose))
        self.comm.send(0, "ACK")

        # Download the global model parameters
        global_model, weight_range = self.comm.recv(0)
        logging.debug("Client {}: get global weights from server: {}".format(self.id, global_model))
        self.unserialize_model(global_model)

        torch.autograd.set_detect_anomaly(True)
        gradients = torch.tensor([0.]*len(global_model)).to(param.DEVICE)
        batch_size = 0
        for data, target in self.train_loader:
            # Training
            # logging.debug("Client {}: training in batch {} with optimer={} ...".format(self.id, batch_idx, type(self.optimizer)))
            data, target = data.to(DEVICE), target.to(DEVICE)
            self.optimizer.zero_grad()
            output = self.model(data)
            # logging.debug("output = {}".format(output))
            # logging.debug("target = {}".format(target))
            loss = -self.criterion(output, target) if self.id in param.BAD_CLIENTS else self.criterion(output, target)
            loss.backward()
            
            gradient = []
            for val in self.model.parameters():
                gradient.append(val.grad.view(-1))
            gradient = torch.cat(gradient)
            norm = torch.sqrt(torch.sum(torch.pow(gradient, 2)))
            if norm > param.NORM_BOUND:
                gradient.div(norm.div(param.NORM_BOUND))

            batch_size = batch_size + 1
            gradients += gradient

        gradients.div(batch_size)
        res = global_model-gradients*param.LEARNING_RATE

        logging.debug("Client {}: norm = {}, weights = {}".format(self.id, norm, res))
        # logging.debug("Client {}: selected_index = {}, max absval = {}".format(self.id, selected_index, torch.max(accum_grad)))
        self.comm.send(0, res)
        assert self.comm.recv(0) == "ACK"
        # logging.debug("Client {}: send local grads to server".format(self.id))

    def evaluate(self):
        """
            Train the model on all rounds.
        """
        self.comm.initialize()
        for rn in range(self.round):
            self.train()