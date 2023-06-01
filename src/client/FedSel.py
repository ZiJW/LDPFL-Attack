import torch
import logging
from base_client import Base_client
from model import load_model, load_criterion, load_optimizer
from util import load_dataset, ExpM, pm_perturbation
from param import DEVICE
import param


class FedSel_client(Base_client):
    """
        The client in FedSel
    """

    def __init__(self, id):
        super().__init__(id, param.N_NODES, param.MODEL, param.MODEL_PARAM,
                         param.OPTIMIZER, param.LEARNING_RATE, param.CRITERION, comm=param.COMM)
        self.train_loader, = load_dataset(param.DATASET, param.FOLDER, [("train", True)], idx=self.id)
        self.round = param.N_ROUND
        self.privacy1_percent = 0.1

    def train(self):
        """
            Training for 1 round.
        """
        logging.debug("Client {}: wait for invitation ...".format(self.id))

        chose = self.comm.recv(0)
        
        logging.debug("Client {}: receive invitation {}".format(self.id, chose))
        if not chose:
            # Not chosen this round.
            self.comm.send(0, "ACK")
            return
        else:
            # send batch number to server
            logging.debug("Client {}: send batch number {} to server".format(self.id, len(self.train_loader)))
            self.comm.send(0, len(self.train_loader))

        batch_num = self.comm.recv(0)
        logging.debug("Client {}: receive batchnum from server".format(self.id))
        self.comm.send(0, "ACK")

        accum_grad = None

        for batch_idx, (data, target) in enumerate(self.train_loader):
            if batch_idx >= batch_num:
                break

             # Download the global model parameters
            global_model, weight_range = self.comm.recv(0)
            # logging.debug("Client {}: get global weights from server".format(self.id))
            self.unserialize_model(global_model)

            # Training
            # logging.debug("Client {}: training in batch {} with optimer={} ...".format(self.id, batch_idx, type(self.optimizer)))
            data, target = data.to(DEVICE), target.to(DEVICE)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            gradients = []
            for val in self.model.parameters():
                gradients.append(val.grad.view(-1))
            gradients = torch.cat(gradients)
            if accum_grad is None:
                accum_grad = gradients
            else:
                accum_grad += gradients
            selected_index = ExpM(accum_grad, param.EPS*self.privacy1_percent)
            # logging.debug("Client {}: selected_index = {}, max absval = {}".format(self.id, selected_index, torch.max(accum_grad)))
            selected_val = pm_perturbation(accum_grad[selected_index], param.CLIPSIZE, param.EPS-param.EPS*self.privacy1_percent)
            self.comm.send(0, (selected_index, selected_val))
            assert self.comm.recv(0) == "ACK"
            # logging.debug("Client {}: send local grads to server".format(self.id))

    def evaluate(self):
        """
            Train the model on all rounds.
        """
        self.comm.initialize()
        for rn in range(self.round):
            self.train()