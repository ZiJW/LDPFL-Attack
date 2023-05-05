import torch

from base_client import Base_client
from model import load_model
from util import log, load_dataset, load_criterion, load_optimizer
from param import DEVICE
import param

class FedAvg_client(Base_client):
    """
        The client in classical FedAvg
        acc = 0.676, loss = 1.937
    """
    
    def __init__(self, id):
        super().__init__(id, param.N_NODES, param.MODEL, param.MODEL_PARAM, 
                         param.OPTIMIZER, param.LEARNING_RATE, param.CRITERION, comm=param.COMM)
        self.train_loader, = load_dataset(param.DATASET, param.FOLDER, [("train", True)], idx=self.id)
        self.epoch = param.N_EPOCH

    def train(self):
        """
            Train the model on 1 epoch.
        """
        TT = torch.Tensor([len(self.train_loader)]).to(torch.int32)
        self.comm.send(0, TT)

        for batch_idx, (data, target) in enumerate(self.train_loader):
                # Download the global model parameters
                global_model = self.comm.recv(0, sum(self.model_size))
                self.unserialize_model(global_model)

                # Training
                data, target = data.to(DEVICE), target.to(DEVICE)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()

                self.optimizer.step()
                
                # Upload the local model
                global_model = self.serialize_model()
                self.comm.send(0, global_model)

    def evaluate(self):
        """
            Train the model on all epoches.
            Test the model after every epoch.
        """
        for ep in range(self.epoch):
            self.train()