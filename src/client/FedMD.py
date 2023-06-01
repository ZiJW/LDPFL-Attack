import torch

from base_client import Base_client
from util import load_dataset
from model import load_model, load_criterion, load_optimizer
from param import DEVICE
import param

class FedMD_client(Base_client):
    """
        The client in FedMD.
    """
    
    def __init__(self, id, size, Model, Model_param, Optimizer, Learning_rate, Criterion, Dataset, Epoch, Alpha, Temp):
        super().__init__(id, size, Model, Model_param, Optimizer, Learning_rate, Criterion)
        
        self.criterion = load_criterion(Criterion, Alpha, Temp)
        self.train_loader, self.test_loader = load_dataset(Dataset, "fl", self.id)
        self.public_loader = load_dataset(Dataset, "public")
        self.epoch = Epoch

    def train(self):
        """
            Train the model on 1 epoch.
        """
        for data, target in self.train_loader:
            # Training
            data, target = data.to(DEVICE), target.to(DEVICE)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
    
    def distillation(self):
        """
            Aggregate logits by knowledge distillation
        """
        TT = torch.Tensor([len(self.public_loader)]).to(torch.int32)
        self.comm.send(0, TT)

        for batch_idx, (data, target) in enumerate(self.public_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            self.optimizer.zero_grad()
            output = self.model(data)

            # Upload the logits
            if self.id == 1:
                self.comm.send(0, torch.Tensor(list(output.shape)).to(torch.int32))

            self.comm.send(0, output)

            # Download the average logits
            teacher = self.comm.recv(0, output.shape)

            # distillation
            loss = self.criterion(output, target, teacher)            
            loss.backward()
            self.optimizer.step()

    def evaluate(self):
        """
            Train the model on all epoches.
            Test the model after every epoch.
        """
        for ep in range(self.epoch):
            print("(Client {}) training ...".format(self.id))
            self.train()
            print("(Client {}) testing ...".format(self.id))
            self.test(ep)
            self.distillation()
            print("(Client {}) testing ...".format(self.id))
            self.test(ep)