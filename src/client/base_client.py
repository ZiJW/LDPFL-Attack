from abc import ABC, abstractmethod
import torch
import logging
import time
import os

from network import load_comm
from model import load_model, load_criterion, load_optimizer
from param import DEVICE
import param
import math

class Base_client(ABC):
    """
        Base client class (abstract).
    """
    def __init__(self, id, size, Model, Model_param, Optimizer, Learning_rate, Criterion, comm="dist"):
        time.sleep(id)
        
        self.log_path = param.LOG_PATH + param.LOG_NAME
        if logging.getLogger().hasHandlers():
            logging.getLogger().handlers.clear()
        os.makedirs(self.log_path, exist_ok=True)
        logging.basicConfig(filename=self.log_path + "/log_client{}.txt".format(id), format="%(asctime)s [%(levelname)s]: %(message)s", filemode="w", level=logging.DEBUG)
    
        self.id = id
        self.size = size
        self.comm = load_comm(comm, id, size)
        self.model = load_model(Model, Model_param)
        self.optimizer = load_optimizer(Optimizer, self.model.parameters(), Learning_rate)
        self.criterion = load_criterion(Criterion)

        self.model_size = []
        sd = self.model.state_dict()
        for x in sd:
            self.model_size.append(sd[x].numel())

    def serialize_model_names(self, model_keys=None, model=None, type="concat"):
        with torch.no_grad():
            res = []
            if model_keys == None:
                model_keys = self.model.state_dict().keys()
            for name in model_keys:
                val = model.state_dict()[name]
                res.append(val.view(-1))
            if type == "concat":
                res = torch.cat(res)
            elif type == "raw":
                pass
            else:
                raise ValueError("Invalid serialize type: {}".format(type))
        return res

    def unserialize_model_names(self, parameters, model=None, model_keys=None):
        with torch.no_grad():
            current_index = 0
            if model_keys == None:
                model_keys = model.state_dict().keys()
            if model == None:
                model = self.model
            for name in model_keys:
                val = model.state_dict()[name]
                sz = val.numel()
                val.copy_(parameters[current_index: current_index + sz].view(val.shape))
                current_index += sz

    def serialize_model(self, type="concat", model=None) -> torch.Tensor:
        if model == None:
            model = self.model
        model_keys = [name for name, _ in model.named_parameters()]
        return self.serialize_model_names(model_keys=model_keys, model=model, type=type)

    def unserialize_model(self, parameters: torch.Tensor, model=None):
        if model == None:
            model = self.model
        model_keys = [name for name, _ in model.named_parameters()]
        self.unserialize_model_names(parameters, model=model, model_keys=model_keys)

    def test(self, ep):
        """
            Test the accuracy and loss on testing dataset.
        """
        Acc, Loss = 0, 0.0
        LossList = []
        N = 0
        for data, target in self.test_loader:
            with torch.no_grad():
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = self.model(data)
                loss = self.criterion(output, target)
                _, pred = torch.max(output, dim=1)
                
                Loss += loss.item()
                LossList.append(loss.item())
                Acc += torch.sum(pred.eq(target)).item()
                N += len(target)

        # print("sum = {:.9f}, mean = {:.9f}, max = {:.9f}, min = {:.9f}".format(sum(LossList), sum(LossList) / len(self.test_loader), max(LossList), min(LossList)))
        Acc = Acc / N
        Loss = Loss / len(self.test_loader)
        print('(Client {}) Epoch: {} Acc = {:.3f}, Loss: {:.9f}'.format(self.id, ep, Acc, Loss))

    def GaussianNoise(self, sigma, shape):
        gaussianNoise = torch.normal(0, sigma, shape, device=param.DEVICE)
        return gaussianNoise
    
    @abstractmethod
    def evaluate():
        pass