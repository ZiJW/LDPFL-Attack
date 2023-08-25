from abc import ABC, abstractmethod
import torch
import logging
from tqdm import tqdm
import param
import os

from network import load_comm
from model import load_model

class Base_server(ABC):
    def __init__(self, size, Model, Model_param, Epoch, comm="dist"):
        self.log_path = param.LOG_PATH + param.LOG_NAME
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)
            os.mkdir(self.log_path + "model")
        logging.basicConfig(filename=self.log_path + "/log_server.txt", format="%(asctime)s [%(levelname)s]: %(message)s", filemode="w", 
                            level=logging.INFO)
        os.system("cp param.py " + self.log_path)

        self.epoch = Epoch
        self.size = size
        self.id = 0
        self.comm = load_comm(comm, self.id, self.size)
        self.model = load_model(Model, Model_param)
        self.temp_model = load_model(Model, Model_param)

    def serialize_model(self, type="concat") -> torch.Tensor:
        res = []
        for val in self.model.state_dict().values():
            res.append(val.view(-1))
        if type == "concat":
            res = torch.cat(res)
        elif type == "raw":
            pass
        else:
            raise ValueError("Invalid serialize type: {}".format(type))
        return res
    
    def unserialize_model(self, parameters: torch.Tensor):
        current_index = 0
        for val in self.model.state_dict().values():
            sz = val.numel()
            val.copy_(parameters[current_index: current_index + sz].view(val.shape))
            current_index += sz

    def unserialize_temp_model(self, parameters: torch.Tensor):
        current_index = 0
        for val in self.temp_model.state_dict().values():
            sz = val.numel()
            val.copy_(parameters[current_index: current_index + sz].view(val.shape))
            current_index += sz

    def test(self, ep: int = -1):
        """
            Test the accuracy and loss on testing dataset.
        """
        Acc, Loss = 0, 0.0
        LossList = []
        N = 0
        for data, target in tqdm(self.test_loader, desc="Test: "):
            with torch.no_grad():
                data, target = data.to(param.DEVICE), target.to(param.DEVICE)
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
        logging.info('(Server) Epoch: {} Acc = {:.3f}, Loss: {:.9f}'.format(ep, Acc, Loss))
        # print('(Server) Epoch: {} Acc = {:.3f}, Loss: {:.9f}'.format(ep, Acc, Loss))

    @abstractmethod
    def evaluate(self):
        pass