import torch
from tqdm import tqdm

from util import load_dataset
from model import load_criterion
import param
from base_server import Base_server

class FedAvg_server(Base_server):
    
    def __init__(self):
        super().__init__(param.N_NODES, param.MODEL, param.MODEL_PARAM, param.N_EPOCH)

        self.model_size = []
        sd = self.model.state_dict()
        _, self.test_loader = load_dataset(param.DATASET, param.FOLDER, "fl", 0, False)
        self.criterion = load_criterion(param.CRITERION)
        for x in sd:
            self.model_size.append(sd[x].numel())
            
    def train(self, ep):
        batch_length = int(self.comm.recv(1, 1, torch.int32))
        for idx in range(2, self.size):
            assert self.comm.recv(idx, 1, torch.int32) == batch_length
 
        for bn in tqdm(range(batch_length), desc="Epoch {}".format(ep)):
            # Broadcase the current model param
            model_param = self.serialize_model()
            for idx in range(1, self.size):
                self.comm.send(idx, model_param)

            # Aggregate the model param of all clients
            res = []
            for idx in range(1, self.size):
                up = self.comm.recv(idx, sum(self.model_size))
                res.append(up)

            res = torch.stack(res)
            res = torch.mean(res, dim=0)
            self.unserialize_model(res)

    def evaluate(self):
        for ep in range(self.epoch):
            self.train(ep)
            self.test(ep)