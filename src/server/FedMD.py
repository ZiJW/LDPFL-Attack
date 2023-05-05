import torch
from tqdm import tqdm

from util import load_dataset, load_criterion
import param
from base_server import Base_server

class FedMD_server(Base_server):
    def __init__(self, size, Model, Model_param, Epoch):
        super().__init__(size, Model, Model_param, Epoch)

    def train(self, ep):
        batch_length = int(self.comm.recv(1, 1, torch.int32))
        for idx in range(2, self.size):
            assert self.comm.recv(idx, 1, torch.int32) == batch_length
        
        # print("--> {} rounds".format(batch_length))

        for bn in tqdm(range(batch_length), desc="(Server) Epoch {}".format(ep)):
            # Aggregate the logits of all clients
            shape = list(self.comm.recv(1, 2, torch.int32))
            # print("--> {}".format(shape))

            res = []
            for idx in range(1, self.size):
                res.append(self.comm.recv(idx, shape))
            res = torch.stack(res)
            res = torch.mean(res, dim=0)

            # Broadcast the average logits
            for idx in range(1, self.size):
                self.comm.send(idx, res)

    def evaluate(self):
        for ep in range(self.epoch):
            self.train(ep)