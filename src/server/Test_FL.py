import torch
import threading
import queue
import random
import logging

from tqdm import tqdm
from network import socket_comm, fake_comm
from util import load_dataset
from model import load_criterion
import param
from base_server import Base_server


class Test_server(Base_server):
    """
        FedSel server
    """

    def __init__(self):
        super().__init__(param.N_NODES, param.MODEL, param.MODEL_PARAM, param.N_EPOCH, comm=param.COMM)
        self.round = param.N_ROUND
        self.kap = param.KAP

        self.test_loader, = load_dataset(param.DATASET, param.FOLDER, [("test", False)])
        self.valid_loader, = load_dataset(param.DATASET, param.FOLDER, [("public", False)])
        self.criterion = load_criterion(param.CRITERION)

        self.model_size = []
        sd = self.model.state_dict()
        for x in sd:
            self.model_size.append(sd[x].numel())

        self.weights_buffer = queue.Queue(maxsize=self.size - 1)

    def collect_weights(self, idx: int):
        msg = self.comm.recv(idx)
        self.weights_buffer.put((idx, msg))
        self.comm.send(idx, "ACK")

    def train(self, rn):
        num_choose = self.kap[rn]
        chose = random.sample(list(range(1, self.size)), num_choose)
        for idx in range(1, self.size):
            self.comm.send(idx, idx in chose)
            assert self.comm.recv(src=idx) == "ACK"
            logging.debug("Server: send invitation {} to Client {} ".format(idx in chose, idx))     

        global_model = self.serialize_model(type="raw")

        weight_range = []
        for weight in global_model:
            mini, maxi = torch.min(weight), torch.max(weight)
            weight_range.append({"center": (float)((mini + maxi) / 2), "range": (float)((maxi - mini) / 2)})
        global_model = torch.cat(global_model)

        if param.COMM == "socket":
            Thr = [threading.Thread(target=socket_comm.send, args=(self.comm, idx, (global_model, weight_range))) for
                idx in chose]
        elif param.COMM == "fake_socket":
            Thr = [threading.Thread(target=fake_comm.send, args=(self.comm, idx, (global_model, weight_range))) for idx
                in chose]
        else:
            raise ValueError("Invalid communication type: {}".format(param.COMM))

        for tr in Thr:
            tr.start()
        for tr in Thr:
            tr.join()

        # for idx in chose:
        #     logging.debug("Server: send global weights to Client {}".format(idx))

        Thr = [threading.Thread(target=Test_server.collect_weights, args=(self, idx)) for idx in chose]
        for tr in Thr:
            tr.start()
        for tr in Thr:
            tr.join()

        # logging.debug("Server: collect grads done")

        res = torch.tensor([0.]*len(global_model)).to(param.DEVICE)
        for _ in range(len(chose)):
            idx, val = self.weights_buffer.get()
            self.unserialize_temp_model(val)
            acc, loss = self.test_on_public(self.temp_model)
            logging.info('(Server) round {}: model from client {} Acc = {:.3f}, Loss: {:.9f}'.format(rn, idx, acc, loss))
            res += val
        
        res = res.div(num_choose)
        self.unserialize_model(res)

        logging.debug("Server: round {} end".format(rn))

    def test_on_public(self, model):
        """
            Test the accuracy and loss on validation dataset.
        """
        Acc, Loss = 0, 0.0
        LossList = []
        N = 0
        for data, target in self.valid_loader:
            with torch.no_grad():
                data, target = data.to(param.DEVICE), target.to(param.DEVICE)
                output = model(data)
                loss = self.criterion(output, target)
                _, pred = torch.max(output, dim=1)
                
                Loss += loss.item()
                LossList.append(loss.item())
                Acc += torch.sum(pred.eq(target)).item()
                N += len(target)

        Acc = Acc / N
        Loss = Loss / len(self.valid_loader)
        return Acc, Loss

    def evaluate(self):
        self.comm.initialize()
        for rn in range(self.round):
            self.train(rn)
            self.test(rn)