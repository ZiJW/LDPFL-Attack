import torch
import threading
import queue
import random

from network import socket_comm, fake_comm
from util import load_dataset
from model import load_criterion
import param
from base_server import Base_server


class FedSel_server(Base_server):
    """
        FedSel server
    """

    def __init__(self):
        super().__init__(param.N_NODES, param.MODEL, param.MODEL_PARAM, param.N_EPOCH, comm=param.COMM)
        self.round = param.N_ROUND
        self.kap = param.KAP

        self.test_loader, = load_dataset(param.DATASET, param.FOLDER, [("test", False)])
        self.criterion = load_criterion(param.CRITERION)

        self.model_size = []
        sd = self.model.state_dict()
        for x in sd:
            self.model_size.append(sd[x].numel())

        self.grads_buffer = queue.Queue(maxsize=self.size - 1)

    def collect_grads(self, idx: int):
        msg = self.comm.recv(idx)
        self.grads_buffer.put(msg)
        self.comm.send(idx, "ACK")

    def train(self, rn):
        chose = random.sample(list(range(1, self.size)), self.kap[rn])
        batch_numbers = []
        for idx in range(1, self.size):
            self.comm.send(idx, idx in chose)
            logging.debug("Server: send invitation {} to Client {} ".format(idx in chose, idx))
            if idx in chose:
                batch_numbers.append(self.comm.recv(src=idx))
            else:
                assert self.comm.recv(src=idx) == "ACK"

        min_batch_number = min(batch_numbers)
        for idx in chose:
            self.comm.send(idx, min_batch_number)
            logging.debug("Server: send min_batch_number to Client {} ".format(idx))
            assert self.comm.recv(src=idx) == "ACK"

        for batch_idx in range(min_batch_number):
            logging.debug("Server: batch {} begin".format(batch_idx))
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

            Thr = [threading.Thread(target=FedSel_server.collect_grads, args=(self, idx)) for idx in chose]
            for tr in Thr:
                tr.start()
            for tr in Thr:
                tr.join()

            # logging.debug("Server: collect grads done")

            res = torch.tensor([0.]*len(global_model)).to(param.DEVICE)
            for idx in chose:
                index, val = self.grads_buffer.get()
                res[index].data += val*param.CLIPSIZE
            
            res = res.div(param.BATCH_SIZE_TRAIN)
            res = global_model-res*param.LEARNING_RATE
            self.unserialize_model(res)

        logging.debug("Server: round {} end".format(rn))

    def evaluate(self):
        self.comm.initialize()
        for rn in range(self.round):
            self.train(rn)
            self.test(rn)