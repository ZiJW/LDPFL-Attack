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

        global_model = torch.cat(global_model)

        if param.COMM == "socket":
            Thr = [threading.Thread(target=socket_comm.send, args=(self.comm, idx, global_model)) for idx in chose]
        elif param.COMM == "fake_socket":
            Thr = [threading.Thread(target=fake_comm.send, args=(self.comm, idx, global_model)) for idx in chose]
        else:
            raise ValueError("Invalid communication type: {}".format(param.COMM))

        for tr in Thr:
            tr.start()
        for tr in Thr:
            tr.join()

        # for idx in chose:
        #     logging.debug("Server: send global weights to Client {}".format(idx))
        if param.TAPPING_CLIENTS == [] :
            Thr = [threading.Thread(target=Test_server.collect_weights, args=(self, idx)) for idx in chose]
            for tr in Thr:
                tr.start()
            for tr in Thr:
                tr.join()

            # logging.debug("Server: collect grads done")

            record_param = []
            for _ in range(len(chose)):
                idx, val = self.weights_buffer.get()
                self.unserialize_temp_model(global_model - param.LEARNING_RATE * val)
                acc, loss = self.test_on_public(self.temp_model)
                logging.info('(Server) round {}: model from client {} Acc = {:.3f}, Loss: {:.9f}'.format(rn, idx, acc, loss))
                record_param.append((idx, val))
        else :
            Thr = [threading.Thread(target=Test_server.collect_weights, args=(self, idx)) for idx in chose if idx not in param.TAPPING_CLIENTS]
            good_client_num = len(Thr)
            for tr in Thr:
                tr.start()
            for tr in Thr:
                tr.join()
            record_param = []
            for _ in range(good_client_num):
                idx, val = self.weights_buffer.get()
                self.unserialize_temp_model(global_model - param.LEARNING_RATE * val)
                acc, loss = self.test_on_public(self.temp_model)
                logging.info('(Server) round {}: model from client {} Acc = {:.3f}, Loss: {:.9f}'.format(rn, idx, acc, loss))
                record_param.append((idx, val))
            
            if param.COMM == "socket":
                Thr = [threading.Thread(target=socket_comm.send, args=(self.comm, idx, record_param)) for idx in chose if idx in param.TAPPING_CLIENTS]
            elif param.COMM == "fake_socket":
                Thr = [threading.Thread(target=fake_comm.send, args=(self.comm, idx, record_param)) for idx in chose if idx in param.TAPPING_CLIENTS]
            else: 
                raise ValueError("Invalid communication type: {}".format(param.COMM))
            for tr in Thr:
                tr.start()
            for tr in Thr:
                tr.join()
            logging.debug("Send other clients' parameter to tapping client")

            Thr = [threading.Thread(target=Test_server.collect_weights, args=(self, idx)) for idx in chose if idx in param.TAPPING_CLIENTS]
            tapping_client_num = len(Thr)
            for tr in Thr:
                tr.start()
            for tr in Thr:
                tr.join()
            for _ in range(tapping_client_num) :
                idx, val = self.weights_buffer.get()
                self.unserialize_temp_model(global_model - param.LEARNING_RATE * val)
                acc, loss = self.test_on_public(self.temp_model)
                logging.info('(Server) round {}: model from client {} Acc = {:.3f}, Loss: {:.9f}'.format(rn, idx, acc, loss))
                record_param.append((idx, val))
            logging.debug("Collect all clients' parameter")

        param_matrix = [val for idx, val in  sorted(record_param, key=lambda x : x[0])]
        if param.TAPPING_SAME :
            for i in param.TAPPING_CLIENTS :
                param_matrix[i - 1] = param_matrix[param.TAPPING_CLIENTS[0] - 1] # if every tapping client update same parameter, used to reduce krum score
        
        if param.MKRUM :
            selected_indices = self.select_multikrum(param_matrix, param.MAX_FAILURE, param.KRUM_SELECTED)
            logging.info("Server multi-krum select aggregating clients : {}".format(selected_indices + 1))
            bad_list_idx = torch.tensor(param.BAD_CLIENTS) - 1
            self.visualize_parameter(param_matrix, "round {}".format(rn), "{}fig/round{}.png".format(self.log_path, rn), 
                                     mode = "MDS", red_list=bad_list_idx.tolist(), blue_list=selected_indices.tolist())

            res = torch.tensor([0.]*len(global_model)).to(param.DEVICE)
            for idx in selected_indices:
                res += param_matrix[idx]
            res = res.div(selected_indices.shape[0])
            self.unserialize_model(global_model - param.LEARNING_RATE * res)
        elif param.TRIMMED_MEAN:
            #res = self.handle_trimmed_mean(param_matrix, param.TRIMMED_MEAN_BETA)
            res, select = self.trimmed_mean_with_selection_stats(param_matrix, param.TRIMMED_MEAN_BETA)
            select = [f"{x:.5f}" for x in select]
            logging.info("Server trimmed mean select ratio : {}".format(select))
            self.unserialize_model(global_model - param.LEARNING_RATE * res)
        else :
            selected_indices = torch.tensor(range(0, self.size - 1))
            logging.info("Server select aggregating clients : {}".format(selected_indices + 1))
            bad_list_idx = torch.tensor(param.BAD_CLIENTS) - 1
            self.visualize_parameter(param_matrix, "round {}".format(rn), "{}fig/round{}.png".format(self.log_path, rn), 
                                 mode = "MDS", red_list=bad_list_idx.tolist(), blue_list=selected_indices.tolist())
        

            res = torch.tensor([0.]*len(global_model)).to(param.DEVICE)
            for idx in selected_indices:
                res += param_matrix[idx]
            res = res.div(selected_indices.shape[0])
            self.unserialize_model(global_model - param.LEARNING_RATE * res)

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
        Acc, Loss = [], []
        for rn in range(self.round):
            self.train(rn)
            acc, loss = self.test(rn)
            Acc.append(acc)
            Loss.append(loss)
        self.draw(Acc, "Acc", "Accurancy")
        self.draw(Loss, "Loss", "Loss")