import torch
import logging
from base_client import Base_client
from model import load_model, load_criterion, load_optimizer
from util import load_dataset, ExpM, pm_perturbation
from param import DEVICE
import param

class Transf(torch.nn.Module):
    def __init__(self, input_size) -> None:
        super().__init__()
        self.params = torch.nn.ParameterDict({
            'alpha': torch.nn.Parameter(torch.rand(input_size)),
            'beta': torch.nn.Parameter(torch.rand(input_size))
        })

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return torch.add(torch.mul(x, self.params['alpha']), self.params['beta'])

class DPSGD_client(Base_client):  
    """
        The client in FedSel
    """

    def __init__(self, id):
        super().__init__(id, param.N_NODES, param.MODEL, param.MODEL_PARAM,
                         param.OPTIMIZER, param.LEARNING_RATE, param.CRITERION, comm=param.COMM)
        self.train_loader = load_dataset(param.DATASET, param.FOLDER, [("train", True)], idx=self.id)[0]
        self.valid_loader = load_dataset(param.DATASET, param.FOLDER, [("public", False)])[0] if id in param.BAD_CLIENTS else []
        self.round = param.N_ROUND
        self.privacy1_percent = 0.1
        self.layer0 = Transf(param.MODEL_PARAM["input_size"]).to(param.DEVICE)
        self.ldp = param.LDP and self.id not in param.BAD_CLIENTS

        if self.id in param.BAD_CLIENTS:
            logging.info("Client {} is an adversary!".format(self.id))

    def train(self):
        """
            Training for 1 round.
        """
        logging.debug("Client {}: wait for invitation ...".format(self.id))

        chose = self.comm.recv(0)
        if not chose:
            self.comm.send(0, "ACK")
            return
        
        logging.debug("Client {} : receive invitation {}".format(self.id, chose))
        self.comm.send(0, "ACK")

        # Download the global model parameters
        global_model, weight_range = self.comm.recv(0)
        logging.debug("Client {}: get global weights from server: {}".format(self.id, global_model))
        self.unserialize_model(global_model)

        torch.autograd.set_detect_anomaly(True)
        gradients = torch.tensor([0.]*len(global_model)).to(param.DEVICE)
        
        if self.id in param.BAD_CLIENTS:
            est_good_gradients = self.fit_on(self.train_loader, len(global_model))[1].mul((param.N_NODES-2)/(param.N_NODES-1))
            self.unserialize_model(global_model-est_good_gradients*param.LEARNING_RATE)
            std_acc = self.test_on_public(self.model)[0]*0.9

            for i in range(param.ADVERSARY_ITERATION):
                est_gradients = est_good_gradients+gradients.div(param.N_NODES-1)
                self.unserialize_model(global_model-est_gradients*param.LEARNING_RATE)
                bad_gradients = self.fit_on(self.train_loader, len(global_model), rev=True)[1]
                gradients += bad_gradients
                for j in range(10):
                    self.unserialize_model(global_model-gradients*param.LEARNING_RATE)
                    acc = self.test_on_public(self.model)[0]
                    if acc > std_acc:
                        break
                    valid_set_gradients = self.fit_on(self.valid_loader, len(global_model))[1]
                    gradients += valid_set_gradients

                acc = self.test_on_public(self.model)[0]
                logging.info("Client {}: iteration {}, valid set acc = {}".format(self.id, i, acc))
                    
        else:
            layer0_grads, train_set_gradients = self.fit_on(self.train_loader, len(global_model), transform=param.USE_TRANSFORM)
            res = []
            for val in self.layer0.state_dict().values():
                res.append(val.view(-1))
            res = torch.cat(res)
            self.unserialize_model(res-layer0_grads*param.LEARNING_RATE, self.layer0)
            self.unserialize_model(self.serialize_model()-train_set_gradients*param.LEARNING_RATE)

        res = self.serialize_model()

        logging.debug("Client {}: weights = {}".format(self.id, res))
        # logging.debug("Client {}: selected_index = {}, max absval = {}".format(self.id, selected_index, torch.max(accum_grad)))
        self.comm.send(0, res)
        assert self.comm.recv(0) == "ACK"
        # logging.debug("Client {}: send local grads to server".format(self.id))

    def fit_on(self, loader, model_size, rev = False, transform=False):
        gradients = torch.tensor([0.]*model_size).to(param.DEVICE)
        layer0_grads = torch.tensor([0.]*(2*param.MODEL_PARAM["input_size"])).to(param.DEVICE)
        cnt_samples = 0
        for data, target in loader:
            # Training
            # logging.debug("Client {}: training in batch {} with optimer={} ...".format(self.id, batch_idx, type(self.optimizer)))
            data, target = data.to(DEVICE), target.to(DEVICE)
            self.optimizer.zero_grad()
            output = self.model(self.layer0(data)) if transform is True else self.model(data)
            # logging.debug("output = {}".format(output))
            # logging.debug("target = {}".format(target))
            loss = -self.criterion(output, target) if rev else self.criterion(output, target)
            loss.backward()
            
            gradient = []
            for val in self.model.parameters():
                gradient.append(val.grad.view(-1))
            gradient = torch.cat(gradient)
            norm = torch.sqrt(torch.sum(torch.pow(gradient, 2)))
            if norm > param.NORM_BOUND:
                gradient.div_(norm.div(param.NORM_BOUND))
            #noise = self.GaussianNoise(param.SIGMA, param.NORM_BOUND, gradients.shape)
            #logging.debug("Client {} norm : {} parameter : {} with noise : {}".format(self.id, norm, gradients, noise))
            #gradient += noise
            gradient.div_(len(data))
            gradients += gradient
            cnt_samples += len(data)
        
            if transform is True:
                gradient = []
                for val in self.layer0.parameters():
                    gradient.append(val.grad.view(-1))
                gradient = torch.cat(gradient)
                norm = torch.sqrt(torch.sum(torch.pow(gradient, 2)))
                if norm > param.NORM_BOUND:
                    gradient.div_(norm.div(param.NORM_BOUND))
                gradient.div_(len(data))
                layer0_grads += gradient
        if self.ldp:
            noise = self.GaussianNoise(param.SIGMA, param.NORM_BOUND, gradients.shape) / cnt_samples
            logging.debug("Client {} norm : {} parameter : {} with noise : {}".format(self.id, norm, gradients, noise))
            gradients += noise

        return layer0_grads, gradients


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
        """
            Train the model on all rounds.
        """
        self.comm.initialize()
        for rn in range(self.round):
            self.train()