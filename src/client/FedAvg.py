import torch
import logging
from base_client import Base_client
from model import load_model, load_criterion, load_optimizer
from util import load_dataset, ExpM, pm_perturbation
from param import DEVICE
import param
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from model.ResNet_model import ResNet18
from torch.utils.data import TensorDataset, DataLoader
import math

class FedAvg_client(Base_client):
    """
    Test
    """

    def __init__(self, id):
        super().__init__(id, param.N_NODES, param.MODEL, param.MODEL_PARAM,
                         param.OPTIMIZER, param.LEARNING_RATE, param.CRITERION, comm=param.COMM)
        
        # Data
        """transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root='./dataset', train=True, download=True, transform=transform_train)
        size = len(trainset)
        each_size = int(math.floor(size / (param.N_NODES - 1)))
        indice = list(range((self.id - 1) * each_size, self.id * each_size))
        self.train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(trainset, indice), batch_size=128, shuffle=True, num_workers=2)
        logging.debug("Client {} get {} data in CIFAR10".format(self.id, range((self.id - 1) * each_size, self.id * each_size)))

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        testset = torchvision.datasets.CIFAR10(
            root='./dataset', train=False, download=True, transform=transform_test)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=1000, shuffle=False, num_workers=2)"""

        self.train_loader = load_dataset(param.DATASET, param.FOLDER, [("train", True)], idx=self.id)[0]
        self.valid_loader = load_dataset(param.DATASET, param.FOLDER, [("public", False)])[0] if id in param.BAD_CLIENTS else []
        
        #self.mdoel = ResNet18().to(param.DEVICE)
        #self.criterion = nn.CrossEntropyLoss()
        #self.optimizer = optim.SGD(self.model.parameters(), lr=param.LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        #self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[60, 120, 160], gamma=0.2)

    def receive_global_model(self):
        logging.debug("Client {}: wait for invitation ...".format(self.id))

        chose = self.comm.recv(0)
        if not chose:
            self.comm.send(0, "ACK")
            return
        
        logging.debug("Client {} : receive invitation {}".format(self.id, chose))
        self.comm.send(0, "ACK")

        # Download the global model parameters
        global_model = self.comm.recv(0)
        logging.debug("Client {}: get global weights from server: {}".format(self.id, global_model))
        return global_model

    def send2server(self, msg):
        self.comm.send(0, msg)
        assert self.comm.recv(0) == "ACK"
        logging.debug("Client {}: send local grads to server".format(self.id))

    def train(self, round, epoch):
        """
        Training for one round.
        """
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        if round != 0 and param.LEARNING_RATE_LIST[round] != param.LEARNING_RATE_LIST[round - 1]:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param.LEARNING_RATE_LIST[round]
        lr_list = []
        for param_group in self.optimizer.param_groups:
            lr_list.append(param_group['lr'])
        logging.debug("Round {} lr :{}".format((round, epoch), lr_list))
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        # Update scheduler
        #self.scheduler.step()
        logging.debug("Epoch {} train | Acc: {} | Loss: {}".format((round, epoch), 100.*correct/total, train_loss/len(self.train_loader)))

    def test(self, epoch):
        """
        Testing the model accuracy and loss on the test set.
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        logging.debug("Epoch {} test | Acc: {} | Loss: {}".format(epoch, 100.*correct/total, test_loss / len(self.test_loader)))
        logging.debug(" ")
        
    def evaluate(self):
        """
        Train the model for all rounds.
        """
        self.comm.initialize()
        for iter in range(param.N_ROUND):
            global_model = self.receive_global_model()
            self.unserialize_model(global_model)
            for epoch in range(param.N_EPOCH):
                self.train(iter, epoch)
            res = self.serialize_model()
            self.send2server(res)
