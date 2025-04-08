import model.ResNet_model
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import model.MLP_model
import model.ConvNet_model
import model.VGG_model
import model.LR_model
import torchvision.models as models

from param import DEVICE

LR = model.LR_model.LR
MLP = model.MLP_model.MLP
MLP_with_Conf = model.MLP_model.MLP_with_Conf
ConvNet = model.ConvNet_model.ConvNet
VGG_Mini = model.VGG_model.VGG_Mini
VGG16 = model.VGG_model.VGG16
ResNet = model.ResNet_model.ResNet


def load_model(Model: str, Param: dict):
    if Model == "None":
        return None
    elif Model == "LR":
        return LR(Param["input_size"], Param["output_size"]).to(DEVICE)
    elif Model == "MLP":
        return MLP(Param["input_size"], Param["output_size"]).to(DEVICE)
    elif Model == "ConvNet":
        return ConvNet().to(DEVICE)
    elif Model == "VGG16":
        return VGG16(Param["output_size"], Param["channel"]).to(DEVICE)
    elif Model == "VGG_Mini":
        return VGG_Mini(Param["output_size"], Param["channel"]).to(DEVICE)
    elif Model == "MLP_with_Conf":
        return MLP_with_Conf(Param["input_size"], Param["output_size"]).to(DEVICE)
    elif Model == "ResNet-18":
        return ResNet(model.ResNet_model.BasicBlock, [2, 2, 2, 2], Param["output_size"], Param["channel"]).to(DEVICE)
    else:
        raise ValueError("Unknown model type: \"{}\"".format(Model))

# ---------------------- Load Loss Functions ----------------

def load_criterion(Criterion: str, Alpha : float = 0.0, Temp : float = 0.0):
    if Criterion == "CrossEntropy":
        return nn.CrossEntropyLoss()
    elif Criterion == "BCELoss":
        return nn.BCELoss()
    elif Criterion == "CrossEntropyLoss_with_distillation":
        return CrossEntropyLoss_with_distillation(nn.CrossEntropyLoss(), Alpha, Temp)
    else:
        raise ValueError("Unknown criterion type: \"{}\"".format(Criterion))

class CrossEntropyLoss_with_distillation(nn.Module):
    def __init__(self, Criterion, Alpha, Temp) -> None:
        super().__init__()
        self.criterion = Criterion
        self.alpha = Alpha
        self.temp = Temp
        self.KL = nn.KLDivLoss(reduction='batchmean')

    def forward(self, input, target, teacher=None):
        if teacher == None:
            loss = self.criterion(input, target)
        else:
            loss = self.alpha * self.criterion(input, target)
            loss += (1.0 - self.alpha) * self.KL(F.log_softmax(input / self.temp, dim=1), F.log_softmax(teacher / self.temp, dim=1))
        return loss

# ---------------------- Load Optimizer ----------------------

def load_optimizer(Optimizer: str, model_param, learning_rate: float):
    if Optimizer == "Adam":
        return optim.Adam(model_param, lr=learning_rate)
    elif Optimizer == "SGD":
        return optim.SGD(model_param, lr=learning_rate)
    elif Optimizer == "SGD-momentum":
        return optim.SGD(model_param, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    else:
        raise ValueError("Unknown optimizer type: \"{}\"".format(Optimizer))