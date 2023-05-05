import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
        Multi-layer DNN
    """
    def __init__(self, input_size, output_size):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.act = nn.ReLU()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        return x

class MLP_with_Conf(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size + 1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        # x = F.softmax(x, dim=1)
        return x