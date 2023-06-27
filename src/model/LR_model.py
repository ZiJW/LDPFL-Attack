import torch.nn as nn
import torch
import torch.nn.functional as F

class LR(nn.Module):
    """
        Logistic Regression
    """
    def __init__(self, input_size, output_size):
        super().__init__()
        
        self.linear = nn.Linear(input_size, output_size)
        self.sigmod = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.sigmod(x)
        return x