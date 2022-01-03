import torch
from torch.nn import Linear

class TimeEncoder(torch.nn.Module):
    def __init__(self, out_channels):
        super(TimeEncoder, self).__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, t):
        return self.lin(t.view(-1, 1)).cos()