import torch
from torch.nn import Linear

class EdgeLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super(EdgeLinkPredictor, self).__init__()
        self.lin = Linear(in_channels, in_channels)
        # self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z):
        # h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = self.lin(z)
        h = h.relu()
        return self.lin_final(h).sigmoid()