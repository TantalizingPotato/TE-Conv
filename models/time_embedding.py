import math
import torch


class TimeEmbedding(torch.nn.Module):
    def __init__(self, args, num_nodes, in_channels, out_channels, msg_dim, time_enc, n_layer, device):
        super(TimeEmbedding, self).__init__()

        self.in_channels = in_channels

        class NormalLinear(torch.nn.Linear):
            # From Jodie code
            def reset_parameters(self):
                stdv = 1. / math.sqrt(self.weight.size(1))
                self.weight.data.normal_(0, stdv)
                if self.bias is not None:
                    self.bias.data.normal_(0, stdv)

        self.embedding_layer = NormalLinear(1, self.in_channels)

    def forward(self, z, time_diffs=None):
        return z * (1 + self.embedding_layer(time_diffs.float().unsqueeze(1)))
