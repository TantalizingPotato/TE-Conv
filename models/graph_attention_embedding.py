import torch
from torch_geometric.nn.conv import TransformerConv


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, args, num_nodes, in_channels, out_channels, msg_dim, time_enc, n_layer, device):
        super(GraphAttentionEmbedding, self).__init__()

        self.device = device

        self.num_nodes = num_nodes
        self.time_enc = time_enc
        self.edge_dim = 0

        self.args = args
        # self.time_encoding = args.time_encoding
        # self.edge_attribute = args.edge_attribute

        if args.time_encoding:
            self.edge_dim += time_enc.out_channels
        if args.edge_attribute:
            self.edge_dim += msg_dim

        self.n_layer = n_layer if n_layer > 0 else 1
        self.layers = [TransformerConv(in_channels, out_channels // 2, heads=2,
                                       dropout=0.1, edge_dim=self.edge_dim if self.edge_dim > 0 else None
                                       ).to(device)]  # lgh: "to(device) is important
        for _ in range(n_layer - 1):
            self.layers.append(TransformerConv(out_channels, out_channels // 2, heads=2,
                                               dropout=0.1, edge_dim=self.edge_dim if self.edge_dim > 0 else None
                                               ).to(device))  # lgh: "to(device) is important

    def forward(self, x, edge_index, edge_time, msg):

        edge_attr = torch.zeros(edge_index.size(-1), 0).float().to(self.device)
        if self.args.time_encoding:
            # rel_t = self.last_update[edge_index[1]] - t             # lgh: edge_index[0] -> edge_index[1]
            rel_t_enc = self.time_enc(edge_time.to(x.dtype))
            edge_attr = torch.cat([edge_attr, rel_t_enc], dim=-1)
        if self.args.edge_attribute:
            edge_attr = torch.cat([edge_attr, msg], dim=-1)

        if self.edge_dim > 0:
            for layer in self.layers:
                x = layer(x, edge_index, edge_attr)
        else:
            for layer in self.layers:
                x = layer(x, edge_index)

        # self.last_update[edge_index[1]] = t          # lgh: edge_index[0] -> edge_index[1]

        return x

    # def reset_timer(self):
    #     self.last_update = torch.zeros(self.num_nodes, dtype=torch.long)