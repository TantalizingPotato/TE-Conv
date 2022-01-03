import torch
from conv import ResGatedGraphConv, TransformerConv
from models import EdgeLinkPredictor

from torch_geometric.nn import TAGConv

class E2EModel(torch.nn.Module):
    def  __init__(self, in_channels, embedding_dim, msg_dim, time_enc, device):
        super(E2EModel, self).__init__()
        self.time_enc = time_enc
        # edge_dim = msg_dim + time_enc.out_channels
        # self.conv = TransformerConv(in_channels, embedding_dim // 2, heads=2,
        #                             dropout=0.1)
        # self.conv = ResGatedGraphConv(in_channels, embedding_dim)
        # self.conv_2 = ResGatedGraphConv(embedding_dim, embedding_dim)
        self.conv = TAGConv(in_channels, embedding_dim, k=1)
        self.conv_2 = ResGatedGraphConv(embedding_dim, embedding_dim, k=1)
        self.link_pred = EdgeLinkPredictor(in_channels=embedding_dim).to(device)

    def forward(self, batch, x, last_update):
        edge_index = batch.edge_index
        # edge_time = batch.edge_time
        # edge_attr = batch.edge_attr
        # rel_t = last_update[edge_index[0]] - edge_time
        rel_t_enc = self.time_enc(batch.ts)
        # edge_attr = torch.cat([rel_t_enc, edge_attr], dim=-1)

        x = torch.cat([batch.x, rel_t_enc, x],dim=-1)
        # print("edge_index.size():", edge_index.size())
        # print("edge_attr.size():", edge_attr.size())
        x = self.conv(x, edge_index)
        x = self.conv_2(x, edge_index)
        x = self.get_root_nodes_embeddings(x, batch)
        return self.link_pred(x)

    def get_root_nodes_embeddings(self, x, batch):
        device = x.device
        # set_indices, batch, num_graphs = batch.set_indices, batch.batch, batch.num_graphs
        # set_indices = set_indices.view(-1, 2)
        # num_nodes = torch.eye(num_graphs)[batch].to(device).sum(dim=0)
        _ , num_nodes = batch.batch.unique_consecutive(return_counts=True)
        assert len(num_nodes) == batch.num_graphs
        zero = torch.tensor([0], dtype=torch.long).to(device)
        indices = torch.cat([zero, torch.cumsum(num_nodes, dim=0, dtype=torch.long)[:-1]])
        # index_bases = index_bases.unsqueeze(1).expand(-1, set_indices.size(-1))
        # assert (index_bases.size(0) == set_indices.size(0)), f"index_bases.size(): {index_bases.size()}, set_indices.size(): {set_indices.size()}"
        # set_indices_batch = index_bases + set_indices
        # print('set_indices shape', set_indices.shape, 'index_bases shape', index_bases.shape, 'x shape:', x.shape)
        x = x[indices]  # shape [B, F]
        return x