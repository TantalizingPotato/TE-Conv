import networkx as nx
import numpy as np
import torch

from tqdm import tqdm
from torch_geometric.data import DataLoader, Data, Batch


def get_root_nodes_indices(batch):
    # device = x.device
    set_indices, batch, num_graphs = batch.set_indices, batch.batch, batch.num_graphs
    set_indices = set_indices.view(-1, 2)
    num_nodes = torch.eye(num_graphs)[batch].sum(dim=0)
    zero = torch.tensor([0], dtype=torch.long)
    index_bases = torch.cat([zero, torch.cumsum(num_nodes, dim=0, dtype=torch.long)[:-1]])
    index_bases = index_bases.unsqueeze(1).expand(-1, set_indices.size(-1))
    assert (index_bases.size(0) == set_indices.size(
        0)), f"index_bases.size(): {index_bases.size()}, set_indices.size(): {set_indices.size()}"
    set_indices_batch = index_bases + set_indices
    # print('set_indices shape', set_indices.shape, 'index_bases shape', index_bases.shape, 'x shape:', x.shape)
    # x = x[set_indices_batch]  # shape [B, set_size, F]
    return set_indices_batch


def get_features_sp_sample(G, node_set, max_sp):
    dim = max_sp + 2
    set_size = len(node_set)
    sp_length = np.ones((G.number_of_nodes(), set_size), dtype=np.int32) * -1
    for i, node in enumerate(node_set):
        for node_ngh, length in nx.shortest_path_length(G, source=node).items():
            sp_length[node_ngh, i] = length
    sp_length = np.minimum(sp_length, max_sp)
    onehot_encoding = np.eye(dim, dtype=np.float64)  # [n_features, n_features]
    features_sp = onehot_encoding[sp_length].sum(axis=1)
    return features_sp


def get_features_rw_sample(adj, node_set, rw_depth):
    epsilon = 1e-6
    adj = adj / (adj.sum(1, keepdims=True) + epsilon)
    rw_list = [np.identity(adj.shape[0])[node_set]]
    for _ in range(rw_depth):
        rw = np.matmul(rw_list[-1], adj)
        rw_list.append(rw)
    features_rw_tmp = np.stack(rw_list, axis=2)  # shape [set_size, N, F]
    # pooling
    features_rw = features_rw_tmp.sum(axis=0)
    return features_rw


def distance_encoding(sources, destinations, timestamps, neighbor_finder, n_neighbors, USE_MEMORY_EMBEDDING,
                      DE_SHORTEST_PATH,
                      DE_RANDOM_WALK, NODE_FEATURE_DIM, full_t, full_msg, memory, gnn, k=2):
    data_list = []
    assoc = torch.empty(9000).long()

    # for i, src, dst, t in tqdm(zip(range(sources.size(0)), sources, destinations, timestamps), total=sources.size(0)):
    for src, dst, t in zip(sources, destinations, timestamps):

        root_nodes = torch.tensor([src, dst], dtype=torch.long, device=src.device)
        # n_id, edge_index, e_id = neighbor_loader(root_nodes, k=2)
        n_id, edge_index, e_id = neighbor_finder.get_k_hop_temporal_neighbor_by_one_timestamp(root_nodes.cpu().numpy(),
                                                                                              t,
                                                                                              n_neighbors=n_neighbors,
                                                                                              k=k)

        n_id, edge_index, e_id = torch.from_numpy(n_id), torch.from_numpy(edge_index), torch.from_numpy(e_id)
        num_nodes = n_id.size(0)
        # lgh: TypeError: can't convert cuda:0 device type tensor to numpy.
        # Use Tensor.cpu() to copy the tensor to host memory first.
        new_G = nx.from_edgelist(edge_index.t().cpu().numpy().astype(dtype=np.int32))
        new_G.add_nodes_from(np.arange(num_nodes, dtype=np.int32))
        assert (new_G.number_of_nodes() == num_nodes)

        x_list = [torch.zeros(num_nodes, 0).float()]

        assoc[n_id] = torch.arange(n_id.size(0))
        new_root_ids = assoc[root_nodes]

        if DE_SHORTEST_PATH:
            features_sp_sample = get_features_sp_sample(new_G, new_root_ids.cpu().numpy(), max_sp=3)
            features_sp_sample = torch.from_numpy(features_sp_sample).float()
            x_list.append(features_sp_sample)

        if DE_RANDOM_WALK:
            adj = np.asarray(nx.adjacency_matrix(new_G, nodelist=np.arange(new_G.number_of_nodes(), dtype=np.int32))
                             .todense().astype(np.float32))  # [n_nodes, n_nodes]
            features_rw_sample = get_features_rw_sample(adj, new_root_ids.cpu().numpy(), rw_depth=3)
            features_rw_sample = torch.from_numpy(features_rw_sample).float()
            x_list.append(features_rw_sample)

        if not DE_SHORTEST_PATH and not DE_RANDOM_WALK:
            x_list.append(torch.zeros(num_nodes, NODE_FEATURE_DIM).float())

        x = torch.cat(x_list, dim=-1)

        data_sample = Data(x=x, edge_index=edge_index, e_id=e_id, src=src, dst=dst, t=t, y=True,
                           set_indices=new_root_ids, n_id=n_id)
        data_list.append(data_sample)

    # data_loader = DataLoader(data_list, batch_size=sources.size(0), shuffle=False, pin_memory=True, num_workers=0)
    #
    # for batch in data_loader:

    batch = Batch.from_data_list(data_list)

    if USE_MEMORY_EMBEDDING:
        n_id = batch.n_id
        n_id_unique = n_id.unique()
        assoc[n_id_unique] = torch.arange(n_id_unique.size(0))

        x, _ = memory(n_id_unique)
        x = x[assoc[n_id]]

        x = torch.cat([x, batch.x], dim=-1)
    else:
        x = batch.x

    edge_index = batch.edge_index
    e_id = batch.e_id
    set_indices_batch = get_root_nodes_indices(batch)

    z = gnn(x, edge_index, timestamps[batch.batch[edge_index[1]]] - full_t[e_id], full_msg[e_id])

    z_set_indices = z[set_indices_batch]

    z_sources, z_destinations = z_set_indices[:, 0, :], z_set_indices[:, 1, :]

    return z_sources, z_destinations
