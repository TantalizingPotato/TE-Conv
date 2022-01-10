# import numpy as np
# import networkx as nx
#
# def get_features_sp_sample(G, node_set, max_sp):
#     dim = max_sp + 2
#     set_size = len(node_set)
#     sp_length = np.ones((G.number_of_nodes(), set_size), dtype=np.int32) * -1
#     for i, node in enumerate(node_set):
#         for node_ngh, length in nx.shortest_path_length(G, source=node).items():
#             sp_length[node_ngh, i] = length
#     sp_length = np.minimum(sp_length, max_sp)
#     onehot_encoding = np.eye(dim, dtype=np.float64)  # [n_features, n_features]
#     features_sp = onehot_encoding[sp_length].sum(axis=1)
#     return features_sp
#
# def get_features_rw_sample(adj, node_set, rw_depth):
#     epsilon = 1e-6
#     adj = adj / (adj.sum(1, keepdims=True) + epsilon)
#     rw_list = [np.identity(adj.shape[0])[node_set]]
#     for _ in range(rw_depth):
#         rw = np.matmul(rw_list[-1], adj)
#         rw_list.append(rw)
#     features_rw_tmp = np.stack(rw_list, axis=2)  # shape [set_size, N, F]
#     # pooling
#     features_rw = features_rw_tmp.sum(axis=0)
#     return features_rw