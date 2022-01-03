import os.path as osp
from tqdm import tqdm
from itertools import combinations

import torch
import numpy as np
import pandas as pd
import networkx as nx

from torch_geometric.data import DataLoader, Data

from .get_features import get_features_rw_sample, get_features_sp_sample

def get_edge_dataloader(name, data, device, neighbor_loader, assoc, val_ratio=0.15, test_ratio=0.15, batch_size=200):

    # skiprows = [i for i in range(1000, 157474)]
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', name+'_events.csv')
    df = pd.read_csv(path)
    path_msg = osp.join(osp.dirname(osp.realpath(__file__)), 'data', name+'_msg.txt')
    df_msg = pd.read_csv(path_msg) if osp.isfile(path_msg) else pd.DataFrame([[] for _ in range(df.shape[0])])

    val_time, test_time = np.quantile(
        df.t.values,
        [1. - val_ratio - test_ratio, 1. - test_ratio])

    val_idx = int((df.t.values <= val_time).sum()) * 2
    test_idx = int((df.t.values <= test_time).sum()) * 2

    data_list = []

    e_id_to_edge_id = torch.zeros((data.num_events,), dtype=torch.long, device=device)
    current_edge_id = 0
    mapper = torch.zeros((data.num_nodes, data.num_nodes), dtype=torch.long, device=device)

    # Sample negative destination nodes.
    neg_dsts = torch.randint(int(df.dst.values.min()), int(df.dst.values.max() + 1), (df.src.size,),
                             dtype=torch.long, device=device)
    # for i, src, pos_dst, neg_dst, t, msg in tqdm(zip(range(dataloader.num_events), dataloader.src, dataloader.dst,
    #                                                  neg_dsts, dataloader.t, dataloader.msg), total=dataloader.num_events):
    for i, src, pos_dst, neg_dst, t, msg in tqdm(zip(range(df.src.size), torch.from_numpy(df.src.values).to(device),
                                                     torch.from_numpy(df.dst.values).to(device),
                                                     neg_dsts, torch.from_numpy(df.t.values).to(device),
                                                     torch.from_numpy(df_msg.values).float().to(device)),
                                                 total=df.src.size):
        for dst in [pos_dst, neg_dst]:
            root_ids = torch.tensor([src, dst], dtype=torch.long, device=src.device)
            n_id, edge_index, e_id, all_ks, neg_edge_list = neighbor_loader(root_ids, k=2)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            num_nodes = n_id.size(0)

            # edge_index.flip(1).t().cpu().numpy().astype(dtype=np.int32)
            all_ks_flip = all_ks.flip(-1)
            edge_list = []
            # for idx, item in enumerate(neg_edge_list.tolist()):
            #     item.append({'k': -1})
            #     edge_list.append(item)
            for idx, item in enumerate(edge_index.flip(1).t().tolist()):
                item.append({'k':all_ks_flip[idx]})
                edge_list.append(item)
            # lgh: TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
            # new_G = nx.from_edgelist(edge_index.t().cpu().numpy().astype(dtype=np.int32))
            new_G = nx.from_edgelist(edge_list)
            new_G.add_nodes_from(np.arange(num_nodes, dtype=np.int32))
            assert (new_G.number_of_nodes() == num_nodes)

            x_list = []

            # x_list.append(torch.tensor([[] for _ in range(num_nodes)], device=device))

            new_root_ids = assoc[root_ids]

            features_sp_sample = get_features_sp_sample(new_G, new_root_ids.cpu().numpy(), max_sp=3)
            features_sp_sample = torch.from_numpy(features_sp_sample).float()
            x_list.append(features_sp_sample)

            adj = np.asarray(nx.adjacency_matrix(new_G, nodelist=np.arange(new_G.number_of_nodes(),dtype=np.int32))
                             .todense().astype(np.float32))  # [n_nodes, n_nodes]
            features_rw_sample = get_features_rw_sample(adj, new_root_ids.cpu().numpy(), rw_depth=3)
            features_rw_sample = torch.from_numpy(features_rw_sample).float()
            x_list.append(features_rw_sample)

            x = torch.cat(x_list, dim=-1)

            edge_graph = nx.Graph()
            # edge_graph.add_node((assoc[src].item(),assoc[dst].item()))

            edge_index_2 =  []
            new_G.add_edge(assoc[src].item(),assoc[dst].item(), k=0)
            for a, b in combinations(new_G.edges, 2):
                if a[0] == b[0] or a[0] == b[1] or a[1] == b[0] or a[1] == b[1]:
                    edge_index_2.append((a, b))
            edge_graph.add_edges_from(edge_index_2)

            edge_graph_node_list = []
            for idx, item in enumerate(new_G.edges.data()):
                edge_node = ((item[0],item[1]), item[2])
                edge_graph_node_list.append(edge_node)
            edge_graph.add_nodes_from(edge_graph_node_list)
            assert edge_graph.number_of_nodes() == new_G.number_of_edges() , f'edge_graph.number_of_nodes():{edge_graph.number_of_nodes()}, new_G.number_of_edges(): {new_G.number_of_edges()}'

            # t_list = [torch.tensor([0], dtype=torch.float, device=device)]
            # x_list_2 = [torch.cat([x[assoc[src]],x[assoc[dst]]], dim=-1)]
            t_list = []
            x_list_2 = []
            e_id_list = []
            k_list = []
            for j in edge_graph.nodes:
                mask = (edge_index[0] == j[0]) & (edge_index[1] == j[1]) | (edge_index[0] == j[1]) & (edge_index[1] == j[0])
                # relative_e_id = e_id[mask]
                if mask.sum(0) != 0 :
                    relative_e_id = e_id[mask]
                    t_sample = t - data.t[relative_e_id].max().unsqueeze(0).float()
                    latest_e_id = relative_e_id[data.t[relative_e_id].argmax()].unsqueeze(0)
                    msg_sum = data.msg[relative_e_id].sum(0)
                else:
                    t_sample = torch.tensor([0], dtype=torch.float, device=device)
                    latest_e_id = torch.tensor([0], dtype=e_id.dtype, device=device)
                    msg_sum = torch.zeros((data.msg.size(-1),), device=device)
                t_list.append(t_sample)
                e_id_list.append(latest_e_id)
                # x_list_2.append(torch.cat([x[j[0]],x[j[1]],msg_sum], dim=-1))
                x_list_2.append(torch.cat([x[j[0]],x[j[1]]], dim=-1))
                k_list.append(torch.tensor([edge_graph.nodes[j]['k']], dtype=torch.long, device=device))
            ts = torch.stack(t_list)
            x_2 = torch.stack(x_list_2)
            e_ids = torch.stack(e_id_list)
            ks = torch.stack(k_list)

            edge_graph = nx.relabel.convert_node_labels_to_integers(edge_graph)
            # edge_index= torch.tensor(list(edge_graph.edges) + list(map(lambda x:(x[1],x[0]),list(edge_graph.edges))), device=device).t()
            edge_list_final = list(edge_graph.edges)
            edge_index= torch.tensor(edge_list_final, device=device).t() if len(edge_list_final) != 0 else torch.zeros((2,0), dtype=torch.long, device=device)
            edge_index = torch.cat((edge_index, edge_index[[1,0]]), dim=-1)
            #TODO: remove edges with k_max-hop nodes as targets

            if dst == pos_dst:
                y = torch.tensor([True], dtype=torch.long)
                t = t
                msg = msg
                if mapper[src, dst] == 0:
                    current_edge_id += 1
                    mapper[src, dst] = mapper[dst, src] = current_edge_id
                e_id_to_edge_id[i + 1] = mapper[src, pos_dst]
            else:
                y = torch.tensor([False], dtype=torch.long)
                t = t.new_full(t.size(), 0)
                msg = msg.new_full(msg.size(), 0)

            data_sample = Data(x=x_2, edge_index=edge_index, ks=ks, ts=ts, y=y, src=src, dst=dst, t=t, msg=msg, edge_id = mapper[src, dst],
                               n_id = e_id_to_edge_id[e_ids])
            data_list.append(data_sample)



        if i % batch_size == 0 and i != 0 : neighbor_loader.insert(torch.from_numpy(
            df.src.values[i - batch_size:i]).to(device), torch.from_numpy(df.dst.values[i - batch_size:i]).to(device))

    return DataLoader(data_list[:val_idx], batch_size=400, shuffle=False, pin_memory=True, num_workers=0), \
           DataLoader(data_list[val_idx:test_idx], batch_size=400, shuffle=False, pin_memory=True, num_workers=0), \
           DataLoader(data_list[test_idx:], batch_size=400, shuffle=False, pin_memory=True, num_workers=0), \
           x_2.size(-1), current_edge_id
