import os.path as osp
import random

import numpy as np
import pandas as pd
import torch

from data.temporal_data import TemporalData
from neighbor_finder import NeighborFinder


# train_data, val_data, test_data = data.train_val_test_split(
#     val_ratio=0.15, test_ratio=0.15)

def get_data(name, val_ratio=0.15, test_ratio=0.15, use_neg=True, use_msg=True, seed=2020, dataset_size=2000000,
             cpu_only=False, nn_ratio=0.1):
    device = torch.device('cuda' if torch.cuda.is_available() and not cpu_only else 'cpu')

    path = osp.join(osp.dirname(osp.realpath(__file__)), '', name + '.csv')
    graph_df = pd.read_csv(path, skiprows=list(range(dataset_size + 1, 157475)))
    graph_df.insert(graph_df.shape[1], 'e_id', np.arange(graph_df.shape[0]))

    path_msg = osp.join(osp.dirname(osp.realpath(__file__)), '', name + '_msg.txt')
    df_msg = pd.read_csv(path_msg, skiprows=list(range(dataset_size + 1, 157475))) if osp.isfile(
        path_msg) and use_msg else pd.DataFrame([[] for _ in range(graph_df.shape[0])])
    messages = np.append(df_msg.values, np.array([[0]]).repeat(df_msg.shape[1], axis=1), axis=0)

    if use_neg:
        path_neg = osp.join(osp.dirname(osp.realpath(__file__)), '', name + '_neg.csv')
        neg_df = pd.read_csv(path_neg)
        neg_df.insert(neg_df.shape[1], 'e_id', - np.ones(neg_df.shape[0]))
    else:
        neg_df = pd.DataFrame()
        for key in graph_df.keys():
            neg_df[key] = []

    val_time, test_time = np.quantile(
        graph_df.ts.values,
        [1. - val_ratio - test_ratio, 1. - test_ratio])

    # cut_1 = np.searchsorted(neg_df.ts.values, test_time)
    # cut_2 = np.searchsorted(neg_df.ts.values, graph_df.ts.values.max())
    # neg_df = neg_df.iloc[cut_1:cut_2, :]

    full_df = graph_df.append(neg_df)
    full_df.sort_values(by='ts', inplace=True)

    sources = full_df.u.values
    destinations = full_df.i.values
    timestamps = full_df.ts.values
    neg_mask = full_df.e_id.values == -1

    random.seed(seed)

    node_set = set(sources) | set(destinations)
    n_total_unique_nodes = len(node_set)

    n_total_unique_nodes_real = max(sources.max(), destinations.max()) + 1

    test_node_set = set(sources[(timestamps > val_time) & ~neg_mask]).union(
        set(destinations[(timestamps > val_time) & ~neg_mask]))

    # Sample nodes which we keep as new nodes (to test inductiveness), so than we have to remove all
    # their edges from training
    new_test_node_set = set(random.sample(test_node_set, int(nn_ratio * n_total_unique_nodes)))

    # Mask saying for each source and destination whether they are new test nodes
    new_test_source_mask = np.array([a in new_test_node_set for a in sources]) & ~neg_mask
    new_test_destination_mask = np.array([a in new_test_node_set for a in destinations]) & ~neg_mask

    # Mask which is true for edges with both destination and source not being new test nodes (because
    # we want to remove all edges involving any new test node)
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask) & ~neg_mask

    # For train we keep edges happening before the validation time which do not involve any new node
    # used for inductiveness
    train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)

    # define the new nodes sets for testing inductiveness of the model
    train_node_set = set(sources[train_mask]).union(destinations[train_mask])
    cross_area = train_node_set & new_test_node_set
    assert len(train_node_set & new_test_node_set) == 0
    new_node_set = node_set - train_node_set

    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time) & ~neg_mask
    test_mask = (timestamps > test_time) & ~neg_mask

    val_node_set = set(sources[val_mask]).union(destinations[val_mask])
    test_node_set = set(sources[test_mask]).union(destinations[test_mask])

    edge_contains_new_node_mask = np.array(
        [(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)]) & ~neg_mask
    nn_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    nn_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    nn_val_node_set = set(sources[nn_val_mask]).union(set(destinations[nn_val_mask]))
    nn_test_node_set = set(sources[nn_test_mask]).union(set(destinations[nn_test_mask]))

    edge_contains_one_new_node_mask = np.array(
        [(a in new_node_set and b not in new_node_set or b in new_node_set and a not in new_node_set) for a, b in
         zip(sources, destinations)]) & ~neg_mask
    nn_one_val_mask = np.logical_and(val_mask, edge_contains_one_new_node_mask)
    nn_one_test_mask = np.logical_and(test_mask, edge_contains_one_new_node_mask)

    nn_one_val_node_set = set(sources[nn_one_val_mask]).union(destinations[nn_one_val_mask])
    nn_one_test_node_set = set(sources[nn_one_test_mask]).union(destinations[nn_one_test_mask])

    edge_contains_two_new_node_mask = np.array(
        [(a in new_node_set and b in new_node_set) for a, b in zip(sources, destinations)]) & ~neg_mask
    nn_two_val_mask = np.logical_and(val_mask, edge_contains_two_new_node_mask)
    nn_two_test_mask = np.logical_and(test_mask, edge_contains_two_new_node_mask)

    nn_two_val_node_set = set(sources[nn_two_val_mask]).union(destinations[nn_two_val_mask])
    nn_two_test_node_set = set(sources[nn_two_test_mask]).union(destinations[nn_two_test_mask])

    print("The {} dataset has {} interactions, involving {} different nodes".format(name, len(sources),
                                                                                    n_total_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_mask.sum(), len(train_node_set)))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
        val_mask.sum(), len(val_node_set)))
    print("The test dataset has {} interactions, involving {} different nodes".format(
        test_mask.sum(), len(test_node_set)))
    print("The new node validation dataset has {} interactions, involving {} different nodes".format(
        nn_val_mask.sum(), len(nn_val_node_set)))
    print("The new node test dataset has {} interactions, involving {} different nodes".format(
        nn_test_mask.sum(), len(nn_test_node_set)))
    print("The old vs new node validation dataset has {} interactions, involving {} different nodes".format(
        nn_one_val_mask.sum(), len(nn_one_val_node_set)))
    print("The old vs new node test dataset has {} interactions, involving {} different nodes".format(
        nn_one_test_mask.sum(), len(nn_one_test_node_set)))
    print("The new vs new node validation dataset has {} interactions, involving {} different nodes".format(
        nn_val_mask.sum(), len(nn_two_val_node_set)))
    print("The new vs new node test dataset has {} interactions, involving {} different nodes".format(
        nn_test_mask.sum(), len(nn_two_test_node_set)))

    print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(
        len(new_test_node_set)))

    neg_test_mask = np.logical_and((timestamps > test_time), neg_mask)

    neg_test_node_set = set(sources[neg_test_mask]).union(destinations[neg_test_mask])
    print("The negative observed test dataset has {} instances, involving {} different nodes".format(
        neg_test_mask.sum(), len(neg_test_node_set)))

    val_range = np.logical_and(timestamps <= test_time, timestamps > val_time)
    test_range = timestamps > test_time

    val_old_mask = np.logical_and(val_mask, ~nn_val_mask)[np.nonzero(val_range)[0]]
    test_old_mask = np.logical_and(test_mask, ~nn_test_mask)[np.nonzero(test_range)[0]]

    nn_val_mask = nn_val_mask[np.nonzero(val_range)[0]]
    nn_test_mask = nn_test_mask[np.nonzero(test_range)[0]]

    nn_one_val_mask = nn_one_val_mask[np.nonzero(val_range)[0]]
    nn_one_test_mask = nn_one_test_mask[np.nonzero(test_range)[0]]

    nn_two_val_mask = nn_two_val_mask[np.nonzero(val_range)[0]]
    nn_two_test_mask = nn_two_test_mask[np.nonzero(test_range)[0]]

    neg_test_mask = neg_test_mask[np.nonzero(test_range)[0]]

    src = torch.from_numpy(sources).to(torch.long).to(device)
    dst = torch.from_numpy(destinations).to(torch.long).to(device)
    t = torch.from_numpy(timestamps).to(torch.long).to(device)
    e_id = torch.from_numpy(full_df.e_id.values).to(torch.long).to(device)
    msg = torch.from_numpy(messages).to(torch.long).to(device)

    # data = TemporalData(src=src, dst=dst, t=t, e_id=e_id).to(device)
    data = TemporalData(src=src, dst=dst, t=t, e_id=e_id)

    num_train_events = train_mask.sum()

    # Ensure to only sample actual destination nodes as negatives.
    min_dst_idx, max_dst_idx = int(destinations.min()), int(destinations.max())

    valid_mask = torch.from_numpy(~neg_mask)
    train_mask = torch.from_numpy(train_mask)
    val_range = torch.from_numpy(val_range)
    test_range = torch.from_numpy(test_range)

    return data[valid_mask], \
           data[train_mask].prepare_t_batch(), data[val_range].prepare_t_batch(), data[test_range].prepare_t_batch(), \
           msg.to(device), t.to(device), \
           val_old_mask, test_old_mask, nn_val_mask, nn_test_mask, nn_one_val_mask, nn_one_test_mask, nn_two_val_mask, \
           nn_two_test_mask, neg_test_mask, \
           int(n_total_unique_nodes_real), num_train_events, min_dst_idx, max_dst_idx


def get_neighbor_finder(data, uniform=False, max_node_idx=None):
    # # skiprows = [i for i in range(1000, 157474)]
    # path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', name + '.csv')
    # graph_df = pd.read_csv(path, skiprows=[i for i in range(data_size+1, 157475)])
    #
    # graph_df.sort_values(by='ts', inplace=True)
    #
    # # sources = graph_df.src.values
    # # destinations = graph_df.dst.values
    # # # edge_idxs = graph_df.idx.values
    # # # labels = graph_df.y.values
    # # timestamps = graph_df.t.values
    # # # neg_flags = graph_df.neg_flag.values
    #
    # sources = graph_df.u.values
    # destinations = graph_df.i.values
    # # edge_idxs = graph_df.idx.values
    # labels = graph_df.label.values
    # timestamps = graph_df.ts.values
    # # neg_flags = graph_df.neg_flag.values

    valid_mask = data.e_id != -1

    sources = data.src[valid_mask].cpu().numpy()
    destinations = data.dst[valid_mask].cpu().numpy()
    timestamps = data.t[valid_mask].cpu().numpy()

    max_node_idx = max(sources.max(), destinations.max()) if max_node_idx is None else max_node_idx

    adj_list = [[] for _ in range(max_node_idx + 1)]
    edge_idx_to_others = {}

    for source, destination, edge_idx, timestamp in zip(sources, destinations,
                                                        range(len(sources)),
                                                        timestamps):
        adj_list[source].append((destination, edge_idx, timestamp))
        adj_list[destination].append((source, edge_idx, timestamp))
        edge_idx_to_others.update({edge_idx: (source, destination, timestamp)})  # Added by lgh

    return NeighborFinder(adj_list, edge_idx_to_others, uniform=uniform)  # Added by lgh: "edge_idx_to_others"


def compute_time_statistics(sources, destinations, timestamps):
    class Statistics:
        def __init__(self):
            pass

    statistics = Statistics

    if isinstance(sources[0], torch.Tensor):
        sources = sources.cpu().numpy()
    if isinstance(destinations[0], torch.Tensor):
        destinations = destinations.cpu().numpy()
    if isinstance(timestamps[0], torch.Tensor):
        timestamps = timestamps.cpu().numpy()

    last_timestamp_sources = dict()
    last_timestamp_dst = dict()
    all_timediffs_src = []
    all_timediffs_dst = []
    for k in range(len(sources)):
        source_id = sources[k]
        dest_id = destinations[k]
        c_timestamp = timestamps[k]
        if source_id not in last_timestamp_sources.keys():
            last_timestamp_sources[source_id] = 0
        if dest_id not in last_timestamp_dst.keys():
            last_timestamp_dst[dest_id] = 0
        all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
        all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
        last_timestamp_sources[source_id] = c_timestamp
        last_timestamp_dst[dest_id] = c_timestamp
    assert len(all_timediffs_src) == len(sources)
    assert len(all_timediffs_dst) == len(sources)
    statistics.mean_time_shift_src = np.mean(all_timediffs_src)
    statistics.std_time_shift_src = np.std(all_timediffs_src)
    statistics.mean_time_shift_dst = np.mean(all_timediffs_dst)
    statistics.std_time_shift_dst = np.std(all_timediffs_dst)
    statistics.mean_time_shift = np.mean(all_timediffs_src + all_timediffs_dst)
    statistics.std_time_shift = np.std(all_timediffs_src + all_timediffs_dst)

    return statistics
