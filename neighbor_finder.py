import numpy as np
import torch


class NeighborFinder:
  def __init__(self, adj_list, edge_idx_to_others=None, uniform=False, seed=None):  # Added by lgh: "edge_idx_to_others=None"
    self.node_to_neighbors = []
    self.node_to_edge_idxs = []
    self.node_to_edge_timestamps = []

    # self.max_edge_idx = 0
    self.edge_idx_to_others = edge_idx_to_others   #Added by lgh

    for i, neighbors in enumerate(adj_list):
      # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
      # We sort the list based on timestamp
      sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
      self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
      self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
      self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

      # # Added by lgh
      # max_edge_idx_temp = max(self.node_to_edge_idxs)
      # if max_edge_idx_temp > self.max_edge_idx:
      #   self.max_edge_idx = max_edge_idx_temp

    self.uniform = uniform

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def find_before(self, src_idx, cut_time):
    """
    Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

    Returns 3 lists: neighbors, edge_idxs, timestamps

    """
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

    return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """
    assert (len(source_nodes) == len(timestamps)) , f"len(source_nodes) = {len(source_nodes)}, len(timestamps) = {len(timestamps)}"

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    # NB! All interactions described in these matrices are sorted in each row by time
    neighbors = - np.ones((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
    edge_times = - np.ones((len(source_nodes), tmp_n_neighbors)).astype(
      np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
    edge_idxs = - np.ones((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                   timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

      if len(source_neighbors) > 0 and n_neighbors > 0:
        if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
          sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

          neighbors[i, :] = source_neighbors[sampled_idx]
          edge_times[i, :] = source_edge_times[sampled_idx]
          edge_idxs[i, :] = source_edge_idxs[sampled_idx]

          # re-sort based on time
          pos = edge_times[i, :].argsort()
          neighbors[i, :] = neighbors[i, :][pos]
          edge_times[i, :] = edge_times[i, :][pos]
          edge_idxs[i, :] = edge_idxs[i, :][pos]
        else:
          # Take most recent interactions
          source_edge_times = source_edge_times[-n_neighbors:]
          source_neighbors = source_neighbors[-n_neighbors:]
          source_edge_idxs = source_edge_idxs[-n_neighbors:]

          assert (len(source_neighbors) <= n_neighbors)
          assert (len(source_edge_times) <= n_neighbors)
          assert (len(source_edge_idxs) <= n_neighbors)

          neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
          edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
          edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

    return neighbors, edge_idxs, edge_times

  def get_k_hop_temporal_neighbor(self, central_nodes, timestamps, n_neighbors=20, k=2):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """

    tg_edge_index_li = []
    edge_idxs_li = []
    all_root_nodes_li = []

    # tg_edge_index = np.array([[],[]])
    # tg_edge_times = np.array([])

    # all_nodes = central_nodes
    all_nodes = np.array([])

    # root_nodes = central_nodes
    # all_root_nodes = np.array([])

    cur_proxy_id = 0

    for i in range(k):
      all_nodes = np.concatenate((all_nodes, central_nodes))

      neighbors, edge_idxs, edge_times = self.get_temporal_neighbor(central_nodes, timestamps, n_neighbors)

      # valid_mask = neighbors.flatten() != -1
      valid_mask = neighbors != -1

      pre_sources_proxy = np.arange(len(central_nodes)) + cur_proxy_id
      cur_proxy_id += len(central_nodes)

      if i == 0: root_nodes = pre_sources_proxy

      # targets = np.repeat(np.expand_dims(central_nodes, axis=1), repeats=edge_idxs.shape[1], axis=1).flatten()
      targets = np.repeat(np.expand_dims(central_nodes, axis=1), repeats=edge_idxs.shape[1], axis=1)
      timestamps = np.repeat(np.expand_dims(timestamps, axis=1), repeats=edge_idxs.shape[1], axis=1)
      root_nodes = np.repeat(np.expand_dims(root_nodes, axis=1), repeats=edge_idxs.shape[1], axis=1)
      # sources = neighbors.flatten()
      # targets_proxy = np.repeat(np.expand_dims(np.arange(len(central_nodes)) + cur_proxy_id, axis=1), repeats=edge_idxs.shape[1], axis=1).flatten()

      if i > 0: tg_edge_index_li[i - 1][0] = pre_sources_proxy

      targets_proxy = np.repeat(np.expand_dims(pre_sources_proxy, axis=1), repeats=edge_idxs.shape[1], axis=1)

      sources, targets, targets_proxy, edge_idxs, timestamps, root_nodes = neighbors[valid_mask], targets[valid_mask],\
                                                   targets_proxy[valid_mask], edge_idxs[valid_mask], timestamps[valid_mask],\
                                                    root_nodes[valid_mask]


      tg_edge_index_li.append(np.vstack((sources, targets_proxy)))
      edge_idxs_li.append(edge_idxs)
      all_root_nodes_li.append(root_nodes)
      # timestamps = np.repeat(timestamps, n_neighbors)[valid_mask]
      # tg_edge_times = np.concatenate((tg_edge_times, edge_times.flatten()[valid_mask]))

      # root_nodes = np.repeat(np.expand_dims(central_nodes if i == 0 else root_nodes, axis=1), repeats=edge_idxs.shape[1], axis=1).flatten()[valid_mask]
      # all_root_nodes = np.concatenate(all_root_nodes, root_nodes)

      # all_nodes = np.concatenate((all_nodes, targets))

      central_nodes = sources
      # source_to_edge_idx = dict(zip(list(zip(neighbors.flatten(),edge_times.flatten())),edge_idxs.flatten()))

    remaining_nodes = np.unique(tg_edge_index_li[-1][0])
    all_nodes = np.concatenate((all_nodes, remaining_nodes))
    if remaining_nodes.size > 0:
      assoc_remaining_nodes = np.empty(remaining_nodes.max() + 1)

      assoc_remaining_nodes[remaining_nodes] = np.arange(remaining_nodes.size) + cur_proxy_id

      tg_edge_index_li[-1][0] = assoc_remaining_nodes[tg_edge_index_li[-1][0]]

    tg_edge_index = np.concatenate(tg_edge_index_li, axis=1)
    all_edge_idxs = np.concatenate(edge_idxs_li)
    all_root_nodes = np.concatenate(all_root_nodes_li)

    all_nodes = torch.from_numpy(all_nodes).long()
    tg_edge_index = torch.from_numpy(tg_edge_index).long()
    all_edge_idxs = torch.from_numpy(all_edge_idxs).long()
    all_root_nodes = torch.from_numpy(all_root_nodes).long()
    #
    # for _ in range(num_hops):
    #     node_mask.fill_(False)
    #     node_mask[subsets[-1]] = True
    #     torch.index_select(node_mask, 0, row, out=edge_mask)
    #     subsets.append(col[edge_mask])

    # return tg_edge_index, tg_edge_times, root_nodes, all_nodes
    return all_nodes, tg_edge_index, all_edge_idxs, all_root_nodes

  # def proxy_to_original_nodes(self, proxy):
  #   if proxy > self.max_edge_idx:
  #     return proxy - self.max_edge_idx
  #   else:
  #     return self.edge_idx_to_others[proxy]

  def get_k_hop_temporal_neighbor_by_one_timestamp(self, central_nodes, timestamp, n_neighbors=20, k=2):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """
    central_nodes = np.unique(central_nodes).astype(int)
    all_nodes = central_nodes

    tg_edge_index_li = []
    edge_idxs_li = []

    inspected_nodes_set = set()

    for i in range(k):

      timestamps = np.array([timestamp]).repeat(len(central_nodes))
      neighbors, edge_idxs, edge_times = self.get_temporal_neighbor(central_nodes, timestamps, n_neighbors)

      valid_mask = neighbors != -1

      targets = np.repeat(np.expand_dims(central_nodes, axis=1), repeats=edge_idxs.shape[1], axis=1)

      sources, targets, edge_idxs = neighbors[valid_mask], targets[valid_mask], edge_idxs[valid_mask]

      tg_edge_index_li.append(np.vstack((sources, targets)))
      edge_idxs_li.append(edge_idxs)

      inspected_nodes_set = inspected_nodes_set.union(set(targets))

      central_nodes = np.array(list(set(sources) - inspected_nodes_set))


    tg_edge_index = np.concatenate(tg_edge_index_li, axis=1).astype(int)
    all_edge_idxs = np.concatenate(edge_idxs_li).astype(int)

    all_nodes = np.unique(np.concatenate([all_nodes, tg_edge_index[0], tg_edge_index[1]])).astype(int)
    assoc = np.empty((all_nodes.max() + 1,), dtype=int)
    assoc[all_nodes] = np.arange(all_nodes.shape[0])

    tg_edge_index = assoc[tg_edge_index]

    # all_nodes = torch.from_numpy(all_nodes).long()
    # tg_edge_index = torch.from_numpy(tg_edge_index).long()
    # all_edge_idxs = torch.from_numpy(all_edge_idxs).long()

    return all_nodes, tg_edge_index, all_edge_idxs

  def get_temporal_recommendation_candidates(self, root_nodes, timestamps, k=3):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """
    if k < 1: k = 1

    all_candidates = np.array([], dtype=int)
    all_root_nodes = np.array([], dtype=int)
    all_root_id = np.array([], dtype=int)
    all_timestamps = np.array([], dtype=int)

    for cur_root_id, cur_root_node, cur_timestamp in zip(range(len(root_nodes)), root_nodes, timestamps):
      source_nodes = np.array([cur_root_node])
      candidates_cur_root_node, timestamps_cur_root_node = np.array([], dtype=int), np.array([],dtype=int)
      root_id_cur_root_node =  np.array([],dtype=int)
      for _ in range(k):
        candidates_cur_hop = np.array([],dtype=int)
        for source_node in source_nodes:
          # extracts all neighbors, interactions indexes and timestamps of
          # all interactions of user source_node happening before cut_time
          candidates, _, _ = self.find_before(source_node, cur_timestamp)
          candidates_cur_hop = np.concatenate([candidates_cur_hop, candidates])

        candidates_cur_hop_unique = np.unique(candidates_cur_hop)
        candidates_cur_root_node = np.concatenate([candidates_cur_root_node, candidates_cur_hop_unique])

        source_nodes = candidates_cur_hop_unique

      candidates_cur_root_node = np.unique(candidates_cur_root_node)
      candidates_cur_root_node = candidates_cur_root_node[candidates_cur_root_node != cur_root_node]
      root_nodes_cur_root_node = np.array([cur_root_node]).repeat(len(candidates_cur_root_node))
      root_id_cur_root_node = np.array(cur_root_id).repeat(len(candidates_cur_root_node))
      timestamps_cur_root_node = np.array([cur_timestamp]).repeat(len(candidates_cur_root_node))

      all_candidates = np.concatenate([all_candidates, candidates_cur_root_node])
      all_root_nodes = np.concatenate([all_root_nodes, root_nodes_cur_root_node])
      all_root_id = np.concatenate([all_root_id, root_id_cur_root_node])
      all_timestamps = np.concatenate([all_timestamps, timestamps_cur_root_node])


    return all_candidates, all_root_nodes, all_root_id, all_timestamps

  # cur_source_nodes = root_nodes
  # cur_root_nodes = root_nodes
  # cur_timestamps = timestamps

  # cur_source_nodes = all_candidates
  # cur_root_nodes = all_root_nodes
  # cur_timestamps = all_timestamps

  # all_candidates = np.concatenate([all_candidates, candidates_unique])
  # all_root_nodes = np.concatenate([all_root_nodes, np.array([root_node]).repeat(candidates_unique.size)])
  # all_timestamps = np.concatenate([all_timestamps, np.array([timestamp]).repeat(candidates_unique.size)])

  def get_k_hop_temporal_neighbor_bak(self, central_nodes, timestamps, n_neighbors=20, k=2):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """

    tg_edge_index = np.array([[], []])
    tg_edge_times = np.array([])
    all_nodes = central_nodes

    root_nodes = np.array([])
    all_root_nodes = np.array([])

    cur_proxy_id = 0

    for i in range(k):
      neighbors, edge_idxs, edge_times = self.get_temporal_neighbor(central_nodes, timestamps, n_neighbors)

      valid_mask = neighbors.flatten() != -1

      targets = np.repeat(np.expand_dims(central_nodes, axis=1), repeats=edge_idxs.shape[1], axis=1).flatten()
      sources = neighbors.flatten()

      sources, targets = sources[valid_mask], targets[valid_mask]

      tg_edge_index = np.concatenate((tg_edge_index, np.vstack((sources, targets))), axis=1)
      timestamps = np.repeat(timestamps, n_neighbors)[valid_mask]
      tg_edge_times = np.concatenate((tg_edge_times, edge_times.flatten()[valid_mask]))

      root_nodes = \
      np.repeat(np.expand_dims(central_nodes if i == 0 else root_nodes, axis=1), repeats=edge_idxs.shape[1],
                axis=1).flatten()[valid_mask]
      all_root_nodes = np.concatenate(all_root_nodes, root_nodes)

      all_nodes = np.concatenate(all_nodes, central_nodes)

      central_nodes = sources
      # source_to_edge_idx = dict(zip(list(zip(neighbors.flatten(),edge_times.flatten())),edge_idxs.flatten()))

    #
    # for _ in range(num_hops):
    #     node_mask.fill_(False)
    #     node_mask[subsets[-1]] = True
    #     torch.index_select(node_mask, 0, row, out=edge_mask)
    #     subsets.append(col[edge_mask])

    return tg_edge_index, tg_edge_times, root_nodes, all_nodes
