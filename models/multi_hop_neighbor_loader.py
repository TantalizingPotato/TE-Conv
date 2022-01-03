import torch

class MultiHopNeighborLoader(object):
    def __init__(self, num_nodes: int, size: int, device=None):
        self.size = size

        self.neighbors = torch.empty((num_nodes, size), dtype=torch.long,
                                     device=device)
        self.e_id = torch.empty((num_nodes, size), dtype=torch.long,
                                device=device)
        self.__assoc__ = torch.empty(num_nodes, dtype=torch.long,
                                     device=device)

        self.reset_state()

    def __call__(self, n_id, k=1):

        all_sources = all_targets = all_e_id = all_ks = n_id.new_empty(0)
        all_n_id = n_id
        neg_edge_list = []
        for i in range(k):
            neighbors = self.neighbors[n_id]
            nodes = n_id.view(-1, 1).repeat(1, self.size)
            e_id = self.e_id[n_id]

            if i == 0:
                for idx in (0,1):
                   neg_edge_list.extend([[neighbors[1-idx, i].item(), n_id[idx].item()] for i in range(neighbors.size(1))
                                         if e_id[1-idx,i] >= 0 and neighbors[1-idx,i].item() not in neighbors[idx].tolist() + [n_id[1-idx].item()] ] )

            # Filter invalid neighbors (identified by `e_id < 0`).
            mask = e_id >= 0
            neighbors, nodes, e_id = neighbors[mask], nodes[mask], e_id[mask]

            all_n_id = torch.cat([all_n_id, neighbors]).unique()
            all_sources = torch.cat([all_sources, neighbors])
            all_targets = torch.cat([all_targets, nodes])
            all_e_id = torch.cat([all_e_id, e_id])
            ks = neighbors.new_full(neighbors.size(), i+1)
            all_ks = torch.cat([all_ks, ks])

            # Filter already seen nodes
            mask = neighbors.new_empty(neighbors.size())
            # mask.map_(neighbors, lambda x,y:y not in all_targets) #lgh: map_ is only implemented on CPU tensors
            assert len(neighbors.size()) == 1 , "The tensor \"neighbors\" has more than 1 dimension"
            for i in range(len(mask)):
                mask[i] = neighbors[i] not in all_targets
            mask = mask.bool()
            n_id = neighbors[mask]

        neg_edge_list = torch.tensor(neg_edge_list, dtype=torch.long, device=device)

        # Relabel node indices.
        self.__assoc__[all_n_id] = torch.arange(all_n_id.size(0), device=n_id.device)
        all_sources, all_targets, neg_edge_list = self.__assoc__[all_sources], self.__assoc__[all_targets], self.__assoc__[neg_edge_list]

        return all_n_id, torch.stack([all_sources, all_targets]), all_e_id, all_ks, neg_edge_list

    def insert(self, src, dst):
        # Inserts newly encountered interactions into an ever growing
        # (undirected) temporal graph.

        # Collect central nodes, their neighbors and the current event ids.
        neighbors = torch.cat([src, dst], dim=0)
        nodes = torch.cat([dst, src], dim=0)
        e_id = torch.arange(self.cur_e_id, self.cur_e_id + src.size(0),
                            device=src.device).repeat(2)
        self.cur_e_id += src.numel()

        # Convert newly encountered interaction ids so that they point to
        # locations of a "dense" format of shape [num_nodes, size].
        nodes, perm = nodes.sort()
        neighbors, e_id = neighbors[perm], e_id[perm]

        n_id = nodes.unique()
        self.__assoc__[n_id] = torch.arange(n_id.numel(), device=n_id.device)

        dense_id = torch.arange(nodes.size(0), device=nodes.device) % self.size
        dense_id += self.__assoc__[nodes].mul_(self.size)

        dense_e_id = e_id.new_full((n_id.numel() * self.size, ), -1)
        dense_e_id[dense_id] = e_id
        dense_e_id = dense_e_id.view(-1, self.size)

        dense_neighbors = e_id.new_empty(n_id.numel() * self.size)
        dense_neighbors[dense_id] = neighbors
        dense_neighbors = dense_neighbors.view(-1, self.size)

        # Collect new and old interactions...
        e_id = torch.cat([self.e_id[n_id, :self.size], dense_e_id], dim=-1)
        neighbors = torch.cat(
            [self.neighbors[n_id, :self.size], dense_neighbors], dim=-1)

        # And sort them based on `e_id`.
        e_id, perm = e_id.topk(self.size, dim=-1)
        self.e_id[n_id] = e_id
        self.neighbors[n_id] = torch.gather(neighbors, 1, perm)

    def reset_state(self):
        self.cur_e_id = 0
        self.e_id.fill_(-1)