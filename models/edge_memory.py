from typing import Callable, Tuple

import torch
from torch import Tensor
from torch.nn import Linear, GRUCell

from .time_encoder import TimeEncoder

from torch_geometric.nn.inits import zeros


class EdgeMemory(torch.nn.Module):
    def __init__(self, num_edges: int, raw_msg_dim: int, memory_dim: int,
                 time_dim: int, message_module: Callable,
                 aggregator_module: Callable):
        super(EdgeMemory, self).__init__()

        self.num_edges = num_edges
        self.raw_msg_dim = raw_msg_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim

        self.current_id = 0

        # self.msg_s_module = message_module
        # self.msg_d_module = copy.deepcopy(message_module)
        self.msg_module = message_module
        self.aggr_module = aggregator_module
        self.time_enc = TimeEncoder(time_dim)
        self.gru = GRUCell(message_module.out_channels, memory_dim)

        self.register_buffer('memory', torch.empty(num_edges, memory_dim))
        last_update = torch.empty(self.num_edges, dtype=torch.long)
        self.register_buffer('last_update', last_update)
        self.register_buffer('__assoc__',
                             torch.empty(num_edges, dtype=torch.long))
        # self.register_buffer('__mapper__', torch.empty((num_edges, num_edges), dtype=torch.long))

        self.msg_store = {}
        # self.msg_s_store = {}
        # self.msg_d_store = {}

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.msg_module, 'reset_parameters'):
            self.msg_module.reset_parameters()
        if hasattr(self.aggr_module, 'reset_parameters'):
            self.aggr_module.reset_parameters()
        self.time_enc.reset_parameters()
        self.gru.reset_parameters()
        self.reset_state()

    def reset_state(self):
        """Resets the memory to its initial state."""
        zeros(self.memory)
        zeros(self.last_update)
        self.__reset_message_store__()

    def detach(self):
        """Detachs the memory from gradient computation."""
        self.memory.detach_()

    def forward(self, e_id : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns, for all nodes :obj:`n_id`, their current memory and their
        last updated timestamp."""
        # print("forward: e_id.size():", e_id.size())
        # print("forward: self.memory.size():", self.memory.size())
        if self.training:
            memory, last_update = self.__get_updated_memory__(e_id)
        else:
            memory, last_update = self.memory[e_id], self.last_update[e_id]

        return memory, last_update

    def update_state(self, e_id, t, raw_msg):
        """Updates the memory with newly encountered interactions"""

        if self.training:
            self.__update_memory__(e_id)
            self.__update_msg_store__(e_id, t, raw_msg, self.msg_store)
        else:
            self.__update_msg_store__(e_id, t, raw_msg, self.msg_store)
            self.__update_memory__(e_id)

    def __reset_message_store__(self):
        i = self.memory.new_empty((0, ), dtype=torch.long)
        msg = self.memory.new_empty((0, self.raw_msg_dim))
        # Message store format: (src, dst, t, msg)
        self.msg_store = {j: (i, i, msg) for j in range(self.num_edges)}

    def __update_memory__(self, e_id):
        memory, last_update = self.__get_updated_memory__(e_id)
        self.memory[e_id] = memory
        self.last_update[e_id] = last_update

    def __get_updated_memory__(self, e_id):
        self.__assoc__[e_id] = torch.arange(e_id.size(0), device=e_id.device)

        # Compute messages (src -> dst).
        msg, t, msg_e_id = self.__compute_msg__(
            e_id, self.msg_store, self.msg_module)

        # print("__get_updated_memry__:msg.size():", msg.size())
        # print("__get_updated_memry__:t.size():", t.size())
        # print("__get_updated_memry__:e_id.size():", e_id.size())


        # Aggregate messages.
        # idx = torch.cat([src_s, src_d], dim=0)
        # msg = torch.cat([msg_s, msg_d], dim=0)
        # t = torch.cat([t_s, t_d], dim=0)
        aggr = self.aggr_module(msg, self.__assoc__[msg_e_id], t, e_id.size(0))

        # Get local copy of updated memory.
        memory = self.gru(aggr, self.memory[e_id])

        # Get local copy of updated `last_update`.
        last_update = self.last_update.scatter(0, msg_e_id, t)[e_id]

        return memory, last_update

    def __update_msg_store__(self, edge_id, t, raw_msg, msg_store):
        e_id, perm = edge_id.sort()
        e_id, count = e_id.unique_consecutive(return_counts=True)
        for i, idx in zip(e_id.tolist(), perm.split(count.tolist())):
            msg_store[i] = (edge_id[idx], t[idx], raw_msg[idx])

    def __compute_msg__(self, e_id, msg_store, msg_module):
        # print(msg_store)
        data = [msg_store[i] for i in e_id.tolist()]
        # print("__compute_msg__: data:", data)
        e_id, t, raw_msg = list(zip(*data))
        # print("__compute_msg__: e_id:", e_id)
        # print("__compute_msg__: t:", t)
        # print("__compute_msg__: raw_msg:", raw_msg)
        e_id = torch.cat(e_id, dim=0)
        t = torch.cat(t, dim=0)
        raw_msg = torch.cat(raw_msg, dim=0)
        # print("__compute_msg__: e_id.size():", e_id.size())
        # print("__compute_msg__: t.size():", t.size())
        # print("__compute_msg__: raw_msg.size():", raw_msg.size())
        t_rel = t - self.last_update[e_id]
        t_enc = self.time_enc(t_rel.to(raw_msg.dtype))

        # print("__compute_msg__: t_rel.size():", t_rel.size())
        # print("__compute_msg__: t_enc.size():", t_enc.size())

        msg = msg_module(raw_msg, t_enc)

        return msg, t, e_id

    def train(self, mode: bool = True):
        """Sets the module in training mode."""
        if self.training and not mode:
            # Flush message store to memory in case we just entered eval mode.
            self.__update_memory__(
                torch.arange(self.num_edges, device=self.memory.device))
            self.__reset_message_store__()
        super(EdgeMemory, self).train(mode)