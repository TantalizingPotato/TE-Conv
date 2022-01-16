import copy
from typing import Callable, Tuple
from collections import defaultdict

import numpy as np
import torch
from torch import Tensor
from torch.nn import Linear, GRUCell, RNNCell

from torch_geometric.nn.inits import zeros


class TimeEncoder(torch.nn.Module):
    def __init__(self, out_channels, mode='tgn', init_parameter=True):
        super(TimeEncoder, self).__init__()
        self.mode = mode.lower()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels)
        if (self.mode == 'tgn' or self.mode == 'tgat') and init_parameter:
            self.lin.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, out_channels)))
                                                 .float().reshape(out_channels, -1))
            self.lin.bias = torch.nn.Parameter(torch.zeros(out_channels).float())

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, t):
        # if self.mode == 'dyrep' or self.mode == 'jodie':
        #     out = self.lin(t.view(-1,1))
        # else:
        #     out = self.lin(t.view(-1,1)).cos()
        out = self.lin(t.view(-1, 1)).cos()
        # out = self.lin(t.view(-1, 1))
        return out


class TGNMemory(torch.nn.Module):
    r"""The Temporal Graph Network (TGN) memory model from the
    `"Temporal Graph Networks for Deep Learning on Dynamic Graphs"
    <https://arxiv.org/abs/2006.10637>`_ paper.

    .. note::

        For an example of using TGN, see `examples/tgn.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
        tgn.py>`_.

    Args:
        num_nodes (int): The number of nodes to save memories for.
        raw_msg_dim (int): The raw message dimensionality.
        memory_dim (int): The hidden memory dimensionality.
        time_dim (int): The time encoding dimensionality.
        message_module (torch.nn.Module): The message function which
            combines source and destination node memory embeddings, the raw
            message and the time encoding.
        aggregator_module (torch.nn.Module): The message aggregator function
            which aggregates messages to the same destination into a single
            representation.
    """

    def __init__(self, num_nodes: int, raw_msg_dim: int, memory_dim: int, embedding_dim: int,
                 time_dim: int, message_module: Callable,
                 aggregator_module: Callable, mode: str = 'tgn'):
        super(TGNMemory, self).__init__()

        self.mode = mode.lower()
        self.num_nodes = num_nodes
        self.raw_msg_dim = raw_msg_dim
        self.memory_dim = memory_dim
        self.embedding_dim = embedding_dim
        self.time_dim = time_dim

        self.msg_s_module = message_module
        self.msg_d_module = copy.deepcopy(message_module)
        self.aggr_module = aggregator_module
        self.time_enc = TimeEncoder(time_dim, mode=mode)
        if self.mode == 'jodie' or self.mode == 'dyrep':
            self.gru = RNNCell(message_module.out_channels, memory_dim)
        else:
            self.gru = GRUCell(message_module.out_channels, memory_dim)

        self.register_buffer('memory', torch.empty(num_nodes, memory_dim))
        last_update = torch.empty(self.num_nodes, dtype=torch.long)
        self.register_buffer('last_update', last_update)
        self.register_buffer('__assoc__',
                             torch.empty(num_nodes, dtype=torch.long))

        self.msg_s_store = {}
        self.msg_d_store = {}

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.msg_s_module, 'reset_parameters'):
            self.msg_s_module.reset_parameters()
        if hasattr(self.msg_d_module, 'reset_parameters'):
            self.msg_d_module.reset_parameters()
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

        # Detach all stored messages
        for k, v in self.msg_s_store.items():
            # new_raw_msg = []
            # for item in v:
            #     new_raw_msg.append(item.detach())
            #
            # self.msg_s_store[k] = tuple(new_raw_msg)
            self.msg_s_store[k] = (v[0], v[1], v[2], v[3], v[4].detach(), v[5].detach())

        for k, v in self.msg_d_store.items():
            # new_raw_msg = []
            # for item in v:
            #     new_raw_msg.append(item.detach())
            #
            # self.msg_d_store[k] = tuple(new_raw_msg)
            self.msg_d_store[k] = (v[0], v[1], v[2], v[3], v[4].detach(), v[5].detach())

    def forward(self, n_id: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns, for all nodes :obj:`n_id`, their current memory and their
        last updated timestamp."""
        if self.training:
            memory, last_update = self.__get_updated_memory__(n_id)
        else:
            memory, last_update = self.memory[n_id], self.last_update[n_id]

        return memory, last_update

    def update_state(self, src, dst, t, raw_msg, src_emb=None, dst_emb=None):
        """Updates the memory with newly encountered interactions
        :obj:`(src, dst, t, raw_msg, src_emb, dst_emb)`."""
        n_id = torch.cat([src, dst]).unique()

        if src_emb is None: src_emb = torch.empty(src.size(0), self.embedding_dim).to(src.device)
        if dst_emb is None: dst_emb = torch.empty(dst.size(0), self.embedding_dim).to(dst.device)

        if self.training:
            self.__update_memory__(n_id)
            self.__update_msg_store__(src, dst, t, raw_msg, src_emb, dst_emb, self.msg_s_store)
            self.__update_msg_store__(dst, src, t, raw_msg, dst_emb, src_emb, self.msg_d_store)
        else:
            self.__update_msg_store__(src, dst, t, raw_msg, src_emb, dst_emb, self.msg_s_store)
            self.__update_msg_store__(dst, src, t, raw_msg, dst_emb, src_emb, self.msg_d_store)
            self.__update_memory__(n_id)

    def __reset_message_store__(self):
        i = self.memory.new_empty((0,), dtype=torch.long)
        msg = self.memory.new_empty((0, self.raw_msg_dim))
        emb = self.memory.new_empty((0, self.embedding_dim))
        # Message store format: (src, dst, t, msg)
        self.msg_s_store = {j: (i, i, i, msg, emb, emb) for j in range(self.num_nodes)}
        self.msg_d_store = {j: (i, i, i, msg, emb, emb) for j in range(self.num_nodes)}

    def __update_memory__(self, n_id):
        memory, last_update = self.__get_updated_memory__(n_id)
        self.memory[n_id] = memory
        self.last_update[n_id] = last_update

    def __get_updated_memory__(self, n_id):
        self.__assoc__[n_id] = torch.arange(n_id.size(0), device=n_id.device)

        # Compute messages (src -> dst).
        msg_s, t_s, src_s, dst_s = self.__compute_msg__(
            n_id, self.msg_s_store, self.msg_s_module)

        # Compute messages (dst -> src).
        msg_d, t_d, src_d, dst_d = self.__compute_msg__(
            n_id, self.msg_d_store, self.msg_d_module)

        # Aggregate messages.
        idx = torch.cat([src_s, src_d], dim=0)
        msg = torch.cat([msg_s, msg_d], dim=0)
        t = torch.cat([t_s, t_d], dim=0)
        aggr = self.aggr_module(msg, self.__assoc__[idx], t, n_id.size(0))

        # Get local copy of updated memory.
        memory = self.gru(aggr, self.memory[n_id])

        # Get local copy of updated `last_update`.
        last_update = self.last_update.scatter(0, idx, t)[n_id]

        return memory, last_update

    def __update_msg_store__(self, src, dst, t, raw_msg, src_emb, dst_emb, msg_store):
        n_id, perm = src.sort()
        n_id, count = n_id.unique_consecutive(return_counts=True)
        for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
            msg_store[i] = (src[idx], dst[idx], t[idx], raw_msg[idx], src_emb[idx], dst_emb[idx])

    def __compute_msg__(self, n_id, msg_store, msg_module):
        data = [msg_store[i] for i in n_id.tolist()]
        src, dst, t, raw_msg, src_emb, dst_emb = list(zip(*data))
        src = torch.cat(src, dim=0)
        dst = torch.cat(dst, dim=0)
        t = torch.cat(t, dim=0)
        raw_msg = torch.cat(raw_msg, dim=0)
        src_emb = torch.cat(src_emb, dim=0)
        dst_emb = torch.cat(dst_emb, dim=0)
        t_rel = t - self.last_update[src]
        t_enc = self.time_enc(t_rel.to(raw_msg.dtype))

        msg = msg_module(self.memory[src], self.memory[dst], raw_msg, t_enc, src_emb, dst_emb)

        return msg, t, src, dst

    def train(self, mode: bool = True):
        """Sets the module in training mode."""
        if self.training and not mode:
            # Flush message store to memory in case we just entered eval mode.
            self.__update_memory__(
                torch.arange(self.num_nodes, device=self.memory.device))
            self.__reset_message_store__()
        super(TGNMemory, self).train(mode)

    def backup_memory(self):
        msg_s_clone = {}
        msg_d_clone = {}
        for k, v in self.msg_s_store.items():
            msg_s_clone[k] = [(x[0].clone(), x[1].clone(), x[2].clone(), x[3].clone()) for x in v]
        for k, v in self.msg_d_store.items():
            msg_d_clone[k] = [(x[0].clone(), x[1].clone(), x[2].clone(), x[3].clone()) for x in v]

        return self.memory.data.clone(), self.last_update.data.clone(), msg_s_clone, msg_d_clone

    def restore_memory(self, memory_backup):
        self.memory.data, self.last_update.data = memory_backup[0].clone(), memory_backup[1].clone()

        self.msg_s_store = {}
        self.msg_d_store = {}
        for k, v in self.memory_backup[2].items():
            self.msg_s_store[k] = [(x[0].clone(), x[1].clone(), x[2].clone(), x[3].clone()) for x in v]
        for k, v in self.memory_backup[3].items():
            self.msg_d_store[k] = [(x[0].clone(), x[1].clone(), x[2].clone(), x[3].clone()) for x in v]


class IdentityMessage(torch.nn.Module):
    def __init__(self, raw_msg_dim: int, time_dim: int, memory_dim: int = 100, embedding_dim: int = 100,
                 mode: str = "tgn"):
        super(IdentityMessage, self).__init__()
        self.mode = mode.lower()
        if self.mode == 'jodie' or self.mode == 'tgn' or self.mode == 'distance_encoding':
            self.out_channels = raw_msg_dim + 2 * memory_dim + time_dim
        elif self.mode == 'dyrep' or mode == 'src_emb':
            self.out_channels = raw_msg_dim + memory_dim + embedding_dim + time_dim
        elif self.mode == 'dual_emb':
            self.out_channels = raw_msg_dim + 2 * embedding_dim + time_dim
        else:
            raise NotImplementedError

    def forward(self, z_src, z_dst, raw_msg, t_enc, src_emb, dst_emb):

        if self.mode == 'jodie' or self.mode == 'tgn' or self.mode == 'distance_encoding':
            out = torch.cat([z_src, z_dst, raw_msg, t_enc], dim=-1)
        elif self.mode == 'dyrep':
            out = torch.cat([z_src, dst_emb, raw_msg, t_enc], dim=-1)
        elif self.mode == 'src_emb':
            out = torch.cat([src_emb, z_dst, raw_msg, t_enc], dim=-1)
        elif self.mode == 'dual_emb':
            out = torch.cat([src_emb, dst_emb, raw_msg, t_enc], dim=-1)
        else:
            raise NotImplementedError

        return out
