import time
import math
from tqdm import tqdm

import numpy as np
import torch
from torch.nn import Linear
from sklearn.metrics import average_precision_score, roc_auc_score, mean_absolute_error, mean_squared_error, \
    mean_squared_log_error
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models.tgn import LastAggregator
from memory import TGNMemory, IdentityMessage, TimeEncoder
from data import get_data, get_neighbor_finder, compute_time_statistics
from abstract_sum import AbstractSum, DyrepAbstractSum
from distance_encoding import distance_encoding
from arg_parser import get_args

# if MODE == "distance_encoding": USE_NEIGHBOR_FINDER=True rank_score="link_pred"
# TIME_ENCODING=False EDGE_ATTRIBUTE=False
MODE = "dyrep"
USE_EMBEDDING = True
USE_MEMORY_EMBEDDING = False
NAME = "wikipedia"
# RANK_SCORE = "likelihood"
RANK_SCORE = "intensity"
# RANK_SCORE = "link_pred"
BATCH_SIZE = 1
NUM_EPOCH = 500
LEARNING_RATE = 0.001
BACKPROP_EVERY_N_BATCHES = 20
NODE_FEATURE_DIM = 100
MEMORY_DIM = 100
TIME_DIM = 100
EMBEDDING_DIM = 100
TIME_ENCODING = True
DISTANCE_ENCODING = False
DE_SHORTEST_PATH = True
DE_RANDOM_WALK = True
EDGE_ATTRIBUTE = False
USE_MSG = False
N_LAYER = 1

NUM_NEG_SAMPLE = 10
N_COEFFICIENT = 2
DYREP_LOSS = True

USE_DYREPABSTRACTSUM = True
NUM_BASIS = 64
MC_NUM_SURV = 1
MC_NUM_TIME = 100

USE_NEIGHBOR_FINDER = True

USE_NEG = False
SAMPLE_NEG_DST_STRICTLY = True
CAN_SEE_NEIGHBORS_IN_SAME_BATCH = True
ONLY_PREDICT_LAST_EVENTS_TIME = True

USE_MEMORY = True

NEIGHBOR_SIZE = 10

DATASET_SIZE = 2000

CPU_ONLY = True

CANDIDATES_FROM = "all_dst"  # "all_dst", "all_nodes" or "neighbors"

# TASK = "link_prediction"
TASK = "time_prediction"


# TASK = "temporal_recommendation"

class TimeEmbedding(torch.nn.Module):
    def __init__(self, num_nodes, in_channels, out_channels, msg_dim, time_enc, n_layer):
        super(TimeEmbedding, self).__init__()

        self.in_channels = in_channels

        class NormalLinear(torch.nn.Linear):
            # From Jodie code
            def reset_parameters(self):
                stdv = 1. / math.sqrt(self.weight.size(1))
                self.weight.data.normal_(0, stdv)
                if self.bias is not None:
                    self.bias.data.normal_(0, stdv)

        self.embedding_layer = NormalLinear(1, self.in_channels)

    def forward(self, z, time_diffs=None):
        return z * (1 + self.embedding_layer(time_diffs.float().unsqueeze(1)))


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, num_nodes, in_channels, out_channels, msg_dim, time_enc, n_layer):
        super(GraphAttentionEmbedding, self).__init__()
        self.num_nodes = num_nodes
        self.time_enc = time_enc
        self.edge_dim = 0
        if TIME_ENCODING:
            self.edge_dim += time_enc.out_channels
        if EDGE_ATTRIBUTE:
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

        edge_attr = torch.zeros(edge_index.size(-1), 0).float().to(device)
        if TIME_ENCODING:
            # rel_t = self.last_update[edge_index[1]] - t             # lgh: edge_index[0] -> edge_index[1]
            rel_t_enc = self.time_enc(edge_time.to(x.dtype))
            edge_attr = torch.cat([edge_attr, rel_t_enc], dim=-1)
        if EDGE_ATTRIBUTE:
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


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super(LinkPredictor, self).__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h).sigmoid()


def tgat_link_predict(z_src, z_dst):
    return (z_src * z_dst).sum(dim=-1, keepdim=True).sigmoid()


def train(gnn, memory, abstract_sum, link_predictor, optimizer, criterion, train_data,
          full_t, full_msg, train_nodes, train_dst, statistics):
    # set the models to state of training
    if USE_MEMORY:
        memory.train()
        memory.reset_state()  # Start with a fresh memory.
    gnn.train()
    link_predictor.train()
    abstract_sum.train()

    neighbor_finder = get_neighbor_finder(train_data)

    total_loss = 0

    num_batch = math.ceil(train_data.num_events / BATCH_SIZE)

    loss = None
    optimizer.zero_grad()

    for i, batch in tqdm(enumerate(train_data.seq_batches(batch_size=BATCH_SIZE)), total=num_batch):

        src, dst, t, cur_e_id = batch.src, batch.dst, batch.t, batch.e_id
        valid_mask = cur_e_id != -1
        src, dst, t, cur_i_id = src[valid_mask], dst[valid_mask], t[valid_mask], cur_e_id[valid_mask]
        # msg = full_msg[cur_e_id]

        neg_dst = do_neg_sampling(src, dst, train_nodes, train_dst)

        dyrep_emb, last_update = None, None

        if MODE == "distance_encoding":
            z_pos_src, z_neg_src, z_pos_dst, z_neg_dst = get_distance_encoding(gnn, memory, neighbor_finder,
                                                                               src, dst, neg_dst, t, full_t, full_msg)
        else:
            dyrep_emb, last_update, z_pos_src, z_neg_src, z_pos_dst, z_neg_dst = get_temporal_embedding(gnn, memory,
                                                                                                        neighbor_finder,
                                                                                                        src, dst,
                                                                                                        neg_dst,
                                                                                                        t,
                                                                                                        full_t,
                                                                                                        full_msg,
                                                                                                        statistics)

        loss_cur_batch = compute_loss(abstract_sum, link_predictor, criterion,
                         src, last_update, t, z_neg_dst, z_neg_src, z_pos_dst, z_pos_src)

        if loss is None:
            loss = loss_cur_batch
        else:
            loss += loss_cur_batch
        if torch.isnan(loss).any() or torch.isinf(loss).any():  # print(f"{i}, loss: {loss}")
            print(f"loss: {loss}")

        total_loss += float(loss_cur_batch) * batch.num_events

        if USE_MEMORY:
            update_memory(memory, src, dst, t, cur_e_id, full_msg, dyrep_emb)

        if (i + 1) % BACKPROP_EVERY_N_BATCHES == 0 or i + 1 == num_batch:
            loss.backward()  # torch.nn.utils.clip_grad_norm_(all_parameters, max_norm=100, norm_type=2)
            optimizer.step()
            loss = 0
            optimizer.zero_grad()
            if USE_MEMORY:
                memory.detach()

    return total_loss / train_data.num_events


def update_memory(memory, src, dst, t, cur_e_id, full_msg, dyrep_emb):
    # Update memory and neighbor loader with ground-truth state.
    update_valid_mask = cur_e_id != -1
    valid_src, valid_dst, valid_t, valid_cur_e_id = \
        src[update_valid_mask], dst[update_valid_mask], t[update_valid_mask], cur_e_id[update_valid_mask]
    msg = full_msg[valid_cur_e_id]
    if MODE == "dyrep":
        assert dyrep_emb is not None, "dyrep_emb is None"
        # saved_edge_index, saved_e_id, saved_t = edge_index, e_id, t
        memory.update_state(valid_src, valid_dst, valid_t, msg, src_emb=dyrep_emb[:valid_src.size(0)],
                            dst_emb=dyrep_emb[valid_src.size(0):2 * valid_src.size(0)])
    else:
        memory.update_state(valid_src, valid_dst, valid_t, msg)
    seen_nodes.extend(torch.cat([valid_src, valid_dst]).unique().tolist())


def compute_loss(abstract_sum, link_predictor, criterion,
                 src, last_update, t, z_neg_dst, z_neg_src, z_pos_dst, z_pos_src):
    num_src = src.size(0)  # number of pos src
    num_pos = 2 * num_src  # total number of pos src and pos dst
    #  1. the last index of neg dst or 2. the split index for neg src and neg dst
    neg_split = (2 + NUM_NEG_SAMPLE) * num_src
    if DYREP_LOSS:
        last_update_src, last_update_dst = last_update[:num_src], last_update[num_src:num_pos]
        pos_delta_t = (t - torch.max(last_update_src, last_update_dst)).float()

        # neg_src is in fact equal to pos_src
        last_update_neg_src, last_update_neg_dst = \
            last_update_src.repeat_interleave(NUM_NEG_SAMPLE, dim=0), last_update[num_pos:neg_split]

        delta_t_neg = (t.repeat_interleave(NUM_NEG_SAMPLE) - torch.max(last_update_neg_src,
                                                                       last_update_neg_dst)).float()
        loss_cur_batch = - torch.log(abstract_sum.intensity_func(pos_delta_t, z_pos_src, z_pos_dst,
                                                                 return_params=False)).sum() / z_pos_src.size(0)
        loss_cur_batch += N_COEFFICIENT * abstract_sum.intensity_func(delta_t_neg, z_neg_src, z_neg_dst,
                                                                      return_params=False).sum() / z_neg_src.size(0)

    else:
        pos_out = link_predictor(z_pos_src, z_pos_dst)
        neg_out = link_predictor(z_neg_src, z_neg_dst)
        loss_cur_batch = criterion(pos_out, torch.ones_like(pos_out))
        loss_cur_batch += N_COEFFICIENT * criterion(neg_out, torch.zeros_like(neg_out))
    return loss_cur_batch


def get_temporal_embedding(gnn, memory, neighbor_finder, src, dst, neg_dst, t, full_t, full_msg, statistics):
    num_src = src.size(0)  # number of pos src
    num_pos = 2 * num_src  # total number of pos src and pos dst
    #  1. the last index of neg dst or 2. the split index for neg src and neg dst
    neg_split = (2 + NUM_NEG_SAMPLE) * num_src

    if DYREP_LOSS:
        # n_id = torch.cat([src, dst, neg_src, neg_dst])
        n_id = torch.cat([src, dst, neg_dst])
    else:
        n_id = torch.cat([src, dst, neg_dst])
    num_focused_node = n_id.size(0)
    ratio = int(num_focused_node / num_src)
    # timestamps = t.repeat(3).cpu().numpy()
    if CAN_SEE_NEIGHBORS_IN_SAME_BATCH:
        timestamps = t.repeat(int(num_focused_node / src.size(0))).cpu().numpy()
    else:
        timestamps = np.array([t[0].item()]).repeat(num_focused_node)
    n_id, edge_index, e_id, root_id_in_batch = neighbor_finder.get_k_hop_temporal_neighbor(n_id.cpu().numpy(),
                                                                                           timestamps,
                                                                                           n_neighbors=NEIGHBOR_SIZE,
                                                                                           k=1)
    n_id, edge_index, e_id, root_id_in_batch = n_id.to(device), edge_index.to(device), e_id.to(
        device), root_id_in_batch.to(device)
    n_id_unique = n_id.unique()
    n_id_old2new = get_map_tensor(n_id_unique)

    if USE_MEMORY:
        z, last_update = memory(n_id_unique)
        z, last_update = z[n_id_old2new[n_id]], last_update[n_id_old2new[n_id]]
    else:
        z, last_update = torch.zeros(size=(len(n_id), MEMORY_DIM), device=device), \
                         torch.zeros(size=(len(n_id),), device=device)

    emb = None

    if MODE == "jodie":

        time_diff = t.repeat(ratio) - last_update[:num_focused_node]
        if ratio == 2 + NUM_NEG_SAMPLE:
            # negative samples are sampled for dst only
            # for pos src
            time_diff[:num_src] = (time_diff[:num_src] - statistics.mean_time_shift_src) / statistics.std_time_shift_src
            # for pos dst and neg dst
            time_diff[num_src:] = (time_diff[num_src:] - statistics.mean_time_shift_dst) / statistics.std_time_shift_dst
        elif ratio == 2 + 2 * NUM_NEG_SAMPLE:
            # negative samples are sampled for both src and dst
            time_diff[:num_src] = (time_diff[:num_src] - statistics.mean_time_shift_src) / statistics.std_time_shift_src
            time_diff[num_src:num_pos] =\
                (time_diff[num_src:num_pos] - statistics.mean_time_shift_dst) / statistics.std_time_shift_dst
            # for neg src
            time_diff[num_pos:neg_split] = \
                (time_diff[num_pos:neg_split] - statistics.mean_time_shift_src) / statistics.std_time_shift_src
            # for neg dst
            time_diff[neg_split:] = \
                (time_diff[neg_split:] - statistics.mean_time_shift_dst) / statistics.std_time_shift_dst
        else:
            raise NotImplementedError
        z = z[:num_focused_node]
        z = gnn(z, time_diff)

    elif MODE == "tgat" or (MODE == "tgn" and USE_EMBEDDING):
        z = gnn(z, edge_index, t.repeat(ratio)[root_id_in_batch] - full_t[e_id], full_msg[e_id])

    elif MODE == "dyrep":
        emb = gnn(z, edge_index, t.repeat(ratio)[root_id_in_batch] - full_t[e_id], full_msg[e_id])

    z_pos_src, z_pos_dst = z[:num_src], z[num_src:num_pos]
    # neg_src is in fact equal to pos_src
    z_neg_src, z_neg_dst = z[:num_src].repeat_interleave(NUM_NEG_SAMPLE, dim=0), z[num_pos:neg_split]

    return emb, last_update, z_pos_src, z_neg_src, z_pos_dst, z_neg_dst


# Return a helper vector to map global node indices to new local ones.
def get_map_tensor(n_id_unique):
    n_id_unique = n_id_unique.unique()
    max_n_id_unique = n_id_unique.max().item()
    n_id_old2new = torch.empty(max_n_id_unique, dtype=torch.long, device=device)
    n_id_old2new[n_id_unique] = torch.arange(n_id_unique.size(0), device=device)
    return n_id_old2new


def get_distance_encoding(gnn, memory, neighbor_finder, src, dst, neg_dst, t, full_t, full_msg):
    z_src, z_dst = distance_encoding(src.repeat(NUM_NEG_SAMPLE + 1), torch.cat([dst, neg_dst]),
                                     t.repeat(NUM_NEG_SAMPLE + 1), neighbor_finder,
                                     n_neighbors=NEIGHBOR_SIZE,
                                     USE_MEMORY_EMBEDDING=USE_MEMORY_EMBEDDING,
                                     DE_SHORTEST_PATH=DE_SHORTEST_PATH,
                                     DE_RANDOM_WALK=DE_RANDOM_WALK,
                                     NODE_FEATURE_DIM=MEMORY_DIM, full_t=full_t, full_msg=full_msg,
                                     memory=memory,
                                     gnn=gnn, k=2)
    z_pos_src, z_pos_dst = z_src[:src.size(0)], z_dst[:src.size(0)]
    z_neg_src, z_neg_dst = z_src[src.size(0):], z_dst[src.size(0):]
    return z_pos_src, z_neg_src, z_pos_dst, z_neg_dst


def do_neg_sampling(src, dst, nodes, dst_nodes):
    # Sample negative destination nodes.
    if SAMPLE_NEG_DST_STRICTLY:
        sample_mask = torch.ones(size=(src.size(0) * NUM_NEG_SAMPLE,), device=device).bool()
        neg_dst = torch.empty(size=(src.size(0) * NUM_NEG_SAMPLE,), device=device, dtype=src.dtype)
        while sample_mask.any():
            neg_dst_index = torch.randint(0, dst_nodes.size(0), (sample_mask.sum(),),
                                          dtype=torch.long, device=device)
            neg_dst[sample_mask] = dst_nodes[neg_dst_index]
            sample_mask = torch.logical_or(neg_dst == dst.repeat_interleave(NUM_NEG_SAMPLE),
                                           neg_dst == src.repeat_interleave(NUM_NEG_SAMPLE))

    else:
        neg_dst_index = torch.randint(0, nodes.size(0), (src.size(0),),
                                      dtype=torch.long, device=device)
        neg_dst = nodes[neg_dst_index]
    return neg_dst


@torch.no_grad()
def test(memory, gnn, link_predictor, inference_data, full_pos_data, full_t, full_msg, full_nodes, full_dst, statistics,
         old_mask, nn_mask, nn_one_mask, nn_two_mask, temporal_neg_mask=None):
    neighbor_finder = get_neighbor_finder(full_pos_data)

    if temporal_neg_mask is None:
        temporal_neg_mask = np.zeros(old_mask.shape).astype(bool)

    if USE_MEMORY:
        memory.eval()

    gnn.eval()
    link_predictor.eval()

    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.

    # aps, aucs = [], []
    all_pos_out, all_neg_out = [], []
    for batch in inference_data.seq_batches(batch_size=BATCH_SIZE):
        src, dst, t, cur_e_id = batch.src, batch.dst, batch.t, batch.e_id

        neg_dst = do_neg_sampling(src, dst, full_nodes, full_dst)

        dyrep_emb = None
        if MODE == "distance_encoding":
            z_pos_src, z_neg_src, z_pos_dst, z_neg_dst = get_distance_encoding(gnn, memory, neighbor_finder,
                                                                               src, dst, neg_dst, t, full_t, full_msg)
        else:
            dyrep_emb, last_update, z_pos_src, z_neg_src, z_pos_dst, z_neg_dst = get_temporal_embedding(gnn, memory,
                                                                                                        neighbor_finder,
                                                                                                        src, dst,
                                                                                                        neg_dst,
                                                                                                        t,
                                                                                                        full_t,
                                                                                                        full_msg,
                                                                                                        statistics)

        pos_out = link_predictor(z_pos_src, z_pos_dst)
        neg_out = link_predictor(z_neg_src, z_neg_dst)

        all_pos_out.extend(pos_out)
        all_neg_out.extend(neg_out)

        if USE_MEMORY:
            update_memory(memory, src, dst, t, cur_e_id, full_msg, dyrep_emb)

    all_pos_out = torch.tensor(all_pos_out)
    all_neg_out = torch.tensor(all_neg_out)
    all_labels = torch.cat([torch.from_numpy(~temporal_neg_mask).float(), torch.zeros_like(all_neg_out)], dim=0).cpu()
    all_predictions = torch.cat([all_pos_out, all_neg_out], dim=0).cpu()
    valid_mask = ~temporal_neg_mask

    valid_mask, old_mask, nn_mask, nn_one_mask, nn_two_mask = np.tile(valid_mask, 2), np.tile(old_mask, 2), np.tile(
        nn_mask, 2), np.tile(nn_one_mask, 2), np.tile(nn_two_mask, 2)

    ap, auc, pos_acc, neg_acc = compute_scores(all_labels, all_predictions, valid_mask)
    old_ap, old_auc, old_pos_acc, old_neg_acc = compute_scores(all_labels, all_predictions, old_mask)
    nn_ap, nn_auc, nn_pos_acc, nn_neg_acc = compute_scores(all_labels, all_predictions, nn_mask)
    nn_one_ap, nn_one_auc, nn_one_pos_acc, nn_one_neg_acc = compute_scores(all_labels, all_predictions, nn_one_mask)
    nn_two_ap, nn_two_auc, nn_two_pos_acc, nn_two_neg_acc = compute_scores(all_labels, all_predictions, nn_two_mask)

    temporal_neg_acc = (all_predictions[np.concatenate((temporal_neg_mask, np.zeros(len(temporal_neg_mask)).astype(bool)
                                                        ))] < 0.5).float().mean()

    return ap, auc, pos_acc, neg_acc, \
           old_ap, old_auc, old_pos_acc, old_neg_acc, \
           nn_ap, nn_auc, nn_pos_acc, nn_neg_acc, \
           nn_one_ap, nn_one_auc, nn_one_pos_acc, nn_one_neg_acc, \
           nn_two_ap, nn_two_auc, nn_two_pos_acc, nn_two_neg_acc, \
           temporal_neg_acc


def compute_scores(labels, predictions, mask):
    ap, auc, pos_acc, neg_acc = average_precision_score(labels[mask], predictions[mask]), \
                                roc_auc_score(labels[mask], predictions[mask]), \
                                (predictions[mask][labels[mask].bool()] >= 0.5).float().mean(), \
                                (predictions[mask][~labels[mask].bool()] < 0.5).float().mean()
    return ap, auc, pos_acc, neg_acc


@torch.no_grad()
def test_time_prediction(gnn, memory, abstract_sum, link_predictor,
                         inference_data, full_pos_data, full_t,  full_msg, statistics):
    if not USE_MEMORY:
        raise NotImplementedError("Time prediction depends on node memory")

    memory.eval()
    gnn.eval()
    if RANK_SCORE == "link_pred":
        link_predictor.eval()
    else:
        abstract_sum.eval()

    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.

    neighbor_finder = get_neighbor_finder(full_pos_data)

    all_t_true, all_t_estimated = np.array([]), np.array([])

    for batch in inference_data.seq_batches(batch_size=1):
        src, dst, t, e_id = get_valid_data(batch)

        if src.size(0) == 0:
            print("No pos_sample in this batch, continue")
            continue

        if ONLY_PREDICT_LAST_EVENTS_TIME:
            src, dst, t, e_id = retrieve_last_events(src, dst, t, e_id)

        dyrep_emb = None
        if MODE == "distance_encoding":

            z_src, z_dst = distance_encoding(src, dst, t, neighbor_finder,
                                             n_neighbors=NEIGHBOR_SIZE, USE_MEMORY_EMBEDDING=USE_MEMORY_EMBEDDING,
                                             DE_SHORTEST_PATH=DE_SHORTEST_PATH, DE_RANDOM_WALK=DE_RANDOM_WALK,
                                             NODE_FEATURE_DIM=MEMORY_DIM, full_t=full_t, full_msg=full_msg,
                                             memory=memory,
                                             gnn=gnn, k=2)
            _, last_update_src = memory(src)
            _, last_update_dst = memory(dst)

            # lgh: It's safe to set `assoc` to None here,
            # because the outside will use `assoc` only if `args.mode` == "dyrep"
            assoc = None

        else:
            n_id, edge_index, edge_id = get_k_hop_temp_neighbors(neighbor_finder, src, dst, t)

            assoc = get_map_tensor(n_id)

            z, last_update = memory(n_id)

            dyrep_emb, z_src, z_dst = get_src_and_dst_embeddings(gnn, edge_index, src, dst, t, full_msg,
                                                                 edge_id, z, last_update, statistics, assoc)

            last_update_src, last_update_dst = last_update[assoc[src]], last_update[assoc[dst]]

        t_true = t - torch.max(torch.cat((last_update_src, last_update_dst), dim=-1), dim=-1)[0]

        seen_mask = t_true != t
        seen_z_src, seen_z_pos_dst, seen_t_true, seen_src, seen_pos_dst, seen_e_id \
            = z_src[seen_mask], z_dst[seen_mask], t_true[seen_mask], \
              src[seen_mask], dst[seen_mask], e_id[seen_mask]

        if seen_z_src.size(0) == 0:
            print(f"e_id: {e_id}, src:{src}, dst:{dst}")
            print("all nodes unseen in this batch, continue to the next batch")
            pass
        else:
            t_estimated = np.clip(
                abstract_sum.estimate_time(z_src[seen_mask], z_dst[seen_mask]).cpu().numpy() * 5000,
                -np.inf,
                100000)

            seen_t_true = seen_t_true.cpu().numpy()
            print(f"e_id: {seen_e_id}, src:{seen_src}, dst:{seen_pos_dst}, "
                  f"t_estimated: {t_estimated}, t_true: {t_true}")
            absolute_error = len(seen_t_true) * mean_absolute_error(seen_t_true, t_estimated)
            print(f"absolute_error: {absolute_error}")

            all_t_true, all_t_estimated = np.concatenate([all_t_true, seen_t_true]), \
                                          np.concatenate([all_t_estimated, t_estimated])

        msg = full_msg[e_id]
        if MODE == "dyrep":
            memory.update_state(src, dst, t, msg, src_emb=dyrep_emb[assoc[src]], dst_emb=dyrep_emb[assoc[dst]])
        else:
            memory.update_state(src, dst, t, msg)

    # all_t_estimated = np.clip(5000 * all_t_estimated, -np.inf, 100000)
    mae = float(mean_absolute_error(all_t_true, all_t_estimated))
    rmse = float(np.sqrt(mean_squared_error(all_t_true, all_t_estimated)))
    rmsle = float(np.sqrt(mean_squared_log_error(all_t_true, all_t_estimated)))

    return mae, rmse, rmsle


def get_src_and_dst_embeddings(gnn, edge_index, src, dst, t, full_msg, e_id, z, last_update, statistics, assoc):
    dyrep_emb = None

    if MODE == "jodie":

        time_diff_src = t - last_update[assoc[src]]
        time_diff_src = (time_diff_src - statistics.mean_time_shift_src) / statistics.std_time_shift_src
        time_diff_pos_dst = t - last_update[assoc[dst]]
        time_diff_pos_dst = (time_diff_pos_dst - statistics.mean_time_shift_dst) / statistics.std_time_shift_dst

        z_src = z[assoc[src]]
        z_dst = z[assoc[dst]]

        z_src = gnn(z_src, time_diff_src)
        z_dst = gnn(z_dst, time_diff_pos_dst)

    else:

        if MODE == "tgat" or (MODE == "tgn" and USE_EMBEDDING):
            z = gnn(z, edge_index, torch.tensor([0]).float().repeat(edge_index.size(1)), full_msg[e_id])
            print(f"z.size(): {z.size()}")
            print(f"assoc[candidates]: {assoc[dst]}")
        elif MODE == "dyrep":
            dyrep_emb = gnn(z, edge_index, torch.tensor([0]).float().repeat(edge_index.size(1)), full_msg[e_id])
        elif MODE == "tgn" and not USE_EMBEDDING:
            pass
        else:
            raise NotImplementedError

        z_src = z[assoc[src]]
        z_dst = z[assoc[dst]]

    return dyrep_emb, z_src, z_dst


def get_k_hop_temp_neighbors(neighbor_finder, src, dst, t):
    n_id = torch.cat([src, dst]).unique()
    edge_index, e_id = None, None
    if not (not USE_EMBEDDING and MODE == "tgn"):
        n_id, edge_index, e_id = \
            neighbor_finder.get_k_hop_temporal_neighbor_by_one_timestamp(n_id, t[0], n_neighbors=NEIGHBOR_SIZE, k=1)
        n_id, edge_index, e_id = torch.from_numpy(n_id).to(device), \
                                 torch.from_numpy(edge_index).to(device), \
                                 torch.from_numpy(e_id).to(device).long()
    return n_id, edge_index, e_id


def get_valid_data(batch):
    src, dst, t, e_id = batch.src, batch.dst, batch.t, batch.e_id
    valid_mask = e_id != -1
    src, dst, t, e_id = src[valid_mask], dst[valid_mask], t[valid_mask], e_id[valid_mask]
    return src, dst, t, e_id


def retrieve_last_events(src, dst, t, cur_e_id):
    counted_nodes = set()
    mask = src.new_zeros(src.shape).bool()
    for i in range(1, src.size(0) + 1):
        mask[-i] = src[-i].item() not in counted_nodes and dst[-i].item() not in counted_nodes
        counted_nodes = counted_nodes | {src[-i].item(), dst[-i].item()}
    src, dst, t, cur_e_id = src[mask], dst[mask], t[mask], cur_e_id[mask]
    return src, dst, t, cur_e_id


@torch.no_grad()
def test_temporal_recommendation(gnn, memory, abstract_sum, link_predictor, inference_data,
                                 full_pos_data, full_t, full_msg, full_nodes, full_dst, statistics,
                                 old_mask, nn_mask, nn_one_mask, nn_two_mask, temporal_neg_mask=None):
    if temporal_neg_mask is None:
        temporal_neg_mask = np.zeros(old_mask.shape).astype(bool)

    valid_mask = ~temporal_neg_mask
    old_mask, nn_mask, nn_one_mask, nn_two_mask = old_mask[valid_mask], \
                                                  nn_mask[valid_mask], \
                                                  nn_one_mask[valid_mask], \
                                                  nn_two_mask[valid_mask]

    if USE_MEMORY:
        memory.eval()
    gnn.eval()
    if RANK_SCORE == "link_pred":
        link_predictor.eval()
    else:
        abstract_sum.eval()

    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.

    neighbor_finder = get_neighbor_finder(full_pos_data)

    rank_list = []
    miss_record = []
    for batch in inference_data.seq_batches(batch_size=1):
        src, dst, t, e_id = get_valid_data(batch)

        if src.size(0) == 0:
            print("No pos_sample in this batch, continue")
            continue

        candidates, root_nodes, root_ids, recommend_times =\
            get_candidates(neighbor_finder, src, t, full_nodes, full_dst)

        dyrep_emb, last_update = None, None
        if MODE == "distance_encoding":  # In this case, it is certain that USE_MEMORY == False

            n_id = root_nodes
            assoc = get_map_tensor(n_id)
            last_update = torch.zeros(len(n_id), device=device)

            z_root, z_candidate = distance_encoding(root_nodes, candidates, recommend_times, neighbor_finder,
                                                    n_neighbors=NEIGHBOR_SIZE,
                                                    USE_MEMORY_EMBEDDING=USE_MEMORY_EMBEDDING,
                                                    DE_SHORTEST_PATH=DE_SHORTEST_PATH, DE_RANDOM_WALK=DE_RANDOM_WALK,
                                                    NODE_FEATURE_DIM=MEMORY_DIM, full_t=full_t, full_msg=full_msg,
                                                    memory=memory,
                                                    gnn=gnn, k=2)

        else:
            n_id, edge_index, edge_id = get_k_hop_temp_neighbors(neighbor_finder, src, dst, t)
            assoc = get_map_tensor(n_id)

            if n_id.size(0) == 0:
                print("n_id.size(0) == 0 in this batch, continue")
                continue

            if USE_MEMORY:
                z, last_update = memory(n_id)
            else:
                z, last_update = torch.zeros(len(n_id), MEMORY_DIM, device=device), \
                                 torch.zeros(len(n_id), device=device)

            dyrep_emb, z_root, z_candidate = \
                get_src_and_dst_embeddings(gnn, edge_index, src, dst, t, full_msg, edge_id, z, last_update,
                                           statistics, assoc)

        last_update_root, last_update_candidate = last_update[assoc[root_nodes]], last_update[assoc[candidates]]
        last_time_max = torch.max(torch.cat((last_update_root, last_update_candidate), dim=-1), dim=-1)[0]
        delta_t = recommend_times - last_time_max

        rank_scores = get_rank_scores(abstract_sum, link_predictor, z_root, z_candidate, delta_t)

        # _, counts = torch.unique_consecutive(recommend_times, return_counts=True)
        # root_nodes_tuple = torch.split(root_nodes, counts.tolist())
        root_ids_unique_consecutive, counts = torch.unique_consecutive(root_ids, return_counts=True)
        scores_tuple = torch.split(rank_scores, counts.tolist())
        candidates_tuple = torch.split(candidates, counts.tolist())

        for root_id, scores, candidates in zip(root_ids_unique_consecutive, scores_tuple, candidates_tuple):
            cur_src, cur_dst = src[root_id], dst[root_id]
            print(f"e_id {e_id[root_id]} src {cur_src} dst {cur_dst}")
            if cur_src.item() not in seen_nodes:
                print(f"src {cur_src.item()} is new !")
            if cur_dst.item() not in seen_nodes:
                print(f"dst {cur_dst.item()} is new !")
            sorted_args = torch.argsort(scores, descending=True)
            sorted_candidates = candidates[sorted_args]
            if cur_dst in sorted_candidates:
                rank = torch.nonzero(sorted_candidates == cur_dst).squeeze() + 1
                miss_record.append(False)
                # print(f"rank:{rank}, dst:{dst}, sorted_candidates:{sorted_candidates}")
                # print(rank.shape)
            else:
                rank = len(sorted_candidates) + 1
                miss_record.append(True)
                print("not hit!")
            rank_list.append(int(rank))
            print(f"rank:{rank}, dst:{cur_dst}, num_candidates:{sorted_candidates.size(0)}")
            # print(f"sorted_candidates:{sorted_candidates}")
            # print(f"sorted_scores: {scores[sorted_args]}")
            # print(f"rank:{rank}, dst:{dst}, num_candidates:{sorted_candidates.size(0)}, "
            #       f"highest_scored_candidate: {sorted_candidates[0]}")
            # print(rank.shape)

        if USE_MEMORY:
            msg = full_msg[e_id]
            if MODE == "dyrep":
                # saved_edge_index, saved_e_id, saved_t = edge_index, e_id, t
                memory.update_state(src, dst, t, msg, src_emb=dyrep_emb[assoc[src]], dst_emb=dyrep_emb[assoc[dst]])
            else:
                memory.update_state(src, dst, t, msg)

        seen_nodes.extend(torch.cat([src, dst]).unique().tolist())

    rank_list_torch = torch.Tensor(rank_list).float()
    miss_record = torch.Tensor(miss_record, device=device).bool()

    mar = rank_list_torch.mean()

    old_mar, old_miss_record = rank_list_torch[old_mask].mean(), miss_record[old_mask]
    nn_mar, nn_miss_record = rank_list_torch[nn_mask].mean(), miss_record[nn_mask]
    nn_one_mar, nn_one_miss_record = rank_list_torch[nn_one_mask].mean(), miss_record[nn_one_mask]
    nn_two_mar, nn_two_miss_record = rank_list_torch[nn_two_mask].mean(), miss_record[nn_two_mask]

    return mar, miss_record, \
        old_mar, old_miss_record, \
        nn_mar, nn_miss_record, \
        nn_one_mar, nn_one_miss_record, \
        nn_two_mar, nn_two_miss_record


def get_rank_scores(abstract_sum, link_predictor, z_root, z_candidate, delta_t):
    if RANK_SCORE == "likelihood":
        rank_scores = abstract_sum.likelihood_func(delta_t, z_src=z_root, z_dst=z_candidate)
    elif RANK_SCORE == "intensity":
        rank_scores = abstract_sum(z_root, z_candidate, delta_t)
    elif RANK_SCORE == "link_pred":
        rank_scores = link_predictor(z_root, z_candidate).squeeze()
    else:
        raise NotImplementedError
    return rank_scores


def get_candidates(neighbor_finder, src, t, full_nodes, full_dst):
    def get_candidates_from(nodes_th):
        len_src, len_nodes = src.size(0), nodes_th.size(0)
        candidates_inner = nodes_th.repeat(len_src)
        root_nodes_inner = src.repeat_interleave(len_nodes)
        root_ids_inner = torch.arange(len_src).repeat_interleave(len_nodes)
        recommend_times_inner = t.repeat_interleave(len_nodes)
        valid_mask = root_nodes_inner != candidates_inner
        candidates_inner, root_nodes_inner, root_ids_inner, recommend_times_inner = candidates_inner[valid_mask], \
                                                                                    root_nodes_inner[valid_mask], \
                                                                                    root_ids_inner[valid_mask], \
                                                                                    recommend_times_inner[valid_mask]
        return candidates_inner, root_nodes_inner, root_ids_inner, recommend_times_inner

    if CANDIDATES_FROM == "neighbors":
        candidates, root_nodes, root_ids, recommend_times = \
            neighbor_finder.get_temporal_recommendation_candidates(src.cpu().numpy(), t.cpu().numpy())

        candidates, root_nodes, root_ids, recommend_times = torch.from_numpy(candidates).to(device), \
                                                            torch.from_numpy(root_nodes).to(device), \
                                                            torch.from_numpy(root_ids).to(device), \
                                                            torch.from_numpy(recommend_times).to(device)
    elif CANDIDATES_FROM == "all_dst":
        candidates, root_nodes, root_ids, recommend_times = get_candidates_from(full_dst)

    elif CANDIDATES_FROM == "all_nodes":
        candidates, root_nodes, root_ids, recommend_times = get_candidates_from(full_nodes)
    else:
        raise NotImplementedError("CANDIDATES_FROM must be \"neighbors\", \"all_dst\" or \"all_nodes\".")

    return candidates, root_nodes, root_ids, recommend_times


def main(args):
    print("preparing data ...")
    full_pos_data, train_data, val_data, test_data, full_msg, full_t, \
    val_old_mask, test_old_mask, nn_val_mask, nn_test_mask, nn_one_val_mask, \
    nn_one_test_mask, nn_two_val_mask, nn_two_test_mask, neg_test_mask, \
    n_total_unique_nodes_real, num_train_events, min_dst_idx, max_dst_idx \
        = get_data(NAME, use_neg=USE_NEG, use_msg=USE_MSG, dataset_size=DATASET_SIZE, cpu_only=CPU_ONLY, nn_ratio=0)
    train_src = torch.unique(train_data.src, sorted=True)
    train_dst = torch.unique(train_data.dst, sorted=True)
    train_nodes = torch.unique(torch.cat([train_src, train_dst]), sorted=True)
    full_src = torch.unique(full_pos_data.src, sorted=True)
    full_dst = torch.unique(full_pos_data.dst, sorted=True)
    full_nodes = torch.unique(torch.cat([full_src, full_dst]), sorted=True)
    # mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst, mean_time_shift, std_time_shift
    statistics = compute_time_statistics(full_pos_data.src, full_pos_data.dst, full_pos_data.t)
    print("data prepared")

    if USE_MEMORY:
        memory = TGNMemory(
            n_total_unique_nodes_real,
            full_msg.size(-1),
            MEMORY_DIM,
            EMBEDDING_DIM,
            TIME_DIM,
            message_module=IdentityMessage(full_msg.size(-1), TIME_DIM, MEMORY_DIM, EMBEDDING_DIM, mode=MODE),
            aggregator_module=LastAggregator(),
            mode=MODE
        ).to(device)
    else:
        memory = None

    arg_dict = {"num_nodes": n_total_unique_nodes_real,
                "in_channels": MEMORY_DIM if MODE != "distance_encoding" else 5 * DE_SHORTEST_PATH + 4 * DE_RANDOM_WALK,
                "out_channels": EMBEDDING_DIM,
                "msg_dim": full_msg.size(-1),
                "time_enc": memory.time_enc if memory is not None else TimeEncoder(out_channels=TIME_DIM, mode=MODE).to(
                    device),
                "n_layer": N_LAYER}
    if MODE == "jodie":
        gnn = TimeEmbedding(**arg_dict).to(device)
    elif MODE == "distance_encoding":
        gnn = GraphAttentionEmbedding(**arg_dict).to(device)
    else:
        gnn = GraphAttentionEmbedding(**arg_dict).to(device)

    link_predictor = LinkPredictor(in_channels=EMBEDDING_DIM).to(device)

    if not USE_DYREPABSTRACTSUM:
        # in_channels, num_basis, num_param, mc_num_surv = 1, mc_num_time = 3
        abstract_sum = AbstractSum(2 * EMBEDDING_DIM, num_basis=NUM_BASIS, mc_num_surv=MC_NUM_SURV,
                                   mc_num_time=MC_NUM_TIME)
    else:
        # in_channels, num_basis, num_param, mc_num_surv = 1, mc_num_time = 3
        abstract_sum = DyrepAbstractSum(2 * EMBEDDING_DIM, mc_num_surv=MC_NUM_SURV, mc_num_time=MC_NUM_TIME)

    if USE_MEMORY:
        memory_parameter_set = set(memory.parameters())
    else:
        memory_parameter_set = set()

    all_parameters = memory_parameter_set | set(gnn.parameters()) | set(abstract_sum.parameters()) \
                     | set(link_predictor.parameters())

    optimizer = torch.optim.Adam(all_parameters, lr=LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(1, args.num_epoch + 1):
        start_epoch = time.time()

        print(f"start epoch {epoch} ...")

        train_loss = train(gnn, memory, abstract_sum, link_predictor, optimizer, criterion, train_data,
                           full_t, full_msg, train_nodes, train_dst, statistics)
        epoch_train_time = time.time() - start_epoch

        print(f"  Epoch: {epoch:02d}, Loss: {train_loss:.4f}")
        print(f"  Train_time: {epoch_train_time:.2f}s")

        if TASK == "time_prediction":

            val_mae, val_rmse, val_rmsle = test_time_prediction(gnn, memory, abstract_sum, link_predictor,
                                                                val_data, full_pos_data, full_t,  full_msg,
                                                                statistics)
            test_mae, test_rmse, test_rmsle = test_time_prediction(gnn, memory, abstract_sum, link_predictor,
                                                                   test_data, full_pos_data, full_t,  full_msg,
                                                                   statistics)

            print(f"Val_MAE: {val_mae:.4f}, Val_RMSE: {val_rmse:.4f}, Val_RMSLE: {val_rmsle:.4f}")
            print(f"Test_MAE: {test_mae:.4f}, Test_RMSE: {test_rmse:.4f}, Test_RMSLE: {test_rmsle:.4f}")

        elif TASK == "temporal_recommendation":

            val_mar, val_miss_record, \
                val_old_mar, val_old_miss_record, \
                val_nn_mar, val_nn_miss_record, \
                val_nn_one_mar, val_nn_one_miss_record, \
                val_nn_two_mar, val_nn_two_miss_record = test_temporal_recommendation(gnn, memory,
                                                                                      abstract_sum, link_predictor,
                                                                                      val_data,
                                                                                      full_pos_data, full_t, full_msg,
                                                                                      full_nodes, full_dst,
                                                                                      statistics,
                                                                                      val_old_mask,
                                                                                      nn_val_mask,
                                                                                      nn_one_val_mask,
                                                                                      nn_two_val_mask)

            test_mar, test_miss_record, \
                test_old_mar, test_old_miss_record, \
                test_nn_mar, test_nn_miss_record, \
                test_nn_one_mar, test_nn_one_miss_record, \
                test_nn_two_mar, test_nn_two_miss_record = test_temporal_recommendation(gnn, memory,
                                                                                        abstract_sum, link_predictor,
                                                                                        test_data,
                                                                                        full_pos_data, full_t, full_msg,
                                                                                        full_nodes, full_dst,
                                                                                        statistics,
                                                                                        test_old_mask,
                                                                                        nn_test_mask,
                                                                                        nn_one_test_mask,
                                                                                        nn_two_test_mask,
                                                                                        neg_test_mask)

            def print_scores(prefix, mar, miss_times, miss_rate):
                print(f"{prefix}_{mar=:.4f}, {prefix}_{miss_times=:.4f}, {prefix}_{miss_rate=:.4f}")

            print(f"  Epoch: {epoch:02d}, Loss: {train_loss:.4f}")
            print_scores("val", val_mar, val_miss_record.sum(), val_miss_record.float().mean())
            print_scores("val_old", val_old_mar, val_old_miss_record.sum(), val_old_miss_record.float().mean())
            print_scores("val_nn", val_nn_mar, val_nn_miss_record.sum(), val_nn_miss_record.float().mean())
            print_scores("val_nn_one", val_nn_one_mar, val_nn_one_miss_record.sum(),
                         val_nn_one_miss_record.float().mean())
            print_scores("val_nn_two", val_nn_two_mar, val_nn_two_miss_record.sum(),
                         val_nn_two_miss_record.float().mean())
            print_scores("test", test_mar, test_miss_record.sum(), test_miss_record.float().mean())
            print_scores("test_old", test_old_mar, test_old_miss_record.sum(), test_old_miss_record.float().mean())
            print_scores("test_nn", test_nn_mar, test_nn_miss_record.sum(), test_nn_miss_record.float().mean())
            print_scores("test_nn_one", test_nn_one_mar, test_nn_one_miss_record.sum(),
                         test_nn_one_miss_record.float().mean())
            print_scores("test_nn_two", test_nn_two_mar, test_nn_two_miss_record.sum(),
                         test_nn_two_miss_record.float().mean())

        elif TASK == "link_prediction":

            val_ap, val_auc, val_pos_acc, val_neg_acc, \
                val_old_ap, val_old_auc, val_old_pos_acc, val_old_neg_acc, \
                val_nn_ap, val_nn_auc, val_nn_pos_acc, val_nn_neg_acc, \
                val_nn_one_ap, val_nn_one_auc, val_nn_one_pos_acc, val_nn_one_neg_acc, \
                val_nn_two_ap, val_nn_two_auc, val_nn_two_pos_acc, val_nn_two_neg_acc, \
                _ = test(gnn, memory, link_predictor,
                         val_data, full_pos_data, full_t, full_msg, full_nodes, full_dst, statistics,
                         val_old_mask, nn_val_mask, nn_one_val_mask, nn_two_val_mask)

            test_ap, test_auc, test_pos_acc, test_neg_acc, \
                test_old_ap, test_old_auc, test_old_pos_acc, test_old_neg_acc, \
                test_nn_ap, test_nn_auc, test_nn_pos_acc, test_nn_neg_acc, \
                test_nn_one_ap, test_nn_one_auc, test_nn_one_pos_acc, test_nn_one_neg_acc, \
                test_nn_two_ap, test_nn_two_auc, test_nn_two_pos_acc, test_nn_two_neg_acc, \
                test_temporal_neg_acc = test(gnn, memory, link_predictor,
                                             test_data, full_pos_data, full_t, full_msg, full_nodes, full_dst,
                                             statistics,
                                             test_old_mask, nn_test_mask, nn_one_test_mask, nn_two_test_mask,
                                             neg_test_mask)

            def print_scores(prefix, ap, auc, pos_acc, neg_acc):
                print(f"{prefix}_{ap=:.4f}, {prefix}_{auc=:.4f}, {prefix}_{pos_acc=:.4f}, {prefix}_{neg_acc=:.4f}")

            print_scores("val", val_ap, val_auc, val_pos_acc, val_neg_acc)
            print_scores("val_old", val_old_ap, val_old_auc, val_old_pos_acc, val_old_neg_acc)
            print_scores("val_nn", val_nn_ap, val_nn_auc, val_nn_pos_acc, val_nn_neg_acc)
            print_scores("val_nn_one", val_nn_one_ap, val_nn_one_auc, val_nn_one_pos_acc, val_nn_one_neg_acc)
            print_scores("val_nn_two", val_nn_two_ap, val_nn_two_auc, val_nn_two_pos_acc, val_nn_two_neg_acc)
            print_scores("test", test_ap, test_auc, test_pos_acc, test_neg_acc)
            print_scores("test_old", test_old_ap, test_old_auc, test_old_pos_acc, test_old_neg_acc)
            print_scores("test_nn", test_nn_ap, test_nn_auc, test_nn_pos_acc, test_nn_neg_acc)
            print_scores("test_nn_one", test_nn_one_ap, test_nn_one_auc, test_nn_one_pos_acc, test_nn_one_neg_acc)
            print_scores("test_nn_two", test_nn_two_ap, test_nn_two_auc, test_nn_two_pos_acc, test_nn_two_neg_acc)

            print(f" Test_neg_ACC: {test_temporal_neg_acc:.4f}")

        epoch_total_time = time.time() - start_epoch
        print(f" Total_epoch_time: {epoch_total_time:.2f}s")
        print()


if __name__ == "__main__":
    all_args = get_args()
    all_args.device = torch.device("cuda" if not all_args.cpu_only and torch.cuda.is_available() else "cpu")
    print(all_args)

    # prepare_data()
    # prepare_models()
    device = all_args.device
    # Helper vector to map global node indices to local ones.
    # assoc = torch.empty(n_total_unique_nodes_real, dtype=torch.long, device=device)
    # n_id_old2new = torch.empty(n_total_unique_nodes_real, dtype=torch.long, device=device)

    seen_nodes = []
    main(all_args)

