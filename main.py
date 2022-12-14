import time
import math
from tqdm import tqdm

import numpy as np
import torch
from torch.nn import Linear
from sklearn.metrics import average_precision_score, roc_auc_score, mean_absolute_error, mean_squared_error, \
    mean_squared_log_error
from torch_geometric.nn.models.tgn import LastAggregator
from data.data_utils import get_data, get_neighbor_finder, compute_time_statistics
from models.abstract_sum import AbstractSum, DyrepAbstractSum
from models.distance_encoding import distance_encoding
from arg_parser import get_args
from models.memory import TGNMemory, IdentityMessage, TimeEncoder
from models import TimeEmbedding, GraphAttentionEmbedding


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


def train(args, gnn, memory, abstract_sum, link_predictor, optimizer, criterion, train_data,
          full_t, full_msg, train_nodes, train_dst, statistics):
    # set the models to state of training
    if args.use_memory:
        memory.train()
        memory.reset_state()  # Start with a fresh memory.
    gnn.train()
    link_predictor.train()
    abstract_sum.train()

    neighbor_finder = get_neighbor_finder(train_data)

    total_loss = 0

    num_batch = math.ceil(train_data.num_events / args.batch_size)

    loss = None
    optimizer.zero_grad()

    for i, batch in tqdm(enumerate(train_data.seq_batches(batch_size=args.batch_size)), total=num_batch):

        src, dst, t, cur_e_id = batch.src, batch.dst, batch.t, batch.e_id
        valid_mask = cur_e_id != -1
        src, dst, t, cur_i_id = src[valid_mask], dst[valid_mask], t[valid_mask], cur_e_id[valid_mask]
        # msg = full_msg[cur_e_id]

        neg_dst = do_neg_sampling(args, src, dst, train_nodes, train_dst)

        dyrep_emb, last_update = None, None

        if args.mode == "distance_encoding":
            z_pos_src, z_neg_src, z_pos_dst, z_neg_dst = get_distance_encoding(args, gnn, memory, neighbor_finder,
                                                                               src, dst, neg_dst, t, full_t, full_msg)
        else:
            dyrep_emb, last_update, z_pos_src, z_neg_src, z_pos_dst, z_neg_dst = get_temporal_embedding(args,
                                                                                                        gnn, memory,
                                                                                                        neighbor_finder,
                                                                                                        src, dst,
                                                                                                        neg_dst,
                                                                                                        t,
                                                                                                        full_t,
                                                                                                        full_msg,
                                                                                                        statistics)

        loss_cur_batch = compute_loss(args, abstract_sum, link_predictor, criterion,
                         src, last_update, t, z_neg_dst, z_neg_src, z_pos_dst, z_pos_src)

        if loss is None:
            loss = loss_cur_batch
        else:
            loss += loss_cur_batch
        if torch.isnan(loss).any() or torch.isinf(loss).any():  # print(f"{i}, loss: {loss}")
            print(f"loss: {loss}")

        total_loss += float(loss_cur_batch) * batch.num_events

        if args.use_memory:
            update_memory(args.mode, memory, src, dst, t, cur_e_id, full_msg, dyrep_emb)

        if (i + 1) % args.backprop_every_n_batches == 0 or i + 1 == num_batch:
            loss.backward()  # torch.nn.utils.clip_grad_norm_(all_parameters, max_norm=100, norm_type=2)
            optimizer.step()
            loss = 0
            optimizer.zero_grad()
            if args.use_memory:
                memory.detach()

    return total_loss / train_data.num_events


def update_memory(arg_mode, memory, src, dst, t, cur_e_id, full_msg, dyrep_emb):
    # Update memory and neighbor loader with ground-truth state.
    update_valid_mask = cur_e_id != -1
    valid_src, valid_dst, valid_t, valid_cur_e_id = \
        src[update_valid_mask], dst[update_valid_mask], t[update_valid_mask], cur_e_id[update_valid_mask]
    msg = full_msg[valid_cur_e_id]
    if arg_mode == "dyrep":
        assert dyrep_emb is not None, "dyrep_emb is None"
        # saved_edge_index, saved_e_id, saved_t = edge_index, e_id, t
        memory.update_state(valid_src, valid_dst, valid_t, msg, src_emb=dyrep_emb[:valid_src.size(0)],
                            dst_emb=dyrep_emb[valid_src.size(0):2 * valid_src.size(0)])
    else:
        memory.update_state(valid_src, valid_dst, valid_t, msg)
    seen_nodes.extend(torch.cat([valid_src, valid_dst]).unique().tolist())


def compute_loss(args, abstract_sum, link_predictor, criterion,
                 src, last_update, t, z_neg_dst, z_neg_src, z_pos_dst, z_pos_src):
    num_src = src.size(0)  # number of pos src
    num_pos = 2 * num_src  # total number of pos src and pos dst
    #  1. the last index of neg dst or 2. the split index for neg src and neg dst
    neg_split = (2 + args.num_neg_sample) * num_src
    if args.dyrep_loss:
        last_update_src, last_update_dst = last_update[:num_src], last_update[num_src:num_pos]
        pos_delta_t = (t - torch.max(last_update_src, last_update_dst)).float()

        # neg_src is in fact equal to pos_src
        last_update_neg_src, last_update_neg_dst = \
            last_update_src.repeat_interleave(args.num_neg_sample, dim=0), last_update[num_pos:neg_split]

        delta_t_neg = (t.repeat_interleave(args.num_neg_sample) - torch.max(last_update_neg_src,
                                                                       last_update_neg_dst)).float()
        loss_cur_batch = - torch.log(abstract_sum.intensity_func(pos_delta_t, z_pos_src, z_pos_dst,
                                                                 return_params=False)).sum() / z_pos_src.size(0)
        loss_cur_batch += args.n_coefficient * abstract_sum.intensity_func(delta_t_neg, z_neg_src, z_neg_dst,
                                                                      return_params=False).sum() / z_neg_src.size(0)

    else:
        pos_out = link_predictor(z_pos_src, z_pos_dst)
        neg_out = link_predictor(z_neg_src, z_neg_dst)
        loss_cur_batch = criterion(pos_out, torch.ones_like(pos_out))
        loss_cur_batch += args.n_coefficient * criterion(neg_out, torch.zeros_like(neg_out))
    return loss_cur_batch


def get_temporal_embedding(args, gnn, memory, neighbor_finder, src, dst, neg_dst, t, full_t, full_msg, statistics):
    num_src = src.size(0)  # number of pos src
    num_pos = 2 * num_src  # total number of pos src and pos dst
    #  1. the last index of neg dst or 2. the split index for neg src and neg dst
    neg_split = (2 + args.num_neg_sample) * num_src

    if args.dyrep_loss:
        # n_id = torch.cat([src, dst, neg_src, neg_dst])
        n_id = torch.cat([src, dst, neg_dst])
    else:
        n_id = torch.cat([src, dst, neg_dst])
    num_focused_node = n_id.size(0)
    ratio = int(num_focused_node / num_src)
    # timestamps = t.repeat(3).cpu().numpy()
    if args.can_see_neighbors_in_same_batch:
        timestamps = t.repeat(int(num_focused_node / src.size(0))).cpu().numpy()
    else:
        timestamps = np.array([t[0].item()]).repeat(num_focused_node)
    n_id, edge_index, e_id, root_id_in_batch = neighbor_finder.get_k_hop_temporal_neighbor(n_id.cpu().numpy(),
                                                                                           timestamps,
                                                                                           args.neighbor_size,
                                                                                           k=1)
    n_id, edge_index, e_id, root_id_in_batch = n_id.to(device), edge_index.to(device), e_id.to(
        device), root_id_in_batch.to(device)
    n_id_unique = n_id.unique()
    n_id_old2new = get_map_tensor(n_id_unique)

    if args.use_memory:
        z, last_update = memory(n_id_unique)
        z, last_update = z[n_id_old2new[n_id]], last_update[n_id_old2new[n_id]]
    else:
        z, last_update = torch.zeros(size=(len(n_id), args.memory_dim), device=device), \
                         torch.zeros(size=(len(n_id),), device=device)

    emb = None

    if args.mode == "jodie":

        time_diff = t.repeat(ratio) - last_update[:num_focused_node]
        if ratio == 2 + args.num_neg_sample:
            # negative samples are sampled for dst only
            # for pos src
            time_diff[:num_src] = (time_diff[:num_src] - statistics.mean_time_shift_src) / statistics.std_time_shift_src
            # for pos dst and neg dst
            time_diff[num_src:] = (time_diff[num_src:] - statistics.mean_time_shift_dst) / statistics.std_time_shift_dst
        elif ratio == 2 + 2 * args.num_neg_sample:
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

    elif args.mode == "tgat" or (args.mode == "tgn" and args.use_embedding):
        z = gnn(z, edge_index, t.repeat(ratio)[root_id_in_batch] - full_t[e_id], full_msg[e_id])

    elif args.mode == "dyrep":
        emb = gnn(z, edge_index, t.repeat(ratio)[root_id_in_batch] - full_t[e_id], full_msg[e_id])

    z_pos_src, z_pos_dst = z[:num_src], z[num_src:num_pos]
    # neg_src is in fact equal to pos_src
    z_neg_src, z_neg_dst = z[:num_src].repeat_interleave(args.num_neg_sample, dim=0), z[num_pos:neg_split]

    return emb, last_update, z_pos_src, z_neg_src, z_pos_dst, z_neg_dst


# Return a helper vector to map global node indices to new local ones.
def get_map_tensor(n_id_unique):
    n_id_unique = n_id_unique.unique()
    max_n_id_unique = n_id_unique.max().item()
    n_id_old2new = torch.empty(max_n_id_unique + 1, dtype=torch.long, device=device)
    n_id_old2new[n_id_unique] = torch.arange(n_id_unique.size(0), device=device)
    return n_id_old2new


def get_distance_encoding(args, gnn, memory, neighbor_finder, src, dst, neg_dst, t, full_t, full_msg):
    z_src, z_dst = distance_encoding(src.repeat(args.num_neg_sample + 1), torch.cat([dst, neg_dst]),
                                     t.repeat(args.num_neg_sample + 1), neighbor_finder,
                                     n_neighbors=args.neighbor_size,
                                     USE_MEMORY_EMBEDDING=args.use_memory_embedding,
                                     DE_SHORTEST_PATH=args.de_shortest_path,
                                     DE_RANDOM_WALK=args.de_random_walk,
                                     NODE_FEATURE_DIM=args.memory_dim, full_t=full_t, full_msg=full_msg,
                                     memory=memory,
                                     gnn=gnn, k=2)
    z_pos_src, z_pos_dst = z_src[:src.size(0)], z_dst[:src.size(0)]
    z_neg_src, z_neg_dst = z_src[src.size(0):], z_dst[src.size(0):]
    return z_pos_src, z_neg_src, z_pos_dst, z_neg_dst


def do_neg_sampling(args, src, dst, nodes, dst_nodes):
    # Sample negative destination nodes.
    if args.sample_neg_dst_strictly:
        sample_mask = torch.ones(size=(src.size(0) * args.num_neg_sample,), device=device).bool()
        neg_dst = torch.empty(size=(src.size(0) * args.num_neg_sample,), device=device, dtype=src.dtype)
        while sample_mask.any():
            neg_dst_index = torch.randint(0, dst_nodes.size(0), (sample_mask.sum(),),
                                          dtype=torch.long, device=device)
            neg_dst[sample_mask] = dst_nodes[neg_dst_index]
            sample_mask = torch.logical_or(neg_dst == dst.repeat_interleave(args.num_neg_sample),
                                           neg_dst == src.repeat_interleave(args.num_neg_sample))

    else:
        neg_dst_index = torch.randint(0, nodes.size(0), (src.size(0),),
                                      dtype=torch.long, device=device)
        neg_dst = nodes[neg_dst_index]
    return neg_dst


@torch.no_grad()
def test(args, memory, gnn, link_predictor,
         inference_data, full_pos_data, full_t, full_msg, full_nodes, full_dst, statistics,
         old_mask, nn_mask, nn_one_mask, nn_two_mask, temporal_neg_mask=None):
    neighbor_finder = get_neighbor_finder(full_pos_data)

    if temporal_neg_mask is None:
        temporal_neg_mask = np.zeros(old_mask.shape).astype(bool)

    if args.use_memory:
        memory.eval()

    gnn.eval()
    link_predictor.eval()

    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.

    # aps, aucs = [], []
    all_pos_out, all_neg_out = [], []
    for batch in inference_data.seq_batches(batch_size=args.batch_size):
        src, dst, t, cur_e_id = batch.src, batch.dst, batch.t, batch.e_id

        neg_dst = do_neg_sampling(args, src, dst, full_nodes, full_dst)

        dyrep_emb = None
        if args.mode == "distance_encoding":
            z_pos_src, z_neg_src, z_pos_dst, z_neg_dst = get_distance_encoding(args, gnn, memory, neighbor_finder,
                                                                               src, dst, neg_dst, t, full_t, full_msg)
        else:
            dyrep_emb, last_update, z_pos_src, z_neg_src, z_pos_dst, z_neg_dst = get_temporal_embedding(args,
                                                                                                        gnn, memory,
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

        if args.use_memory:
            update_memory(args.mode, memory, src, dst, t, cur_e_id, full_msg, dyrep_emb)

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
def test_time_prediction(args, gnn, memory, abstract_sum, link_predictor,
                         inference_data, full_pos_data, full_t,  full_msg, statistics):
    if not args.use_memory:
        raise NotImplementedError("Time prediction depends on node memory")

    memory.eval()
    gnn.eval()
    if args.rank_score == "link_pred":
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

        if args.only_predict_last_events_time:
            src, dst, t, e_id = retrieve_last_events(src, dst, t, e_id)

        dyrep_emb = None
        if args.mode == "distance_encoding":

            z_src, z_dst = distance_encoding(src, dst, t, neighbor_finder,
                                             n_neighbors=args.neighbor_size,
                                             USE_MEMORY_EMBEDDING=args.use_memory_embedding,
                                             DE_SHORTEST_PATH=args.de_shortest_path, DE_RANDOM_WALK=args.de_random_walk,
                                             NODE_FEATURE_DIM=args.memory_dim, full_t=full_t, full_msg=full_msg,
                                             memory=memory,
                                             gnn=gnn, k=2)
            _, last_update_src = memory(src)
            _, last_update_dst = memory(dst)

            # lgh: It's safe to set `assoc` to None here,
            # because the outside will use `assoc` only if `args.mode` == "dyrep"
            assoc = None

        else:
            n_id, edge_index, edge_id = get_k_hop_temp_neighbors(args, neighbor_finder, src, dst, t)

            assoc = get_map_tensor(n_id)

            z, last_update = memory(n_id)

            dyrep_emb, z_src, z_dst = get_src_and_dst_embeddings(args, gnn, edge_index, src, dst, t, full_msg,
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
        if args.mode == "dyrep":
            memory.update_state(src, dst, t, msg, src_emb=dyrep_emb[assoc[src]], dst_emb=dyrep_emb[assoc[dst]])
        else:
            memory.update_state(src, dst, t, msg)

    # all_t_estimated = np.clip(5000 * all_t_estimated, -np.inf, 100000)
    mae = float(mean_absolute_error(all_t_true, all_t_estimated))
    rmse = float(np.sqrt(mean_squared_error(all_t_true, all_t_estimated)))
    rmsle = float(np.sqrt(mean_squared_log_error(all_t_true, all_t_estimated)))

    return mae, rmse, rmsle


def get_src_and_dst_embeddings(args, gnn, edge_index, src, dst, t, full_msg, e_id, z, last_update, statistics, assoc):
    dyrep_emb = None

    if args.mode == "jodie":

        time_diff_src = t - last_update[assoc[src]]
        time_diff_src = (time_diff_src - statistics.mean_time_shift_src) / statistics.std_time_shift_src
        time_diff_pos_dst = t - last_update[assoc[dst]]
        time_diff_pos_dst = (time_diff_pos_dst - statistics.mean_time_shift_dst) / statistics.std_time_shift_dst

        z_src = z[assoc[src]]
        z_dst = z[assoc[dst]]

        z_src = gnn(z_src, time_diff_src)
        z_dst = gnn(z_dst, time_diff_pos_dst)

    else:

        if args.mode == "tgat" or (args.mode == "tgn" and args.use_embedding):
            z = gnn(z, edge_index, torch.tensor([0]).float().repeat(edge_index.size(1)), full_msg[e_id])
            print(f"z.size(): {z.size()}")
            print(f"assoc[candidates]: {assoc[dst]}")
        elif args.mode == "dyrep":
            dyrep_emb = gnn(z, edge_index, torch.tensor([0]).float().repeat(edge_index.size(1)), full_msg[e_id])
        elif args.mode == "tgn" and not args.use_embedding:
            pass
        else:
            raise NotImplementedError

        z_src = z[assoc[src]]
        z_dst = z[assoc[dst]]

    return dyrep_emb, z_src, z_dst


def get_k_hop_temp_neighbors(args, neighbor_finder, src, dst, t):
    n_id = torch.cat([src, dst]).unique()
    edge_index, e_id = None, None
    if not (not args.use_embedding and args.mode == "tgn"):
        n_id, edge_index, e_id = \
            neighbor_finder.get_k_hop_temporal_neighbor_by_one_timestamp(n_id, t[0], args.neighbor_size, k=1)
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
def test_temporal_recommendation(args, gnn, memory, abstract_sum, link_predictor, inference_data,
                                 full_pos_data, full_t, full_msg, full_nodes, full_dst, statistics,
                                 old_mask, nn_mask, nn_one_mask, nn_two_mask, temporal_neg_mask=None):
    if temporal_neg_mask is None:
        temporal_neg_mask = np.zeros(old_mask.shape).astype(bool)

    valid_mask = ~temporal_neg_mask
    old_mask, nn_mask, nn_one_mask, nn_two_mask = old_mask[valid_mask], \
                                                  nn_mask[valid_mask], \
                                                  nn_one_mask[valid_mask], \
                                                  nn_two_mask[valid_mask]

    if args.use_memory:
        memory.eval()
    gnn.eval()
    if args.rank_score == "link_pred":
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
            get_candidates(args.candidates_from, neighbor_finder, src, t, full_nodes, full_dst)

        dyrep_emb, last_update = None, None
        if args.mode == "distance_encoding":  # In this case, it is certain that USE_MEMORY == False

            n_id = root_nodes
            assoc = get_map_tensor(n_id)
            last_update = torch.zeros(len(n_id), device=device)

            z_root, z_candidate = distance_encoding(root_nodes, candidates, recommend_times, neighbor_finder,
                                                    n_neighbors=args.neighbor_size,
                                                    USE_MEMORY_EMBEDDING=args.use_memory_embedding,
                                                    DE_SHORTEST_PATH=args.de_shortest_path,
                                                    DE_RANDOM_WALK=args.de_random_walk,
                                                    NODE_FEATURE_DIM=args.memory_dim,
                                                    full_t=full_t,
                                                    full_msg=full_msg,
                                                    memory=memory,
                                                    gnn=gnn, k=2)

        else:
            n_id, edge_index, edge_id = get_k_hop_temp_neighbors(args, neighbor_finder, src, dst, t)
            assoc = get_map_tensor(n_id)

            if n_id.size(0) == 0:
                print("n_id.size(0) == 0 in this batch, continue")
                continue

            if args.use_memory:
                z, last_update = memory(n_id)
            else:
                z, last_update = torch.zeros(len(n_id), args.memory_dim, device=device), \
                                 torch.zeros(len(n_id), device=device)

            dyrep_emb, z_root, z_candidate = \
                get_src_and_dst_embeddings(args, gnn, edge_index, src, dst, t, full_msg, edge_id, z, last_update,
                                           statistics, assoc)

        last_update_root, last_update_candidate = last_update[assoc[root_nodes]], last_update[assoc[candidates]]
        last_time_max = torch.max(torch.cat((last_update_root, last_update_candidate), dim=-1), dim=-1)[0]
        delta_t = recommend_times - last_time_max

        rank_scores = get_rank_scores(args.rank_score, abstract_sum, link_predictor, z_root, z_candidate, delta_t)

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

        if args.use_memory:
            msg = full_msg[e_id]
            if args.mode == "dyrep":
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


def get_rank_scores(arg_rank_score, abstract_sum, link_predictor, z_root, z_candidate, delta_t):
    if arg_rank_score == "likelihood":
        rank_scores = abstract_sum.likelihood_func(delta_t, z_src=z_root, z_dst=z_candidate)
    elif arg_rank_score == "intensity":
        rank_scores = abstract_sum(z_root, z_candidate, delta_t)
    elif arg_rank_score == "link_pred":
        rank_scores = link_predictor(z_root, z_candidate).squeeze()
    else:
        raise NotImplementedError
    return rank_scores


def get_candidates(arg_candidates_from, neighbor_finder, src, t, full_nodes, full_dst):
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

    if arg_candidates_from == "neighbors":
        candidates, root_nodes, root_ids, recommend_times = \
            neighbor_finder.get_temporal_recommendation_candidates(src.cpu().numpy(), t.cpu().numpy())

        candidates, root_nodes, root_ids, recommend_times = torch.from_numpy(candidates).to(device), \
                                                            torch.from_numpy(root_nodes).to(device), \
                                                            torch.from_numpy(root_ids).to(device), \
                                                            torch.from_numpy(recommend_times).to(device)
    elif arg_candidates_from == "all_dst":
        candidates, root_nodes, root_ids, recommend_times = get_candidates_from(full_dst)

    elif arg_candidates_from == "all_nodes":
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
        = get_data(args.name, use_neg=args.use_neg, use_msg=args.use_msg, dataset_size=args.dataset_size,
                   cpu_only=args.cpu_only, nn_ratio=0)
    train_src = torch.unique(train_data.src, sorted=True)
    train_dst = torch.unique(train_data.dst, sorted=True)
    train_nodes = torch.unique(torch.cat([train_src, train_dst]), sorted=True)
    full_src = torch.unique(full_pos_data.src, sorted=True)
    full_dst = torch.unique(full_pos_data.dst, sorted=True)
    full_nodes = torch.unique(torch.cat([full_src, full_dst]), sorted=True)
    # mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst, mean_time_shift, std_time_shift
    statistics = compute_time_statistics(full_pos_data.src, full_pos_data.dst, full_pos_data.t)
    print("data prepared")

    if args.use_memory:
        memory = TGNMemory(
            n_total_unique_nodes_real,
            full_msg.size(-1),
            args.memory_dim,
            args.embedding_dim,
            args.time_dim,
            message_module=IdentityMessage(full_msg.size(-1), args.time_dim, args.memory_dim, args.embedding_dim,
                                           mode=args.mode),
            aggregator_module=LastAggregator(),
            mode=args.mode
        ).to(device)
    else:
        memory = None

    arg_dict = {"args": args,
                "num_nodes": n_total_unique_nodes_real,
                "in_channels": args.memory_dim if args.mode != "distance_encoding" else
                5 * args.de_shortest_path + 4 * args.de_random_walk,
                "out_channels": args.embedding_dim,
                "msg_dim": full_msg.size(-1),
                "time_enc": memory.time_enc if memory is not None else
                TimeEncoder(out_channels=args.time_dim, mode=args.mode).to(device),
                "n_layer": args.n_layer,
                "device": device}
    if args.mode == "jodie":
        gnn = TimeEmbedding(**arg_dict).to(device)
    elif args.mode == "distance_encoding":
        gnn = GraphAttentionEmbedding(**arg_dict).to(device)
    else:
        gnn = GraphAttentionEmbedding(**arg_dict).to(device)

    link_predictor = LinkPredictor(in_channels=args.embedding_dim).to(device)

    if not args.use_dyrepabstractsum:
        # in_channels, num_basis, num_param, mc_num_surv = 1, mc_num_time = 3
        abstract_sum = AbstractSum(2 * args.embedding_dim, num_basis=args.num_basis, mc_num_surv=args.mc_num_surv,
                                   mc_num_time=args.mc_num_time)
    else:
        # in_channels, num_basis, num_param, mc_num_surv = 1, mc_num_time = 3
        abstract_sum = DyrepAbstractSum(2 * args.embedding_dim,
                                        mc_num_surv=args.mc_num_surv, mc_num_time=args.mc_num_time)

    if args.use_memory:
        memory_parameter_set = set(memory.parameters())
    else:
        memory_parameter_set = set()

    all_parameters = memory_parameter_set | set(gnn.parameters()) | set(abstract_sum.parameters()) \
                     | set(link_predictor.parameters())

    optimizer = torch.optim.Adam(all_parameters, lr=args.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(1, args.num_epoch + 1):
        start_epoch = time.time()

        print(f"start epoch {epoch} ...")

        train_loss = train(args, gnn, memory, abstract_sum, link_predictor, optimizer, criterion, train_data,
                           full_t, full_msg, train_nodes, train_dst, statistics)
        epoch_train_time = time.time() - start_epoch

        print(f"  Epoch: {epoch:02d}, Loss: {train_loss:.4f}")
        print(f"  Train_time: {epoch_train_time:.2f}s")

        if args.task == "time_prediction":

            val_mae, val_rmse, val_rmsle = test_time_prediction(args, gnn, memory, abstract_sum, link_predictor,
                                                                val_data, full_pos_data, full_t,  full_msg,
                                                                statistics)
            test_mae, test_rmse, test_rmsle = test_time_prediction(args, gnn, memory, abstract_sum, link_predictor,
                                                                   test_data, full_pos_data, full_t,  full_msg,
                                                                   statistics)

            print(f"Val_MAE: {val_mae:.4f}, Val_RMSE: {val_rmse:.4f}, Val_RMSLE: {val_rmsle:.4f}")
            print(f"Test_MAE: {test_mae:.4f}, Test_RMSE: {test_rmse:.4f}, Test_RMSLE: {test_rmsle:.4f}")

        elif args.task == "temporal_recommendation":

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

        elif args.task == "link_prediction":

            val_ap, val_auc, val_pos_acc, val_neg_acc, \
                val_old_ap, val_old_auc, val_old_pos_acc, val_old_neg_acc, \
                val_nn_ap, val_nn_auc, val_nn_pos_acc, val_nn_neg_acc, \
                val_nn_one_ap, val_nn_one_auc, val_nn_one_pos_acc, val_nn_one_neg_acc, \
                val_nn_two_ap, val_nn_two_auc, val_nn_two_pos_acc, val_nn_two_neg_acc, \
                _ = test(args, gnn, memory, link_predictor,
                         val_data, full_pos_data, full_t, full_msg, full_nodes, full_dst, statistics,
                         val_old_mask, nn_val_mask, nn_one_val_mask, nn_two_val_mask)

            test_ap, test_auc, test_pos_acc, test_neg_acc, \
                test_old_ap, test_old_auc, test_old_pos_acc, test_old_neg_acc, \
                test_nn_ap, test_nn_auc, test_nn_pos_acc, test_nn_neg_acc, \
                test_nn_one_ap, test_nn_one_auc, test_nn_one_pos_acc, test_nn_one_neg_acc, \
                test_nn_two_ap, test_nn_two_auc, test_nn_two_pos_acc, test_nn_two_neg_acc, \
                test_temporal_neg_acc = test(args, gnn, memory, link_predictor,
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

