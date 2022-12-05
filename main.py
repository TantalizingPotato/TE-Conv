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
# from models.distance_encoding import distance_encoding
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


def train(args, gnn, memory, abstract_sum, link_predictor, all_parameters, optimizer, criterion, train_data,
          full_pos_data, full_t, full_msg, train_nodes, train_dst, statistics):
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

    milestone_basis = num_batch // 5

    #     for i, batch in tqdm(enumerate(train_data.seq_batches(batch_size=args.batch_size)), total=num_batch):
    for i, batch in enumerate(train_data.seq_batches(batch_size=args.batch_size)):

        src, dst, t, cur_e_id = batch.src, batch.dst, batch.t, batch.e_id
        valid_mask = cur_e_id != -1
        src, dst, t, cur_i_id = src[valid_mask], dst[valid_mask], t[valid_mask], cur_e_id[valid_mask]
        # msg = full_msg[cur_e_id]

        neg_dst = do_neg_sampling(args, src, dst, train_nodes, train_dst)

        dyrep_emb, last_update = None, None

        dyrep_emb, last_update, z_pos_src, z_neg_src, z_pos_dst, z_neg_dst = get_temporal_embedding(args,
                                                                                                    gnn, memory,
                                                                                                    neighbor_finder,
                                                                                                    src, dst,
                                                                                                    neg_dst,
                                                                                                    t,
                                                                                                    full_t,
                                                                                                    full_msg,
                                                                                                    statistics)

        # loss_cur_batch = compute_loss(args, abstract_sum, link_predictor, criterion,
        #                               src, last_update, t, z_neg_dst, z_neg_src, z_pos_dst, z_pos_src)

        # calculating loss
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
            loss_cur_batch += args.n_coefficient * \
                              abstract_sum.intensity_func(delta_t_neg, z_neg_src, z_neg_dst,
                                                          return_params=False).sum() / z_neg_src.size(0)
        else:
            # if args.other_loss == "time_prediction":
            #     t_true, t_estimated = test_time_prediction(args, gnn, memory, abstract_sum, link_predictor,
            #                                                val_data, full_pos_data, full_t, full_msg,
            #                                                statistics)
            #     loss_cur_batch = None
            # elif args.other_loss == "link_prediction":
            pos_out = link_predictor(z_pos_src, z_pos_dst)
            neg_out = link_predictor(z_neg_src, z_neg_dst)
            loss_cur_batch = criterion(pos_out, torch.ones_like(pos_out))
            loss_cur_batch += args.n_coefficient * criterion(neg_out, torch.zeros_like(neg_out))

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
            loss.backward()
            if args.mode == "jodie":
                torch.nn.utils.clip_grad_norm_(all_parameters, max_norm=100, norm_type=2)
            optimizer.step()
            loss = 0
            optimizer.zero_grad()
            if args.use_memory:
                memory.detach()

        if (i + 1) % milestone_basis == 0 or (i + 1) == num_batch:
            print(f"{100 * (i + 1) / num_batch:.2f}% trained for this epoch ...")

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


def train_time_prediction(args, gnn, memory, abstract_sum, link_predictor, all_parameters, optimizer, criterion,
                          train_data, full_pos_data, full_t, full_msg, train_nodes, train_dst, statistics):
    if args.batch_size != 1:
        raise NotImplementedError("Currently, for time prediction, batch size should be one")

    # set the models to state of training
    if args.use_memory:
        memory.train()
    memory.reset_state()  # Start with a fresh memory.
    gnn.train()
    link_predictor.train()
    abstract_sum.train()

    # neighbor_finder = get_neighbor_finder(train_data)

    total_loss = 0

    loss_function_mse = torch.nn.MSELoss()
    loss = None
    optimizer.zero_grad()

    # num_batch = math.ceil(train_data.num_events / args.batch_size)
    num_batch = train_data.num_batch
    milestone_basis = num_batch // 5

    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.

    neighbor_finder = get_neighbor_finder(full_pos_data)

    # all_t_true, all_t_estimated = np.array([]), np.array([])

    # all_last_update = memory.last_update
    # print(f"all_last_update: {all_last_update}")
    # print(f"last_update[100]: {all_last_update[100]}")
    count_trained_instance = 0
    for i, batch in enumerate(train_data.seq_batches(batch_size=args.batch_size)):
    # for i, batch in tqdm(enumerate(train_data.seq_batches(batch_size=args.batch_size)), total=num_batch):
        # for batch in inference_data.seq_batches(batch_size=1):
        src, dst, t, e_id = get_valid_data(batch)

        if src.size(0) == 0:
            print("train_time_prediction: No pos_sample in this batch, continue")
            continue

        # if args.only_predict_last_events_time:
        #     src, dst, t, e_id = retrieve_last_events(src, dst, t, e_id)

        dyrep_emb = None

        n_id, edge_index, edge_id = get_k_hop_temp_neighbors(args, neighbor_finder, src, dst, t)

        assoc = get_map_tensor(n_id)

        # all_last_update = memory.last_update
        # print(f"all_last_update: {all_last_update}")
        # print(f"last_update[100]: {all_last_update[100]}")
        # print(f"all_last_update[n_id]: {all_last_update[n_id]}")
        if args.use_memory:
            z, last_update = memory(n_id)
        else:
            _, last_update = memory(n_id)
            z = torch.zeros((n_id.size(0), args.memory_dim), dtype=torch.float, device=args.device)

        dyrep_emb, z_src, z_dst = get_src_and_dst_embeddings(args, gnn, edge_index, src, dst, t, full_msg,
                                                             edge_id, z, last_update, statistics, assoc)

        # print(f"last_update: {last_update}")
        last_update_src, last_update_dst = last_update[assoc[src]], last_update[assoc[dst]]

        t_true = (t - torch.max(last_update_src, last_update_dst)).float()


        seen_mask = (last_update_src == last_update_dst) &(t - t_true > 1e-3) & (t_true - 0 > 1e-3)

        seen_z_src, seen_z_pos_dst, seen_t_true, seen_src, seen_pos_dst, seen_e_id \
            = z_src[seen_mask], z_dst[seen_mask], t_true[seen_mask], \
              src[seen_mask], dst[seen_mask], e_id[seen_mask]

        pos_delta_t = seen_t_true
        batch_size = seen_z_src.size(0)
        if batch_size == 0:
            # print(f"t: {t}")
            # print(f"t_true: {t_true}")
            # print(f"e_id: {e_id}, src:{src}, dst:{dst}, all nodes unseen in this batch, continue to the next batch")
            msg = full_msg[e_id]
            if args.use_memory:
                if args.mode == "dyrep":
                    memory.update_state(src, dst, t, msg, src_emb=dyrep_emb[assoc[src]], dst_emb=dyrep_emb[assoc[dst]])
                else:
                    memory.update_state(src, dst, t, msg)
            continue

        # print(f"z_src.device: {z_src.device}")
        # print(f"z_dst.device: {z_dst.device}")

        t_estimated, abstract_sum_params = abstract_sum.estimate_time(z_src[seen_mask], z_dst[seen_mask],
                                                                      return_params=True)
        t_estimated = torch.clamp(
            t_estimated * 100,  # t_estimated * 5000
            max=100000)

        int_vals = abstract_sum.intensity_func(pos_delta_t, seen_z_src, seen_z_pos_dst, params=abstract_sum_params,
                                               return_params=False)
        loss_cur_batch = - torch.log(int_vals).sum() / batch_size
        # print(f"pos_delta_t.device: {pos_delta_t.device}")
        loss_cur_batch += args.n_coefficient * \
                          abstract_sum.compensator_func(pos_delta_t, seen_z_src, seen_z_pos_dst,
                                                        params=abstract_sum_params).sum() / batch_size
        loss_cur_batch_mse = loss_function_mse(t_estimated, seen_t_true) / batch_size
        loss_cur_batch += args.mse_loss_coefficient * loss_cur_batch_mse

        if loss is None:
            loss = loss_cur_batch
        else:
            loss += loss_cur_batch
        if torch.isnan(loss).any() or torch.isinf(loss).any():  # print(f"{i}, loss: {loss}")
            print(f"loss: {loss}")

        # total_loss += float(loss_cur_batch) * batch.num_events
        total_loss += float(loss_cur_batch) * batch_size

        # count_trained_instance += 1
        count_trained_instance += batch_size

        msg = full_msg[e_id]
        # if args.use_memory:
        if args.mode == "dyrep":
            memory.update_state(src, dst, t, msg, src_emb=dyrep_emb[assoc[src]], dst_emb=dyrep_emb[assoc[dst]])
        else:
            memory.update_state(src, dst, t, msg)

        if (i + 1) % args.backprop_every_n_batches == 0 or i + 1 == num_batch:
            loss.backward()
            if args.mode == "jodie":
                torch.nn.utils.clip_grad_norm_(all_parameters, max_norm=100, norm_type=2)
            optimizer.step()
            loss = 0
            optimizer.zero_grad()
            if args.use_memory:
                memory.detach()

        if (i + 1) % milestone_basis == 0 or (i + 1) == num_batch:
            print(f"{100 * (i + 1) / num_batch:.2f}% trained for this epoch ...")

    print(f"count_trained_instance: {count_trained_instance}")
    return total_loss / (count_trained_instance + 1e-9)


@torch.no_grad()
def test_time_prediction(args, gnn, memory, abstract_sum, link_predictor,
                         inference_data, full_pos_data, full_t, full_msg, statistics):
    # if not args.use_memory:
    #     raise NotImplementedError("Time prediction depends on node memory")

    memory.eval()
    gnn.eval()
    if args.rank_score == "link_pred":
        link_predictor.eval()
    else:
        abstract_sum.eval()

    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.

    neighbor_finder = get_neighbor_finder(full_pos_data)

    all_t_true = torch.tensor([], dtype=torch.float).to(device)
    all_t_estimated = torch.tensor([], dtype=torch.float).to(device)

    for batch in inference_data.seq_batches(batch_size=1):
        # src, dst, t, e_id = batch.src, batch.dst, batch.t, batch.e_id
        # print(f"src:{src},"
        #       f"dst:{dst},"
        #       f"t:{t}"
        #       f"e_id:{e_id}")
        src, dst, t, e_id = get_valid_data(batch)

        if src.size(0) == 0:
            print("test_time_prediction: No pos_sample in this batch, continue")
            continue

        if args.only_predict_last_events_time:
            src, dst, t, e_id = retrieve_last_events(src, dst, t, e_id)

        dyrep_emb = None

        n_id, edge_index, edge_id = get_k_hop_temp_neighbors(args, neighbor_finder, src, dst, t)

        assoc = get_map_tensor(n_id)

        z, last_update = memory(n_id)

        dyrep_emb, z_src, z_dst = get_src_and_dst_embeddings(args, gnn, edge_index, src, dst, t, full_msg,
                                                             edge_id, z, last_update, statistics, assoc)

        last_update_src, last_update_dst = last_update[assoc[src]], last_update[assoc[dst]]

        t_true = t - torch.max(torch.cat((last_update_src, last_update_dst), dim=-1), dim=-1)[0]

        # seen_mask = (t_true != t) & (t_true != 0)
        # seen_mask = (t - t_true > 1e-3) & (t_true - 0 > 1e-3)
        seen_mask = (last_update_src == last_update_dst) & (t - t_true > 1e-3) & (t_true - 0 > 1e-3)
        # seen_mask = (last_update_src == last_update_dst) & (t_true != t)
        seen_z_src, seen_z_pos_dst, seen_t_true, seen_src, seen_pos_dst, seen_e_id \
            = z_src[seen_mask], z_dst[seen_mask], t_true[seen_mask], \
              src[seen_mask], dst[seen_mask], e_id[seen_mask]

        batch_size = seen_z_src.size(0)
        if batch_size == 0:
            # print(f"e_id: {e_id}, src:{src}, dst:{dst}, all nodes unseen in this batch, continue to the next batch")
            pass
        else:
            t_estimated = torch.clamp(
                abstract_sum.estimate_time(z_src[seen_mask], z_dst[seen_mask]) * 100,  # edited by lgh, * 5000
                max=100000)
            # t_estimated = np.clip(
            #     abstract_sum.estimate_time(z_src[seen_mask], z_dst[seen_mask]).cpu().numpy() * 5000,
            #     -np.inf,
            #     100000)

            # seen_t_true = seen_t_true.cpu().numpy()
            # print(f"e_id: {seen_e_id}, src:{seen_src}, dst:{seen_pos_dst}, "
            #       f"t_estimated: {t_estimated}, t_true: {t_true}")
            # absolute_error = len(seen_t_true) * mean_absolute_error(seen_t_true, t_estimated)
            # print(f"absolute_error: {absolute_error}")

            # print(f"all_t_true.device: {all_t_true.device}")
            # print(f"seen_t_true.device: {seen_t_true.device}")
            all_t_true, all_t_estimated = torch.cat([all_t_true, seen_t_true]), \
                                          torch.cat([all_t_estimated, t_estimated])

            print(f"seen_t_true: {seen_t_true},"
                  f"t_estimated: {t_estimated}")
            # all_t_true, all_t_estimated = np.concatenate([all_t_true, seen_t_true]), \
            #                               np.concatenate([all_t_estimated, t_estimated])

        msg = full_msg[e_id]
        if args.use_memory:
            if args.mode == "dyrep":
                memory.update_state(src, dst, t, msg, src_emb=dyrep_emb[assoc[src]], dst_emb=dyrep_emb[assoc[dst]])
            else:
                memory.update_state(src, dst, t, msg)

    # # all_t_estimated = np.clip(5000 * all_t_estimated, -np.inf, 100000)
    all_t_true, all_t_estimated = all_t_true.cpu().numpy(), all_t_estimated.cpu().numpy()
    mae = float(mean_absolute_error(all_t_true, all_t_estimated))
    rmse = float(np.sqrt(mean_squared_error(all_t_true, all_t_estimated)))
    rmsle = float(np.sqrt(mean_squared_log_error(all_t_true, all_t_estimated)))

    return mae, rmse, rmsle
    # return all_t_true, all_t_estimated


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
            time_diff[num_src:num_pos] = \
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


def compute_scores(labels, predictions, mask):
    ap, auc, pos_acc, neg_acc = average_precision_score(labels[mask], predictions[mask]), \
                                roc_auc_score(labels[mask], predictions[mask]), \
                                (predictions[mask][labels[mask].bool()] >= 0.5).float().mean(), \
                                (predictions[mask][~labels[mask].bool()] < 0.5).float().mean()
    return ap, auc, pos_acc, neg_acc


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
            z = gnn(z, edge_index, torch.tensor([0]).float().repeat(edge_index.size(1)).to(device), full_msg[e_id])
            # print(f"z.size(): {z.size()}")
            # print(f"assoc[candidates]: {assoc[dst]}")
        elif args.mode == "dyrep":
            dyrep_emb = gnn(z, edge_index, torch.tensor([0]).float().repeat(edge_index.size(1)).to(device), full_msg[e_id])
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

    # if args.use_memory:
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
    # else:
    #     memory = None

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
        abstract_sum = AbstractSum(2 * args.embedding_dim, num_basis=args.num_basis, basis=args.basis,
                                   mc_num_surv=args.mc_num_surv, mc_num_time=args.mc_num_time,
                                   device=device)
    else:
        # in_channels, num_basis, num_param, mc_num_surv = 1, mc_num_time = 3
        abstract_sum = DyrepAbstractSum(2 * args.embedding_dim,
                                        mc_num_surv=args.mc_num_surv, mc_num_time=args.mc_num_time,
                                        device=device)

    if args.use_memory:
        memory_parameter_set = set(memory.parameters())
    else:
        memory_parameter_set = set()

    all_parameters = memory_parameter_set | set(gnn.parameters()) | set(abstract_sum.parameters()) \
                     | set(link_predictor.parameters())

    optimizer = torch.optim.Adam(all_parameters, lr=args.learning_rate)
    # criterion = torch.nn.BCEWithLogitsLoss()
    criterion = torch.nn.BCELoss()

    for epoch in range(1, args.num_epoch + 1):
        start_epoch = time.time()

        print(f"start epoch {epoch} ...")

        if args.task == "time_prediction":
            train_loss = train_time_prediction(args, gnn, memory, abstract_sum, link_predictor, all_parameters,
                                               optimizer, criterion, train_data, full_pos_data, full_t, full_msg,
                                               train_nodes, train_dst, statistics)
        else:
            train_loss = train(args, gnn, memory, abstract_sum, link_predictor, all_parameters, optimizer, criterion,
                           train_data, full_pos_data, full_t, full_msg, train_nodes, train_dst, statistics)

        epoch_train_time = time.time() - start_epoch
        print(f"  Epoch: {epoch:02d}, Loss: {train_loss:.4f}")
        print(f"  Train_time: {epoch_train_time:.2f}s")

        if args.task == "time_prediction":

            val_mae, val_rmse, val_rmsle = test_time_prediction(args, gnn, memory, abstract_sum, link_predictor,
                                                                val_data, full_pos_data, full_t, full_msg,
                                                                statistics)

            test_mae, test_rmse, test_rmsle = test_time_prediction(args, gnn, memory, abstract_sum, link_predictor,
                                                                   test_data, full_pos_data, full_t, full_msg,
                                                                   statistics)

            print(f"Val_MAE: {val_mae:.4f}, Val_RMSE: {val_rmse:.4f}, Val_RMSLE: {val_rmsle:.4f}")
            print(f"Test_MAE: {test_mae:.4f}, Test_RMSE: {test_rmse:.4f}, Test_RMSLE: {test_rmsle:.4f}")

        else:
            raise NotImplementedError("In this project, only time prediction task is supported.")

        epoch_total_time = time.time() - start_epoch
        print(f" Total_epoch_time: {epoch_total_time:.2f}s")
        print()


if __name__ == "__main__":
    all_args = get_args()
    all_args.device = torch.device("cuda" if not all_args.cpu_only and torch.cuda.is_available() else "cpu")
    print(all_args)

    assert all_args.use_memory == False or (all_args.mode != "tgat" and all_args.mode != "distance_encoding"), \
        "tgat 和 distance_encoding 不支持 memory"

    # prepare_data()
    # prepare_models()
    device = all_args.device
    # Helper vector to map global node indices to local ones.
    # assoc = torch.empty(n_total_unique_nodes_real, dtype=torch.long, device=device)
    # n_id_old2new = torch.empty(n_total_unique_nodes_real, dtype=torch.long, device=device)

    seen_nodes = []
    main(all_args)
