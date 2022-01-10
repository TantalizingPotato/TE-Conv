# lgh code fragments:

# ### from main.py

# 1.
# sample_mask = torch.ones((src.size(0) * N_SAMPLE,), device=device).bool()
# if DYREP_LOSS:
#     neg_src = torch.empty((src.size(0) * N_SAMPLE,), device=device, dtype=src.dtype)
#     while sample_mask.any():
#         neg_dst_index = torch.randint(0, train_dst.size(0), (sample_mask.sum(),),
#                                       dtype=torch.long, device=device)
#         neg_src[sample_mask] = train_dst[neg_dst_index]
#         sample_mask = torch.logical_or(neg_src == dst.repeat_interleave(N_SAMPLE),
#                                        neg_src == src.repeat_interleave(N_SAMPLE))
#         # sample_mask = neg_src == dst.repeat_interleave(N_SAMPLE) | neg_src == src.repeat_interleave(
#         #     N_SAMPLE)
# if DYREP_LOSS:
#     neg_src_index = torch.randint(0, train_nodes.size(0), (src.size(0),),
#                                   dtype=torch.long, device=device)
#     neg_src = train_nodes[neg_src_index]


# 2.
# Get updated memory of all nodes involved in the computation.
# if MODE == "dyrep":
#     z = memory(n_id_unique)[0]
#     z = z[assoc_n_id[n_id]]
# z = memory.memory[n_id]
# emb = gnn(z, saved_edge_index, saved_t[saved_edge_index[1]] - full_t[saved_e_id],full_msg[saved_e_id])
# memory.update_state(src, dst, t, msg, dst_emb=emb[src.size(0):2 * src.size(0)])


# 3.
# if DYREP_LOSS:
#     last_update_src, last_update_dst = last_update[:num_src], last_update[num_src:num_pos]
#     pos_delta_t = (t - torch.max(last_update_src, last_update_dst)).float()
#     # last_update_neg_src, last_update_neg_dst = last_update[2 * num_src: (N_SAMPLE + 2) * num_src], \
#     #                                            last_update[(N_SAMPLE + 2) * num_src:(2*N_SAMPLE + 2) * src.size(0)]
#     # neg_src is in fact equal to pos_src
#     last_update_neg_src, last_update_neg_dst = \
#         last_update_src.repeat_interleave(NUM_NEG_SAMPLE, dim=0), last_update[num_pos:neg_split]
#
#     delta_t_neg = (t.repeat_interleave(NUM_NEG_SAMPLE) - torch.max(last_update_neg_src,
#                                                                    last_update_neg_dst)).float()
#     # loss_cur_batch = -abstract_sum.ll(z_pos_src, z_pos_dst, pos_delta_t).sum() / z_pos_src.size(0)
#     # loss_cur_batch = - abstract_sum.intensity_func(pos_delta_t, z_pos_src, z_pos_dst,
#     #                                                return_params=False).sum() / z_pos_src.size(0)
#     loss_cur_batch = - torch.log(abstract_sum.intensity_func(pos_delta_t, z_pos_src, z_pos_dst,
#                                                              return_params=False)).sum() / z_pos_src.size(0)
#     # loss_cur_batch += abstract_sum.compensator_func(delta_t_neg, z[2 * src.size(0): (N_SAMPLE + 2) * src.size(0)],
#     #     z[(N_SAMPLE + 2) * src.size(0):(2*N_SAMPLE + 2) * src.size(0)]).sum() / N_SAMPLE
#     # loss_cur_batch += abstract_sum.compensator_func(delta_t_neg, z[:src.size(0)].repeat_interleave(N_SAMPLE, dim=0),
#     #     z[2 * src.size(0): (N_SAMPLE + 2) * src.size(0)]).sum() / N_SAMPLE
#     loss_cur_batch += N_COEFFICIENT * abstract_sum.intensity_func(delta_t_neg, z_neg_src, z_neg_dst,
#                                                                   return_params=False).sum() / z_neg_src.size(0)
#     # loss_cur_batch +=  - N_COEFFICIENT * torch.log(1 - abstract_sum.intensity_func(delta_t_neg,
#     #     z_neg_src, z_neg_dst, return_params=False)).sum() / z_neg_src.size(0)
#     # loss_cur_batch +=  N_COEFFICIENT * abstract_sum.compensator_func(delta_t_neg,
#     #     z_neg_src, z_neg_dst).sum() / z_neg_src.size(0)


# 4.
# if DISTANCE_ENCODING: pass
# z = memory(n_id)[0] if USE_MEMORY else torch.zeros(len(n_id), MEMORY_DIM)


# 5.
# n_id = torch.cat([src, pos_dst, neg_dst]).unique()
# n_id = torch.cat([src, dst, neg_dst])
# num_focused_node = n_id.size(0)
# if CAN_SEE_NEIGHBORS_IN_SAME_BATCH:
#     timestamps = t.repeat(2 + NUM_NEG_SAMPLE).cpu().numpy()
# else:
#     timestamps = np.array([t[0].item()]).repeat(num_focused_node)
# n_id, edge_index, e_id, root_id_in_batch = neighbor_finder.get_k_hop_temporal_neighbor(n_id.cpu().numpy(),
#                                                                                        timestamps,
#                                                                                        n_neighbors=NEIGHBOR_SIZE,
#                                                                                        k=1)
#
# n_id, edge_index, e_id, root_id_in_batch = n_id.to(device), edge_index.to(device), e_id.to(
#     device), root_id_in_batch.to(device)
#
# n_id_unique = n_id.unique()
# n_id_old2new[n_id_unique] = torch.arange(n_id_unique.size(0), device=device)
#
# if USE_MEMORY:
#     z, last_update = memory(n_id_unique)
#     z, last_update = z[n_id_old2new[n_id]], last_update[n_id_old2new[n_id]]
# else:
#     z = torch.zeros(len(n_id), MEMORY_DIM, device=device)
#
# # z = memory(n_id)[0] if USE_MEMORY else torch.zeros(len(n_id), MEMORY_DIM)
# # emb = gnn(z, edge_index, full_t[e_id], full_msg[e_id])
#
# # emb = gnn(z, edge_index, t.repeat(num_focused_node / src.size(0))[root_id_in_batch] - full_t[e_id], full_msg[e_id])
#
# # if MODE == "tgat" or MODE == "tgn": z = emb
#
# if MODE == "jodie":
#     time_diff = t.repeat(int(num_focused_node / src.size(0))) - last_update[:num_focused_node]
#     if num_focused_node / src.size(0) == 2 + NUM_NEG_SAMPLE:
#         time_diff[:src.size(0)] = (time_diff[:src.size(0)] - mean_time_shift_src) / std_time_shift_src
#         time_diff[src.size(0):] = (time_diff[src.size(0):] - mean_time_shift_dst) / std_time_shift_dst
#     elif num_focused_node / src.size(0) == 2 + 2 * NUM_NEG_SAMPLE:
#         time_diff[:src.size(0)] = (time_diff[:src.size(0)] - mean_time_shift_src) / std_time_shift_src
#         time_diff[src.size(0):2 * src.size(0)] = (time_diff[src.size(0):2 * src.size(
#             0)] - mean_time_shift_dst) / std_time_shift_dst
#         time_diff[2 * src.size(0):(2 + NUM_NEG_SAMPLE) * src.size(0)] = \
#             (time_diff[2 * src.size(0):(2 + NUM_NEG_SAMPLE) * src.size(0)] - mean_time_shift_src) / std_time_shift_src
#         time_diff[(2 + NUM_NEG_SAMPLE) * src.size(0):(2 + 2 * NUM_NEG_SAMPLE) * src.size(0)] = \
#             (time_diff[(2 + NUM_NEG_SAMPLE) * src.size(0):(2 + 2 * NUM_NEG_SAMPLE) * src.size(0)] -
#              mean_time_shift_dst) / std_time_shift_dst
#     else:
#         raise NotImplementedError
#     z[:num_focused_node] = gnn(z[:num_focused_node], time_diff)
# elif MODE == "tgat" or (MODE == "tgn" and USE_EMBEDDING):
#     z = gnn(z, edge_index, t.repeat(int(num_focused_node / src.size(0)))[root_id_in_batch] - full_t[e_id],
#             full_msg[e_id])
# elif MODE == "dyrep":
#     emb = gnn(z, edge_index, t.repeat(int(num_focused_node / src.size(0)))[root_id_in_batch] - full_t[e_id],
#               full_msg[e_id])
# # else:
# #     emb = gnn(z, edge_index, t.repeat(int(num_focused_node / src.size(0)))[root_id_in_batch] - full_t[e_id],
# #               full_msg[e_id])
# #
# # if MODE == "tgat" or MODE == "tgn": z = emb
#
# # if MODE == "jodie":
# #     z[:num_focused_node] = (1 + gnn.time_enc(t.repeat(num_focused_node / src.size(0)) -
# #                                              last_update[:num_focused_node])) * z[:num_focused_node]
# # pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])
# # neg_out = link_pred(z[assoc[src]], z[assoc[neg_dst]])


# 6.
# y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
# y_true = torch.cat(
#     [torch.ones(pos_out.size(0)),
#      torch.zeros(neg_out.size(0))], dim=0)
#
# aps.append(average_precision_score(y_true, y_pred))
# aucs.append(roc_auc_score(y_true, y_pred))


# 7.
# # Update memory and neighbor loader with ground-truth state.
# update_valid_mask = cur_e_id != -1
# src, dst, t, cur_e_id = src[update_valid_mask], dst[update_valid_mask], \
#                         t[update_valid_mask], cur_e_id[update_valid_mask]
# if USE_MEMORY:
#     msg = full_msg[cur_e_id]
#     if MODE == "dyrep":
#         memory.update_state(src, dst, t, msg, src_emb=dyrep_emb[:src.size(0)],
#                             dst_emb=dyrep_emb[src.size(0):2 * src.size(0)])
#     else:
#         memory.update_state(src, dst, t, msg)

# 8.
# elif TASK == "link_prediction":
#     val_ap, val_auc, val_pos_acc, val_neg_acc, \
#     val_old_ap, val_old_auc, val_old_pos_acc, val_old_neg_acc, \
#     val_nn_ap, val_nn_auc, val_nn_pos_acc, val_nn_neg_acc, \
#     val_nn_one_ap, val_nn_one_auc, val_nn_one_pos_acc, val_nn_one_neg_acc, \
#     val_nn_two_ap, val_nn_two_auc, val_nn_two_pos_acc, val_nn_two_neg_acc, \
#     _ = test(val_data, val_old_mask, nn_val_mask, nn_one_val_mask, nn_two_val_mask)
#
#     test_ap, test_auc, test_pos_acc, test_neg_acc, \
#     test_old_ap, test_old_auc, test_old_pos_acc, test_old_neg_acc, \
#     test_nn_ap, test_nn_auc, test_nn_pos_acc, test_nn_neg_acc, \
#     test_nn_one_ap, test_nn_one_auc, test_nn_one_pos_acc, test_nn_one_neg_acc, \
#     test_nn_two_ap, test_nn_two_auc, test_nn_two_pos_acc, test_nn_two_neg_acc, \
#     test_temporal_neg_acc = test(test_data, test_old_mask, nn_test_mask, nn_one_test_mask, nn_two_test_mask,
#                                  neg_test_mask)
#
#     # val_ap, val_auc, val_pos_acc, val_neg_acc = test(val_dataloader)
#     # test_ap, test_auc, test_pos_acc, test_neg_acc = test(test_dataloader)
#     print(f" Val AP: {val_ap:.4f},  Val AUC: {val_auc:.4f}, {val_pos_acc:.4f}, {val_neg_acc:.4f}")
#     print(f"Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}, {test_pos_acc:.4f}, {test_neg_acc:.4f}")
#
#    print(f" Val_old AP: {val_old_ap:.4f},  Val_old AUC: {val_old_auc:.4f}, "
#          f"{val_old_pos_acc:.4f}, {val_old_neg_acc:.4f}")
#    print(f" Test_old AP: {test_old_ap:.4f}, Test_old AUC: {test_old_auc:.4f}, "
#          f"{test_old_pos_acc:.4f}, {test_old_neg_acc:.4f}")
#
#     # val_nn_ap, val_nn_auc, val_nn_pos_acc, val_nn_neg_acc = test(val_nn_dataloader)
#     # test_nn_ap, test_nn_auc, test_nn_pos_acc, test_nn_neg_acc = test(test_nn_dataloader)
#     print(f" Val_NN AP: {val_nn_ap:.4f},  Val_NN AUC: {val_nn_auc:.4f}, {val_nn_pos_acc:.4f}, {val_nn_neg_acc:.4f}")
#     print(
#         f"Test_NN AP: {test_nn_ap:.4f}, Test_NN AUC: {test_nn_auc:.4f}, {test_nn_pos_acc:.4f}, {test_nn_neg_acc:.4f}")
#
#     # val_nn_one_ap, val_nn_one_auc, val_nn_one_pos_acc, val_nn_one_neg_acc = test(val_nn_one_dataloader)
#     # test_nn_one_ap, test_nn_one_auc, test_nn_one_pos_acc, test_nn_one_neg_acc = test(test_nn_one_dataloader)
#     print(f" Val_NN_1 AP: {val_nn_one_ap:.4f},  Val_NN_1 AUC: {val_nn_one_auc:.4f}, "
#           f"{val_nn_one_pos_acc:.4f}, {val_nn_one_neg_acc:.4f}")
#     print(f"Test_NN_1 AP: {test_nn_one_ap:.4f}, Test_NN_1 AUC: {test_nn_one_auc:.4f}, "
#           f"{test_nn_one_pos_acc:.4f}, {test_nn_one_neg_acc:.4f}")
#
#     # val_nn_two_ap, val_nn_two_auc, val_nn_two_pos_acc, val_nn_two_neg_acc = test(val_nn_two_dataloader)
#     # test_nn_two_ap, test_nn_two_auc, test_nn_two_pos_acc, test_nn_two_neg_acc = test(test_nn_two_dataloader)
#     print(f" Val_NN_2 AP: {val_nn_two_ap:.4f},  Val_NN_2 AUC: {val_nn_two_auc:.4f}, "
#           f"{val_nn_two_pos_acc:.4f}, {val_nn_two_neg_acc:.4f}")
#     print(f"Test_NN_2 AP: {test_nn_two_ap:.4f}, Test_NN_2 AUC: {test_nn_two_auc:.4f}, "
#           f"{test_nn_two_pos_acc:.4f}, {test_nn_two_neg_acc:.4f}")
#
#     # test_neg_accuracy = test_neg(test_neg_dataloader)
#     # print(f" Test_neg AP: {test_neg_ap:.4f}")
#     print(f" Test_neg ACC: {test_temporal_neg_acc:.4f}")


# 9.
# ap, auc, pos_acc, neg_acc = average_precision_score(all_labels[valid_mask], all_predictions[valid_mask]), \
#                             roc_auc_score(all_labels[valid_mask], all_predictions[valid_mask]), \
#                             (all_predictions[valid_mask][all_labels[valid_mask].bool()] >= 0.5).float().mean(), \
#                             (all_predictions[valid_mask][~all_labels[valid_mask].bool()] < 0.5).float().mean()
# old_ap, old_auc, old_pos_acc, old_neg_acc = average_precision_score(all_labels[old_mask],
#                                                                     all_predictions[old_mask]), \
#                                             roc_auc_score(all_labels[old_mask], all_predictions[old_mask]), \
#                                             (all_predictions[old_mask][
#                                                  all_labels[old_mask].bool()] >= 0.5).float().mean(), \
#                                             (all_predictions[old_mask][
#                                                  ~all_labels[old_mask].bool()] < 0.5).float().mean()
# nn_ap, nn_auc, nn_pos_acc, nn_neg_acc = average_precision_score(all_labels[nn_mask], all_predictions[nn_mask]), \
#                                         roc_auc_score(all_labels[nn_mask], all_predictions[nn_mask]), \
#                                         (all_predictions[nn_mask][
#                                              all_labels[nn_mask].bool()] >= 0.5).float().mean(), \
#                                         (all_predictions[nn_mask][~all_labels[nn_mask].bool()] < 0.5).float().mean()
# nn_one_ap, nn_one_auc, nn_one_pos_acc, nn_one_neg_acc = average_precision_score(all_labels[nn_one_mask],
#                                                                                 all_predictions[nn_one_mask]), \
#                                                         roc_auc_score(all_labels[nn_one_mask],
#                                                                       all_predictions[nn_one_mask]), \
#                                                         (all_predictions[nn_one_mask][
#                                                              all_labels[nn_one_mask].bool()] >= 0.5).float().mean(), \
#                                                         (all_predictions[nn_one_mask][
#                                                              ~all_labels[nn_one_mask].bool()] < 0.5).float().mean()
# average_precision_score(all_labels[nn_two_mask],
#                                                                                 all_predictions[nn_two_mask]), \
#                                                         roc_auc_score(all_labels[nn_two_mask],
#                                                                       all_predictions[nn_two_mask]), \
#                                                         (all_predictions[nn_two_mask][
#                                                              all_labels[nn_two_mask].bool()] >= 0.5).float().mean(), \
#                                                         (all_predictions[nn_two_mask][
#                                                              ~all_labels[nn_two_mask].bool()] < 0.5).float().mean()


# 10.
# if USE_MEMORY:
#     memory.eval()
#     memory.reset_state()


# 11.
# if USE_NEIGHBOR_FINDER:
#     neighbor_finder = get_neighbor_finder(full_pos_data)
# if USE_NEIGHBOR_FINDER: neighbor_finder = get_neighbor_finder(full_pos_data)


# 12.
# @torch.no_grad()
# def test_time_prediction_bak(inference_data):
#     if not USE_MEMORY:
#         raise NotImplementedError("Time prediction depends on node memory")
#
#     neighbor_finder = get_neighbor_finder(full_pos_data)
#     memory.eval()
#     # gnn.eval()
#     # link_pred.eval()
#     abstract_sum.eval()
#
#     torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.
#
#     aes = []
#     for batch in inference_data.seq_batches(batch_size=1):
#         src, pos_dst, t, cur_e_id = batch.src, batch.dst, batch.t, batch.e_id
#         valid_mask = cur_e_id != -1
#         src, pos_dst, t, cur_i_id = src[valid_mask], pos_dst[valid_mask], t[valid_mask], cur_e_id[valid_mask]
#
#         if src.size(0) == 0:
#             print("No pos_sample in this batch, continue")
#             continue
#
#         if ONLY_PREDICT_LAST_EVENTS_TIME:
#             counted_nodes = set()
#             mask = src.new_zeros(src.shape).bool()
#             for i in range(1, src.size(0) + 1):
#                 mask[-i] = src[-i].item() not in counted_nodes and pos_dst[-i].item() not in counted_nodes
#                 counted_nodes = counted_nodes | {src[-i].item(), pos_dst[-i].item()}
#             src, pos_dst, t, cur_e_id = src[mask], pos_dst[mask], t[mask], cur_e_id[mask]
#
#         n_id = torch.cat([src, pos_dst]).unique()
#         assoc[n_id] = torch.arange(n_id.size(0), device=device)
#
#         # # Get updated memory of all nodes involved in the computation.
#         z, last_update = memory(n_id)
#
#         # print(f"src:{src}, dst:{pos_dst}, z: {z}")
#
#         src_dst = torch.stack([src, pos_dst]).t()
#         t_true = torch.min(t.unsqueeze(-1) - last_update[assoc[src_dst]], dim=-1)[0].float().cpu()
#         seen_mask = t_true != t
#         seen_src, seen_pos_dst, seen_t, seen_cur_e_id, seen_t_true = src[seen_mask], pos_dst[seen_mask],\
#                                                                      t[seen_mask], \
#                                                                      cur_e_id[seen_mask], t_true[seen_mask]
#         if seen_src.size(0) == 0:
#             # print("all nodes unseen in this batch, continue to the next batch")
#             pass
#         else:
#             # print("There are seen nodes")
#             # t_true[t_true > 5] = 5.
#
#             t_estimated = abstract_sum.estimate_time(z[assoc[seen_src]], z[assoc[seen_pos_dst]]).cpu()
#             # t_estimated[t_estimated < 1] = 0.
#             # print(t_estimated)
#
#             absolute_error = valid_mask.sum() * mean_absolute_error(seen_t_true, t_estimated)
#             # print("i")
#             aes.append(absolute_error)
#
#         # Update memory and neighbor loader with ground-truth state.
#         if MODE == "dyrep":
#             n_id, edge_index, e_id, root_id_in_batch = \
#                 neighbor_finder.get_k_hop_temporal_neighbor(n_id.cpu().numpy(),
#                                                             t.repeat(2), n_neighbors=NEIGHBOR_SIZE, k=1)
#             emb = gnn(z, edge_index, t.repeat(2)[root_id_in_batch] - full_t[e_id],
#                       full_msg[e_id])
#             memory.update_state(src, pos_dst, t, full_msg[cur_e_id], emb[:src.size(0)], emb[src.size(0):])
#         else:
#             memory.update_state(src, pos_dst, t, full_msg[cur_e_id])
#     return float(torch.tensor(aes).mean())


# 13.
# if seen_z_src.size(0) == 0:
#     print(f"e_id: {cur_e_id}, src:{src}, dst:{dst}")
#     print("all nodes unseen in this batch, continue to the next batch")
#     pass
# else:
#     seen_t_true = seen_t_true.cpu().numpy()
#     t_estimated = np.clip(
#         abstract_sum.estimate_time(z_src[seen_mask], z_dst[seen_mask]).cpu().numpy() * 5000, -np.inf, 100000)
#     # t_estimated[t_estimated < 1] = 0.
#     print(
#         f"e_id: {seen_cur_e_id}, src:{seen_src}, dst:{seen_pos_dst}, t_estimated: {t_estimated}, t_true: {t_true}")
#
#     # absolute_error = len(seen_t_true) * mean_absolute_error(seen_t_true, t_estimated)
#     print(f"absolute_error: {abs(seen_t_true - t_estimated)}")
#     # print("i")
#     # aes.append(absolute_error)
#
#     all_t_true, all_t_estimated = np.concatenate([all_t_true, seen_t_true]), np.concatenate(
#         [all_t_estimated, t_estimated])


# 14.
# # in def test_temporal_recommendation
# batch = batch.to(device)
# mask = batch.y == 1
# label = batch.y.unsqueeze(1).float().cpu()
# n_id = batch.n_id.unique()
# src, dst, t, msg =
# batch.src[mask], batch.dst[mask], batch.t[mask], batch.msg.view(len(batch.y), full_msg.size(-1))[mask]
# src, dst, t = src[mask], dst[mask], t[mask]


# 15.
# Update memory and neighbor loader with ground-truth state.
# src, dst, t, msg = batch.src[mask], batch.dst[mask], batch.t[mask], batch.msg.view(len(batch.y), raw_msg_dim)[mask]
# print(src.shape)
# n_id = torch.cat([src, dst]).unique()
# memory.update_state(src, dst, t, msg)


# 16.
# torch.autograd.set_detect_anomaly(True)


# 17.
# print(f"val_mar: {val_mar}, val_miss_times: {val_miss_record.sum()}, "
#       f"val_miss_rate: {val_miss_record.float().mean()}")
# print(f"val_old_mar: {val_old_mar}, val_old_miss_times: {val_old_miss_record.sum()}, "
#       f"val_old_miss_rate: {val_old_miss_record.float().mean()}")
# print(f"val_nn_mar: {val_mar}, val_nn_miss_times: {val_nn_miss_record.sum()}, "
#       f"val_nn_miss_rate: {val_nn_miss_record.float().mean()}")
# print(f"val_nn_one_mar: {val_nn_one_mar}, val_nn_one_miss_times: {val_nn_one_miss_record.sum()}, "
#       f"val_nn_one_miss_rate: {val_nn_one_miss_record.float().mean()}")
# print(f"val_nn_two_mar: {val_mar}, val_nn_two_miss_times: {val_nn_two_miss_record.sum()}, "
#       f"val_nn_two_miss_rate: {val_nn_two_miss_record.float().mean()}")
# print(f"test_mar: {test_mar}, test_miss_times: {test_miss_record.sum()}, "
#       f"test_miss_rate: {test_miss_record.float().mean()}")
# print(f"test_old_mar: {test_old_mar}, test_old_miss_times: {test_old_miss_record.sum()}, "
#       f"test_old_miss_rate: {test_old_miss_record.float().mean()}")
# print(f"test_nn_mar: {test_nn_mar}, test_nn_miss_times: {test_nn_miss_record.sum()}, "
#       f"test_nn_miss_rate: {test_nn_miss_record.float().mean()}")
# print(f"test_nn_one_mar: {test_nn_one_mar}, test_nn_one_miss_times: {test_nn_one_miss_record.sum()}, "
#       f"test_nn_one_miss_rate: {test_nn_one_miss_record.float().mean()}")
# print(f"test_nn_two_mar: {test_nn_two_mar}, test_nn_two_miss_times: {test_nn_two_miss_record.sum()}, "
#       f"test_nn_two_miss_rate: {test_nn_two_miss_record.float().mean()}")


# 18.
# val_ap, val_auc, val_pos_acc, val_neg_acc = test(val_dataloader)
# test_ap, test_auc, test_pos_acc, test_neg_acc = test(test_dataloader)
# print(f"{val_ap=:.4f}, {val_auc=:.4f}, {val_pos_acc=:.4f}, {val_neg_acc=:.4f}")
# print(f"{test_ap=:.4f}, {test_auc=:.4f}, {test_pos_acc=:.4f}, {test_neg_acc=:.4f}")
#
# print(
#     f" Val_old AP: {val_old_ap:.4f},  Val_old AUC: {val_old_auc:.4f}, " \
#     f"{val_old_pos_acc:.4f}, {val_old_neg_acc:.4f}")
# print(
#     f"Test_old AP: {test_old_ap:.4f}, Test_old AUC: {test_old_auc:.4f}, " \
#     f"{test_old_pos_acc:.4f}, {test_old_neg_acc:.4f}")
#
# # val_nn_ap, val_nn_auc, val_nn_pos_acc, val_nn_neg_acc = test(val_nn_dataloader)
# # test_nn_ap, test_nn_auc, test_nn_pos_acc, test_nn_neg_acc = test(test_nn_dataloader)
# print(f" Val_NN AP: {val_nn_ap:.4f},  Val_NN AUC: {val_nn_auc:.4f}, {val_nn_pos_acc:.4f}, {val_nn_neg_acc:.4f}")
# print(
#     f"Test_NN AP: {test_nn_ap:.4f}, Test_NN AUC: {test_nn_auc:.4f}, {test_nn_pos_acc:.4f}, {test_nn_neg_acc:.4f}")
#
# # val_nn_one_ap, val_nn_one_auc, val_nn_one_pos_acc, val_nn_one_neg_acc = test(val_nn_one_dataloader)
# # test_nn_one_ap, test_nn_one_auc, test_nn_one_pos_acc, test_nn_one_neg_acc = test(test_nn_one_dataloader)
# print(
#     f" Val_NN_1 AP: {val_nn_one_ap:.4f},  Val_NN_1 AUC: {val_nn_one_auc:.4f}, " \
#     f"{val_nn_one_pos_acc:.4f}, {val_nn_one_neg_acc:.4f}")
# print(
#     f"Test_NN_1 AP: {test_nn_one_ap:.4f}, Test_NN_1 AUC: {test_nn_one_auc:.4f}, " \
#     f"{test_nn_one_pos_acc:.4f}, {test_nn_one_neg_acc:.4f}")
#
# # val_nn_two_ap, val_nn_two_auc, val_nn_two_pos_acc, val_nn_two_neg_acc = test(val_nn_two_dataloader)
# # test_nn_two_ap, test_nn_two_auc, test_nn_two_pos_acc, test_nn_two_neg_acc = test(test_nn_two_dataloader)
# print(
#     f" Val_NN_2 AP: {val_nn_two_ap:.4f},  Val_NN_2 AUC: {val_nn_two_auc:.4f},
#     {val_nn_two_pos_acc:.4f}, {val_nn_two_neg_acc:.4f}")
# print(
#     f"Test_NN_2 AP: {test_nn_two_ap:.4f}, Test_NN_2 AUC: {test_nn_two_auc:.4f},
#     {test_nn_two_pos_acc:.4f}, {test_nn_two_neg_acc:.4f}")
#
# # test_neg_accuracy = test_neg(test_neg_dataloader)
# # print(f" Test_neg AP: {test_neg_ap:.4f}")


# 19.
# def print_vals(*args):
#     line = ", ".join([f"{arg=:.4f}" for arg in args])
#     print(line)


# 20.
# def get_candidates_from(nodes_th):
#     candidates_inner = nodes_th.repeat(src.size(0))
#     root_nodes_inner = src.repeat_interleave(full_dst.size(0))
#     root_ids_inner = torch.arange(src.size(0)).repeat_interleave(full_dst.size(0))
#     recommend_times_inner = t.repeat_interleave(full_dst.size(0))
#     valid_mask = root_nodes_inner != candidates_inner
#     candidates_inner, root_nodes_inner, root_ids_inner, recommend_times_inner = candidates_inner[valid_mask], \
#                                                                                 root_nodes_inner[valid_mask], \
#                                                                                 root_ids_inner[valid_mask], \
#                                                                                 recommend_times_inner[valid_mask]
#     return candidates_inner, root_nodes_inner, root_ids_inner, recommend_times_inner


# 21.
# def prepare_data():
#     # global full_pos_data, train_data, val_data, test_data, full_msg, full_t, val_old_mask, test_old_mask, nn_val_mask, nn_test_mask, nn_one_val_mask, nn_one_test_mask, nn_two_val_mask, nn_two_test_mask, neg_test_mask, n_total_unique_nodes_real, train_dst, train_nodes, full_dst, full_nodes, mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst
#     print("preparing data ...")
#     full_pos_data, train_data, val_data, test_data, full_msg, full_t, \
#         val_old_mask, test_old_mask, nn_val_mask, nn_test_mask, nn_one_val_mask, \
#         nn_one_test_mask, nn_two_val_mask, nn_two_test_mask, neg_test_mask, \
#         n_total_unique_nodes_real, num_train_events, min_dst_idx, max_dst_idx \
#         = get_data(NAME, use_neg=USE_NEG, use_msg=USE_MSG, dataset_size=DATASET_SIZE, cpu_only=CPU_ONLY, nn_ratio=0)
#     train_src = torch.unique(train_data.src, sorted=True)
#     train_dst = torch.unique(train_data.dst, sorted=True)
#     train_nodes = torch.unique(torch.cat([train_src, train_dst]), sorted=True)
#     full_src = torch.unique(full_pos_data.src, sorted=True)
#     full_dst = torch.unique(full_pos_data.dst, sorted=True)
#     full_nodes = torch.unique(torch.cat([full_src, full_dst]), sorted=True)
#     mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst, mean_time_shift, std_time_shift \
#         = compute_time_statistics(full_pos_data.src, full_pos_data.dst, full_pos_data.t)
#     print("data prepared")


# 22.
# def prepare_models():
#     # global memory, gnn, link_predictor, abstract_sum, optimizer, criterion
#     if USE_MEMORY:
#         memory = TGNMemory(
#             n_total_unique_nodes_real,
#             full_msg.size(-1),
#             MEMORY_DIM,
#             EMBEDDING_DIM,
#             TIME_DIM,
#             message_module=IdentityMessage(full_msg.size(-1), TIME_DIM, MEMORY_DIM, EMBEDDING_DIM, mode=MODE),
#             aggregator_module=LastAggregator(),
#             mode=MODE
#         ).to(device)
#     else:
#         memory = None
#     arg_dict = {"num_nodes": n_total_unique_nodes_real,
#                 "in_channels": MEMORY_DIM if MODE != "distance_encoding" else 5 * DE_SHORTEST_PATH + 4 * DE_RANDOM_WALK,
#                 "out_channels": EMBEDDING_DIM,
#                 "msg_dim": full_msg.size(-1),
#                 "time_enc": memory.time_enc if memory is not None else TimeEncoder(out_channels=TIME_DIM, mode=MODE).to(
#                     device),
#                 "n_layer": N_LAYER}
#     if MODE == "jodie":
#         gnn = TimeEmbedding(**arg_dict).to(device)
#     elif MODE == "distance_encoding":
#         gnn = GraphAttentionEmbedding(**arg_dict).to(device)
#     else:
#         gnn = GraphAttentionEmbedding(**arg_dict).to(device)
#     link_predictor = LinkPredictor(in_channels=EMBEDDING_DIM).to(device)
#     if not USE_DYREPABSTRACTSUM:
#         # in_channels, num_basis, num_param, mc_num_surv = 1, mc_num_time = 3
#         abstract_sum = AbstractSum(2 * EMBEDDING_DIM, num_basis=NUM_BASIS, mc_num_surv=MC_NUM_SURV,
#                                    mc_num_time=MC_NUM_TIME)
#     else:
#         # in_channels, num_basis, num_param, mc_num_surv = 1, mc_num_time = 3
#         abstract_sum = DyrepAbstractSum(2 * EMBEDDING_DIM, mc_num_surv=MC_NUM_SURV, mc_num_time=MC_NUM_TIME)
#     if USE_MEMORY:
#         memory_parameter_set = set(memory.parameters())
#     else:
#         memory_parameter_set = set()
#     all_parameters = memory_parameter_set | set(gnn.parameters()) | set(abstract_sum.parameters()) | set(
#         link_predictor.parameters())
#     optimizer = torch.optim.Adam(all_parameters, lr=LEARNING_RATE)
#     criterion = torch.nn.BCEWithLogitsLoss()
