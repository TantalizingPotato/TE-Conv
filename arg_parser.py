import argparse


def get_args():
    parser = argparse.ArgumentParser('Interface for TE-Conv framework')

    # if MODE == "distance_encoding": USE_NEIGHBOR_FINDER=True rank_score="link_pred"
    # TIME_ENCODING=False EDGE_ATTRIBUTE=False
    parser.add_argument("--mode", type=str, default="tgn",
                        choices=["tgn", "dyrep", "tgat", "jodie", "distance_encoding"],
                        help="the model name, choices=[\"tgn\", \"dyrep\", \"tgat\", \"jodie\", \"distance_encoding\"]")
    parser.add_argument("--use_embedding", type=bool, default=True,
                        help="for tgn only, using gnn embedding or not")
    parser.add_argument("--use_memory_embedding", type=bool, default=False,
                        help="for distance_encoding only, concatenating memory embedding to "
                             "initial node embedding or not")
    parser.add_argument("--name", type=str, default="wikipedia",
                        help="used dataset name")
    parser.add_argument("--rank_score", default="intensity", choices=["likelihood", "intensity", "link_pred"],
                        help="adopted rank_score for temporal recommendation, "
                             "choices=[\"likelihood\", \"intensity\", \"link_pred\"]")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="batch size, number of samples in one batch")
    parser.add_argument("--num_epoch", type=int, default=500,
                        help="number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--backprop_every_n_batches", type=int, default=20,
                        help="if set to n, backpropagation will be done every n batches")
    parser.add_argument("--node_feature_dim", type=int, default=100,
                        help="for distance_encoding only, if de_shortest_path and de_random_walk are both false,"
                             "this argument specifies the dimension number of otherwise adopted zero node feature "
                             "vectors")
    parser.add_argument("--memory_dim", type=int, default=100,
                        help="the dimension number of memory vectors, effective only if use_memory is true")
    parser.add_argument("--time_dim", type=int, default=100,
                        help="the dimension number of time encoding")
    parser.add_argument("--embedding_dim", type=int, default=100,
                        help="the dimension number of finally obtained node embedding")
    parser.add_argument("--time_encoding", type=bool, default=True,
                        help="whether to use time encoding")
    parser.add_argument("--distance_encoding", type=bool, default=False,
                        help="")
    parser.add_argument("--de_shortest_path", type=bool, default=True,
                        help="for distance_encoding only, whether to use shortest_path")
    parser.add_argument("--de_random_walk = True", type=bool, default=True,
                        help="for distance encoding only, whether to use random_walk")
    parser.add_argument("--edge_attribute", type=bool, default=False,
                        help="whether to use edge attribute when generating node embedding")
    parser.add_argument("--use_msg", type=bool, default=False,
                        help="whether to load message (edge attribute) from file")
    parser.add_argument("--n_layer", type=int, default=1,
                        help="number of layers in the GNN model")

    parser.add_argument("--num_neg_sample", type=int, default=10,
                        help="number of negative samples (that correspond to one positive sample)")
    parser.add_argument("--n_coefficient", type=int, default=2,
                        help="loss weight for negative sample loss")

    parser.add_argument("--dyrep_loss", type=bool, default=True,
                        help="whether to adopt dyrep-style loss function")
    parser.add_argument("--use_dyrepabstractsum", type=bool, default=True,
                        help="whether to adopt dyrep-style time predictor")

    parser.add_argument("--num_basis", type=int, default=64,
                        help="number of basis in time predictor")
    parser.add_argument("--mc_num_surv", type=int, default=1,
                        help="monte carlo sample number for survival term in time predictor")
    parser.add_argument("--mc_num_time", type=int, default=100,
                        help="monte carlo sample number for computing some integral expression in time predictor")

    parser.add_argument("--use_neighbor_finder = True", type=bool, default=True,
                        help="whether to use neighbor_finder (if set to True) or neighbor_loader (if set to False)")

    parser.add_argument("--use_neg", type=bool, default=False,
                        help="whether to insert negative samples into batches")
    parser.add_argument("--sample_neg_dst_strictly", type=bool, default=True,
                        help="whether to sample negative dst nodes strictly (avoiding coincident positive samples)")
    parser.add_argument("--can_see_neighbors_in_same_batch", type=bool, default=True,
                        help="whether the model can see a node's temporal neighbors that is in the same batch with it")
    parser.add_argument("--only_predict_last_events_time", type=bool, default=True,
                        help="whether only to predict the last time of the same events in a batch")

    parser.add_argument("--use_memory", type=bool, default=True,
                        help="whether to use memory vectors")

    parser.add_argument("--neighbor_size", type=int, default=10,
                        help="the number of a node's neighbors returned by neighbor_finder or neighbor_loader")

    parser.add_argument("--dataset_size = 2000", type=int, default=2000,
                        help="the number of samples to draw from the dataset")

    parser.add_argument("--cpu_only = True", type=bool, default=True,
                        help="whether to run the model only on CPU (to run without GPU)")

    parser.add_argument("--candidates_from", type=str, default="all_dst", choices=["all_dst", "all_nodes", "neighbors"],
                        help="where to draw temporal recommendation candidates from, "
                             "choices=[\"all_dst\", \"all_nodes\", \"neighbors\"]")

    parser.add_argument("--task", type=str, default="time_prediction",
                        choices=["link_prediction", "time_prediction", "temporal_recommendation"],
                        help="the task, choices=[\"link_prediction\", \"time_prediction\","
                             " \"temporal_recommendation\"]")

    args = parser.parse_args()

    return args
