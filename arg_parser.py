import argparse

parser = argparse.ArgumentParser('Interface for TE-Conv framework')

parser.add_argument('--dataset', type=str, default='wikipedia',
                    help='dataset name')
parser.add_argument('--conv', type=str, default='TransformerConv', help='conv model to use',
                    choices=['TransformerConv', 'ResGatedGraphConv', 'TAGConv'])
parser.add_argument('--layers', type=int, default=1, help='largest number of layers')

parser.add_argument('--bs', type=int, default=200, help='minibatch size')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')

parser.add_argument('--without_raw_msg', default=False, action='store_true',
                    help='not to use raw messages')

# # general model and training setting
# parser.add_argument('--dataset', type=str, default='celegans',
#                     help='dataset name')  # currently relying on dataset to determine task
# parser.add_argument('--test_ratio', type=float, default=0.1, help='ratio of the test against whole')
# parser.add_argument('--model', type=str, default='DE-GNN', help='model to use',
#                     choices=['DE-GNN', 'GIN', 'GCN', 'GraphSAGE', 'GAT'])
# parser.add_argument('--layers', type=int, default=2, help='largest number of layers')
# parser.add_argument('--hidden_features', type=int, default=100, help='hidden dimension')
# parser.add_argument('--metric', type=str, default='auc', help='metric for evaluating performance',
#                     choices=['acc', 'auc'])
# parser.add_argument('--seed', type=int, default=0, help='seed to initialize all the random modules')
# parser.add_argument('--gpu', type=int, default=0, help='gpu id')
# # parser.add_argument('--adj_norm', type=str, default='asym', help='how to normalize adj', choices=['asym', 'sym', 'None'])
# parser.add_argument('--data_usage', type=float, default=1.0, help='use partial dataset')
# parser.add_argument('--directed', type=bool, default=False,
#                     help='(Currently unavailable) whether to treat the graph as directed')
# parser.add_argument('--parallel', default=False, action='store_true',
#                     help='(Currently unavailable) whether to use multi cpu cores to prepare data')
#
# # features and positional encoding
# parser.add_argument('--prop_depth', type=int, default=1, help='propagation depth (number of hops) for one layer')
# parser.add_argument('--use_degree', type=bool, default=True, help='whether to use node degree as the initial feature')
# parser.add_argument('--use_attributes', type=bool, default=False,
#                     help='whether to use node attributes as the initial feature')
# parser.add_argument('--feature', type=str, default='sp',
#                     help='distance encoding category: shortest path or random walk (landing probabilities)')  # sp (shortest path) or rw (random walk)
# parser.add_argument('--rw_depth', type=int, default=3, help='random walk steps')  # for random walk feature
# parser.add_argument('--max_sp', type=int, default=3, help='maximum distance to be encoded for shortest path feature')
#
# # model training
# parser.add_argument('--epoch', type=int, default=1000, help='number of epochs to train')
# parser.add_argument('--bs', type=int, default=64, help='minibatch size')
# parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
# parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer to use')
# parser.add_argument('--l2', type=float, default=0, help='l2 regularization weight')
# parser.add_argument('--dropout', type=float, default=0, help='dropout rate')
#
# # simulation (valid only when dataset == 'simulation')
# parser.add_argument('--k', type=int, default=3, help='node degree (k) or synthetic k-regular graph')
# parser.add_argument('--n', nargs='*', help='a list of number of nodes in each connected k-regular subgraph')
# parser.add_argument('--N', type=int, default=1000, help='total number of nodes in simultation')
# parser.add_argument('--T', type=int, default=6, help='largest number of layers to be tested')
#
# # logging & debug
# parser.add_argument('--log_dir', type=str, default='./log/',
#                     help='log directory')  # sp (shortest path) or rw (random walk)
# parser.add_argument('--summary_file', type=str, default='result_summary.log',
#                     help='brief summary of training result')  # sp (shortest path) or rw (random walk)
# parser.add_argument('--debug', default=False, action='store_true',
#                     help='whether to use debug mode')