import os.path as osp

import torch
from sklearn.metrics import average_precision_score, roc_auc_score


from datasets import JODIEDataset
from models import NodeMemory, IdentityMessage, LastAggregator, MultiHopNeighborLoader,\
    TimeEncoder, EdgeMemory, EdgeIdentityMessage, EdgeLinkPredictor
from utils import get_edge_dataloader
from e2e_model import E2EModel


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'JODIE')
dataset = JODIEDataset(path, name='wikipedia')
data = dataset[0].to(device)

# Helper vector to map global node indices to local ones.
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)

# Ensure to only sample actual destination nodes as negatives.
min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

train_data, val_data, test_data = data.train_val_test_split(
    val_ratio=0.15, test_ratio=0.15)

neighbor_loader = MultiHopNeighborLoader(data.num_nodes, size=10, device=device)

saving_path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'data_loader_3.pt')

print("preparing data ...")
train_dataloader, val_dataloader, test_dataloader, de_dim, num_edges = get_edge_dataloader("wikipedia", data, device, neighbor_loader, assoc)
# train_dataloader, val_dataloader, test_dataloader, de_dim, num_edges = torch.load(saving_path)
print("data prepared")

# Helper vector to map global node indices to local ones in edge graph.
edge_assoc = torch.empty(num_edges + 1, dtype=torch.long, device=device)

# saving_path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'data_loader_3.pt')
# torch.save((train_dataloader, val_dataloader, test_dataloader, de_dim, num_edges), saving_path)



memory_dim = time_dim = embedding_dim = 100

memory = EdgeMemory(
    num_edges + 1,
    data.msg.size(-1),
    memory_dim,
    time_dim,
    message_module=EdgeIdentityMessage(data.msg.size(-1), time_dim),
    aggregator_module=LastAggregator(),
).to(device)

model = E2EModel(
    in_channels=memory_dim + de_dim + time_dim,
    embedding_dim=embedding_dim,
    msg_dim=data.msg.size(-1),
    time_enc=memory.time_enc,
    device=device,
).to(device)

optimizer = torch.optim.Adam(
    set(memory.parameters()) | set(model.parameters()), lr=0.0001)
criterion = torch.nn.BCEWithLogitsLoss()


def train():
    memory.train()
    model.train()
    memory.reset_state()  # Start with a fresh memory.

    total_loss = 0

    for batch in train_dataloader:
        optimizer.zero_grad()
        batch = batch.to(device)
        label = batch.y.unsqueeze(1).float()
        all_n_id = batch.n_id.squeeze(1)
        n_id = batch.n_id.unique()

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)

        edge_assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = z[edge_assoc[all_n_id]], last_update[edge_assoc[all_n_id]]

        prediction = model(batch, z, last_update)
        loss = criterion(prediction, label)

        # Update memory and neighbor loader with ground-truth state.
        mask = batch.y == 1
        e_id, t, msg = batch.edge_id[mask], batch.t[mask], batch.msg.view(-1, data.msg.size(-1))[mask]
        memory.update_state(e_id, t, msg)

        loss.backward()
        optimizer.step()
        memory.detach()
        total_loss += float(loss) * batch.num_graphs
    return total_loss / train_data.num_events


@torch.no_grad()
def test(data_loader):
    memory.eval()
    model.eval()

    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.

    aps, aucs = [], []
    for batch in data_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        label = batch.y.unsqueeze(1).float().cpu()
        n_id = batch.n_id.unique()
        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        edge_assoc[n_id] = torch.arange(n_id.size(0), device=device)

        old_n_id = batch.n_id.squeeze(1)
        # old_n_id = torch.tensor([], dtype=torch.long, device=device)
        # num_nodes = torch.eye(batch.num_graphs)[batch.batch].to(device).sum(dim=0).long()
        # index_base = 0
        # for i in range(len(num_nodes)):
        #     n_id = batch.n_id[index_base: index_base + num_nodes[i]]
        #     old_n_id = torch.cat([old_n_id, n_id.unique()])
        #     index_base += num_nodes[i]

        z, last_update = z[edge_assoc[old_n_id]], last_update[edge_assoc[old_n_id]]
        prediction = model(batch, z, last_update).cpu()

        aps.append(average_precision_score(label, prediction))
        aucs.append(roc_auc_score(label, prediction))

        # Update memory and neighbor loader with ground-truth state.
        mask = batch.y == 1
        e_id, t, msg = batch.edge_id[mask], batch.t[mask], batch.msg.view(-1, data.msg.size(-1))[mask]
        memory.update_state(e_id, t, msg)
    return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean())


for epoch in range(1, 51):
    loss = train()
    print(f'  Epoch: {epoch:02d}, Loss: {loss:.4f}')
    val_ap, val_auc = test(val_dataloader)
    test_ap, test_auc = test(test_dataloader)
    print(f' Val AP: {val_ap:.4f},  Val AUC: {val_auc:.4f}')
    print(f'Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}')
