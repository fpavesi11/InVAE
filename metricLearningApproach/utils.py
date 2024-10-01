import torch
from torch import nn

class reshape_to_matrix(nn.Module):
    def __init__(self):
        super(reshape_to_matrix, self).__init__()

    def forward(self, x):
        mat_dim = int(torch.sqrt(torch.tensor(x.size(1))))
        return x.view(x.size(0), mat_dim, mat_dim)


class mask_out(nn.Module):
    def __init__(self):
        super(mask_out, self).__init__()

    def forward(self, x):
        mask = torch.tril(torch.ones(x.size(1), x.size(2)), diagonal=-1).to(x.device)
        return x * mask
    
def make_mlp(in_channels, hidden_channels, out_channels, depth, activation):
    layers = []
    layers.append(nn.Linear(in_channels, hidden_channels))
    layers.append(activation)
    for _ in range(depth - 2):
        layers.append(nn.Linear(hidden_channels, hidden_channels))
        layers.append(activation)
    layers.append(nn.Linear(hidden_channels, out_channels))
    return layers

def to_dense(edge_index, edge_weight):
    N = edge_index.max().item() + 1
    dense = torch.zeros(N, N, dtype=edge_weight.dtype, device=edge_index.device)
    dense[edge_index[0], edge_index[1]] = edge_weight
    return dense

def to_sparse(dense):
    edge_index = dense.nonzero(as_tuple=False).t()
    edge_weight = dense[edge_index[0], edge_index[1]]
    return edge_index, edge_weight

def to_sparse_batch(adj_matrices):
    offset, row, col = adj_matrices.nonzero().t()
    max_nodes = adj_matrices.size(1)
    edge_weight = adj_matrices[offset, row, col]
    row = row + (offset * max_nodes)
    col = col + (offset * max_nodes)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index, edge_weight


def to_dense_batch(edge_index, edge_weight, batch_size, max_nodes):
    adj_matrices = torch.zeros(batch_size, max_nodes, max_nodes, dtype=edge_weight.dtype, device=edge_weight.device)
    row, col = edge_index
    batch_idx = row // max_nodes  # integer division to get the batch index
    row = row % max_nodes
    col = col % max_nodes
    adj_matrices[batch_idx, row, col] = edge_weight
    return adj_matrices
    
    