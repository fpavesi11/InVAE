from torch_geometric.nn import MessagePassing, global_add_pool
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import warnings 
from metricLearningApproach.utils import reshape_to_matrix, mask_out, make_mlp

"""
Injectivity is ensured by having an analytic activation at the end
of each MLP, plus the embedding dimension being m = 2n(d+1) + 1, (m = 2nd + 1 if unweighted)
check Proposition 3.6 of Amir (2023)

Also section 6.2 of Amir (2023) is fundamental for understanding the implementation 
"""
class MPLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add', activation=nn.SiLU(), **kwargs):
        super(MPLayer, self).__init__(aggr=aggr, **kwargs)

        self.A = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.b = nn.Parameter(torch.Tensor(1, out_channels))
        self.eta = nn.Parameter(torch.Tensor(1, 1))
        self.activation = activation

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.A)
        init.xavier_uniform_(self.b)
        init.xavier_uniform_(self.eta)

    def forward(self, x, edge_index, edge_weight=None):
        # Call propagate and pass the edge_weight to message function
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_i, x_j, edge_weight):
        # x_i is the central node, x_j is the neighbor node
        # edge_weight is used to scale the message
        message = self.activation(torch.matmul(x_j + self.eta * x_i, self.A) + self.b)
        if edge_weight is not None:
            message = message * edge_weight.view(-1, 1)  # Scale by edge weight
        return message


"""
NOTE: The readout layer still outputs node embeddings, a graph summation is needed
""" 
class Readout(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.SiLU()):
        super(Readout, self).__init__()
        self.A = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.b = nn.Parameter(torch.Tensor(1, out_channels))
        self.activation = activation
        
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.A)
        init.xavier_uniform_(self.b)

    def forward(self, x):
        return self.activation(torch.matmul(x, self.A) + self.b)
    

# version 2   
class MultisetInjective(nn.Module):
    def __init__(self, max_nodes, in_channels, hidden_channels, num_layers, activation=nn.SiLU()):
        super(MultisetInjective, self).__init__()
        
        if hidden_channels < 2 * in_channels * (max_nodes + 1) + 1:
            hidden_channels = 2 * in_channels * (max_nodes + 1) + 1
            warnings.warn('The hidden dimension is not enough to ensure injectivity, setting it to the minimum value of 2nd + 1')

        self.max_nodes = max_nodes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        self.layers.append(MPLayer(in_channels, hidden_channels, activation=activation))
        for _ in range(num_layers - 2):
            self.layers.append(MPLayer(hidden_channels, hidden_channels, activation=activation))
        self.layers.append(MPLayer(hidden_channels, hidden_channels, activation=activation))

        self.readout = Readout(hidden_channels, hidden_channels, activation=activation)

    def forward(self, x, edge_index, batch=None, edge_weight=None):
        # Default edge_weight to None
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
        
        # Apply readout
        self.readout(x)

        # Perform global pooling
        x = global_add_pool(x, batch)
        return x
    
    
class InjectiveEncoder(nn.Module):
    def __init__(self, MSI, in_channels, hidden_channels, num_layers, out_channels, activation=nn.SiLU()):
        super(InjectiveEncoder, self).__init__()
        self.MSI = MSI
        self.estim_mu = nn.Sequential(*make_mlp(hidden_channels, hidden_channels, out_channels, num_layers, activation))

        self.estim_log_sigma = nn.Sequential(*make_mlp(hidden_channels, hidden_channels, out_channels, num_layers, activation))

        self.estim_L_mask = nn.Sequential(*make_mlp(hidden_channels, hidden_channels, out_channels**2, num_layers, activation),
                                          reshape_to_matrix(),
                                          mask_out())

    def forward(self, x, edge_index, batch, edge_weight=None):
        x = self.MSI(x, edge_index, batch, edge_weight)
        mu = self.estim_mu(x)
        log_sigma = self.estim_log_sigma(x)
        L_mask = self.estim_L_mask(x)
        return mu, log_sigma, L_mask
    
    
class ReparametrizationTrick(nn.Module):
    def __init__(self):
        super(ReparametrizationTrick, self).__init__()

    def forward(self, mu, log_sigma, L_mask):
        L_diag = L_mask + torch.diag_embed(torch.exp(log_sigma)) #notice here is sigma, while in standard vae is sigma2
        z = mu + torch.bmm(L_diag, torch.randn_like(mu).unsqueeze(-1)).squeeze(-1)
        return z, L_diag
