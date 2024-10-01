#%%
import os 
import networkx as nx
if os.getcwd().split('\\')[-1] != 'metricLearningApproach':
    os.chdir('metricLearningApproach')
import torch
from DAGgen import load_ENAS_graphs
from tqdm import tqdm

g_list_train, g_list_test, graph_args = load_ENAS_graphs('final_structures6', n_types=6, fmt='igraph', rand_seed=0, with_y=True, burn_in=1000)

print('Number of training graphs:', len(g_list_train))
print('Number of test graphs:', len(g_list_test))
print('Number of node types:', graph_args.num_vertex_type)
print('Maximum number of nodes:', graph_args.max_n)
print('Start vertex type:', graph_args.START_TYPE)
print('End vertex type:', graph_args.END_TYPE)

#%%
from torch_geometric.utils.convert import from_networkx
from torch.nn.functional import one_hot

"""a = g_list_train[0][0]
print(a)

print(torch.tensor(a.get_adjacency().data))
g = from_networkx(a.to_networkx())
g.x = torch.tensor(a.vertex_coloring_greedy())
print(g)
print(g.x)
print(g.edge_index)"""

"""def igraph_to_torch(igraph_graph, num_classes):
    g = from_networkx(igraph_graph.to_networkx())
    g.x = one_hot(torch.tensor(igraph_graph.vertex_coloring_greedy()), num_classes)
    return g"""
def igraph_to_torch(igraph_graph, num_classes):
    # Convert igraph to NetworkX and then to PyTorch Geometric
    g = from_networkx(igraph_graph.to_networkx())
    
    # Node features: One-hot encoding based on vertex coloring
    g.x = one_hot(torch.tensor(igraph_graph.vertex_coloring_greedy()), num_classes)
    
    # Edge weights: Since the graph is unweighted, we assign a weight of 1 to all edges
    num_edges = g.edge_index.shape[1]
    g.edge_attr = torch.ones(num_edges, dtype=torch.float)  # Edge weights of 1
    
    return g

def igraph_to_torch_batch(igraph_graphs, num_classes=-1):
    return [igraph_to_torch(g, num_classes) for g, _ in tqdm(igraph_graphs)]

def igraph_batch_extract_label(igraph_graphs):
    return [l for _, l in tqdm(igraph_graphs)]

#%%
num_classes = 6
g_list_train_torch = igraph_to_torch_batch(g_list_train, num_classes)
g_list_test_torch = igraph_to_torch_batch(g_list_test, num_classes)

y_train = igraph_batch_extract_label(g_list_train)
y_test = igraph_batch_extract_label(g_list_test)


#%%

from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

class NNData(Dataset):
    def __init__(self, g_list, y_list):
        self.g_list = g_list
        self.y_list = y_list

    def __len__(self):
        return len(self.g_list)

    def __getitem__(self, idx):
        return self.g_list[idx], self.y_list[idx]
    
train_data = NNData(g_list_train_torch, y_train)
test_data = NNData(g_list_test_torch, y_test)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


#%%
from MultisetInjective import *
from torch import nn
from decoders import ARDecoder, BigARDecoder

max_nodes=8
in_channels=6


MSI = MultisetInjective(max_nodes=max_nodes,
                        in_channels=6,
                        hidden_channels=512,
                        num_layers=20,
                        activation=nn.Tanh())

encoder = InjectiveEncoder(MSI = MSI,
                            in_channels=6,
                            hidden_channels=512,
                            num_layers=5,
                            out_channels=256,
                            activation=nn.Tanh())

"""decoder = ARDecoder(in_channels = 32,
                    hidden_channels=512,
                    graph_size=8,
                    out_channels=6,
                    depth=6,
                    activation=nn.Tanh())"""
                    
decoder = BigARDecoder(in_channels = 256,
                        hidden_channels=512,
                        graph_size=8,
                        out_channels=6,
                        linear_depth=5,
                        recurrent_depth=1,
                        activation=nn.Tanh())

from torch_geometric.utils import dense_to_sparse
from metricLearningApproach.utils import to_dense_batch, to_sparse_batch

class InVAE(nn.Module):
    def __init__(self, encoder, decoder, max_nodes, num_edge_types=6):
        super(InVAE, self).__init__()
        self.encoder = encoder
        self.reparametrization = ReparametrizationTrick()
        self.decoder = decoder
        self.max_nodes = max_nodes
        self.num_edge_types = num_edge_types
        
    def forward(self, x, edge_index, batch, edge_weight=None):  
        mu, log_sigma, L_mask = self.encoder(x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
        z, L_diag = self.reparametrization(mu, log_sigma, L_mask)
        
        # build m_true. Notice the last row is fixed and hence excluded
        adj = to_dense_batch(edge_index, edge_weight, mu.size(0), max_nodes=self.max_nodes)
        m_true = torch.cat([adj, x.view(-1, self.max_nodes, self.num_edge_types)], dim=-1)
        
        m_hat = self.decoder(z, m_true=m_true)
        
        # m_hat does not contain the last row, which is known and fixed
        # As decoder reconstructs backwards, the output is flipped
        m_hat = m_hat.flip([1])
        
        adj_matrices, vertex = torch.split(m_hat, [self.max_nodes, self.num_edge_types], dim=-1)
        
        # ADJ MATRICES 
        # TODO: optimize instead of masking
        # pass adj through sigmoid and mask adj matrices to be upper triangular (DAG)
        adj_matrices = torch.nn.Sigmoid()(adj_matrices) * torch.triu(torch.ones_like(adj_matrices), diagonal=1)
        
        # VERTEX
        # TODO: we need again to fix the last node because of softmax. 
        # This is not so expensive but should be fixed
        vertex = torch.nn.Softmax(dim=-1)(vertex)
        
        return adj_matrices, vertex, L_diag, mu, log_sigma
    
    
model = InVAE(encoder, decoder, 8, 6)

#%%

from metricLearningApproach.utils import to_dense_batch
for batch in train_loader:
    y = batch[1]
    x, edge_index, edge_weight = batch[0].x.float(), batch[0].edge_index, batch[0].edge_attr
    adj = to_dense_batch(edge_index, edge_weight, 32, max_nodes=8)
    m_true = torch.cat([adj, x.view(-1, 8, 6)], dim=-1)
    z = torch.randn((batch[0].num_graphs, 32))
    #out = ard.forward(z, m_true)
    adj_matrices, vertex, L_diag, mu, log_sigma = model.forward(x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch[0].batch)
    break


#%%
"""loss_model = VAELoss(max_nodes=6, in_channels=6, hidden_channels=128, depth=5)
for batch in tqdm(train_loader):
    y = batch[1]
    x, edge_index, edge_weight = batch[0].x.float(), batch[0].edge_index, batch[0].edge_attr
    edge_index_hat, edge_weight_hat, x_hat, L_diag, mu, log_sigma = model.forward(x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch[0].batch)
    loss = loss_model(x_hat, edge_index_hat, x, edge_index, batch[0].batch, mu, L_diag, edge_weight, edge_weight_hat)
    print(x_hat)
    break"""
    
import torch.optim as optim
from metricLearningApproach.losses import VAELoss

batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


device = 'cpu'


# Define the loss function and optimizer
loss_model = VAELoss(max_nodes = 8, num_edge_types=6, alpha=0.0001)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training routine
def train_model(model, train_loader, loss_model, optimizer, device='cpu', num_epochs=100):
    model.train()
    model.to(device)
    for epoch in range(num_epochs):
        total_loss = 0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
            for batch in train_loader:
                y = batch[1]
                x, edge_index, edge_weight = batch[0].x.float(), batch[0].edge_index, batch[0].edge_attr
                x = x.to(device)
                edge_index = edge_index.to(device)
                edge_weight = edge_weight.to(device)
                optimizer.zero_grad()
                adj_matrices, vertex, L_diag, mu, log_sigma = model.forward(
                    x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch[0].batch
                )
                adj_true = to_dense_batch(edge_index, edge_weight, mu.size(0), max_nodes=8)
                #adj_true = adj_true#[:, :-1, :]
                x_true = x.view(-1, 8, 6)#[:, :-1, :]
                loss, vertex_loss, adj_loss, KL_div, kl_weight = loss_model(
                    vertex, adj_matrices, x_true, adj_true, batch[0].batch, mu, L_diag
                )
                loss.backward()
                optimizer.step()
                
                pbar.set_postfix(
                    loss=loss.item(),
                    vertex_loss=vertex_loss.item(),
                    adj_loss=adj_loss.item(),
                    KL_div=KL_div.item(),
                    kl_weight=kl_weight
                )
                pbar.update(1)

# Train the model
train_model(model, train_loader, loss_model, optimizer, device, num_epochs=10)

    

#%%
