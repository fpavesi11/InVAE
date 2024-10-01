from metricLearningApproach.utils import to_dense_batch
from metricLearningApproach.MultisetInjective import ReparametrizationTrick
import torch
from torch import nn

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