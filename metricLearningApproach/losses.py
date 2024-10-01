from torch.nn.modules.loss import _Loss
import torch
from torch import nn
from torch.nn import functional as F
   
def cross_entropy_onehot_softmax(predicted_softmax, true_onehot):

    # Check if the predicted tensor is already softmaxed
    assert torch.allclose(predicted_softmax.sum(dim=-1), torch.ones_like(predicted_softmax.sum(dim=-1))), \
        "Predicted tensor should be the result of a softmax operation."
    
    # Convert one-hot encoded true labels to class indices
    true_labels = torch.argmax(true_onehot, dim=-1)  # shape: (batch_size, num_nodes)
    
    # Compute the log of softmax predictions
    log_predicted = torch.log(predicted_softmax)  # shape: (batch_size, num_nodes, num_edge_types)
    
    # Reshape the tensors for NLLLoss
    log_predicted_reshaped = log_predicted.view(-1, log_predicted.size(-1))  # shape: (batch_size * num_nodes, num_edge_types)
    true_labels_reshaped = true_labels.view(-1)  # shape: (batch_size * num_nodes)
    
    # Define NLLLoss criterion
    criterion = nn.NLLLoss()
    
    # Compute the loss
    loss = criterion(log_predicted_reshaped, true_labels_reshaped)
    
    return loss

class CustomCELoss(_Loss):
    def __init__(self, reduction='mean'):
        super(CustomCELoss, self).__init__(reduction=reduction)
        self.reduction = reduction
        self.criterion = nn.NLLLoss(reduction=reduction)
        
    def forward(self, predicted, true):
        # Check if the predicted tensor is already softmaxed
        assert torch.allclose(predicted.sum(dim=-1), torch.ones_like(predicted.sum(dim=-1))), \
            "Predicted tensor should be the result of a softmax operation."
        # Convert one-hot encoded true labels to class indices
        true_labels = torch.argmax(true, dim=-1)  # shape: (batch_size, num_nodes)
        
        # Compute the log of softmax predictions
        log_predicted = torch.log(predicted)  # shape: (batch_size, num_nodes, num_edge_types)
        
        # Reshape the tensors for NLLLoss
        log_predicted_reshaped = log_predicted.view(-1, log_predicted.size(-1))  # shape: (batch_size * num_nodes, num_edge_types)
        true_labels_reshaped = true_labels.view(-1)  # shape: (batch_size * num_nodes)
        
        # Compute the loss
        loss = self.criterion(log_predicted_reshaped, true_labels_reshaped)
        
        return loss
     
class VAELoss(nn.Module):
    def __init__(self, max_nodes, num_edge_types,
                 reduction='sum', alpha=None, lam=1.0, add_noise=False, norm_constant=None):
        super(VAELoss, self).__init__()
        
        # KL -----
        self.reduction = reduction
        self.add_noise = add_noise
        self.alpha = alpha
        self.norm_constant = norm_constant
        if alpha is not None:
            if alpha < 1:
                self.kl_weight = 0
                self.method = 'Bowman'
            else:
                self.kl_weight = alpha
                if norm_constant is not None:
                    self.kl_weight *= norm_constant
                self.method = 'Beta'
        else:
            self.kl_weight = 1
        
        # Embedder -------
        self.max_nodes = max_nodes
        self.num_edge_types = num_edge_types
        
        # DAG regularization
        self.lam = lam
        
        # Vertex loss
        self.vertex_loss = CustomCELoss(reduction=reduction)

    def forward(self, x_hat, adj_hat, x_true, adj_true, batch, mu, L_diag, edge_weight_true=None, edge_weight_hat=None):
        
        # Reconstruction loss ---------------------
        #vertex_loss = F.cross_entropy(x_hat, x_true.view(-1, self.max_nodes-1, self.num_edge_types), reduction=self.reduction)
        vertex_loss = self.vertex_loss(x_hat, x_true)
        adj_loss = F.binary_cross_entropy(adj_hat, adj_true, reduction=self.reduction)
        
        # KL ---------------------------
        # Adds noise if needed        
        if self.add_noise:
            noise_scale = 1e-6
            noise = torch.randn_like(mu) * noise_scale
            L_diag += torch.diag_embed(noise)
            
        muTmu = torch.sum(mu  * mu, dim=1, keepdim=True) #simplifies matmul
        #tr_sigma = sigma.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1).unsqueeze(-1)
        tr_sigma = L_diag.pow(2).sum(dim=-1).sum(dim=-1).unsqueeze(-1)
        #tr_sigma =  sigma.diagonal(dim1=-1, dim2=-2).sum(-1).unsqueeze(-1)
        k = mu.size(1)
        # calculate determinant with the advantage of cholesky decomposition
        #log_det_sigma = L_diag.diagonal(offset=0, dim1=-1, dim2=-2).pow(2).prod(-1).log().unsqueeze(-1) 
        #log_det_sigma = 2 * L_diag.diagonal(offset=0, dim1=-1, dim2=-2).log().sum(-1).unsqueeze(-1)
        log_det_sigma = 2 * torch.log(L_diag.diagonal(dim1=-1, dim2=-2)).sum(-1).unsqueeze(-1)
        KL_div = 0.5*(muTmu + tr_sigma - k - log_det_sigma).mean()
        
        # Bowman method
        if self.kl_weight < 1 and self.method=='Bowman':
            self.kl_weight += self.alpha
            if self.kl_weight > 1:
                self.kl_weight = 1
                            
        return vertex_loss + adj_loss + KL_div * self.kl_weight + self.lam, vertex_loss, adj_loss, KL_div, self.kl_weight
        