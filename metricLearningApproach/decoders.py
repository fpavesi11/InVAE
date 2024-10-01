import torch
from torch import nn
from metricLearningApproach.MultisetInjective import make_mlp
from torch.nn import functional as F
"""
Auto Regressive Decoder
""" 
class ARDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, graph_size, out_channels, depth, activation=nn.SiLU()):
        super(ARDecoder, self).__init__()
        self.estim_h = nn.Sequential(*make_mlp(in_channels, hidden_channels, hidden_channels, depth, activation))
        self.estim_c = nn.Sequential(*make_mlp(in_channels, hidden_channels, hidden_channels, depth, activation))
        self.rnn_1 = nn.LSTMCell(graph_size + out_channels, hidden_channels)
        self.rnn_2 = nn.LSTMCell(hidden_channels, hidden_channels)
        self.rnn_3 = nn.LSTMCell(hidden_channels, graph_size + out_channels)
        self.mid_h_adapter = nn.Linear(hidden_channels, hidden_channels)
        self.mid_c_adapter = nn.Linear(hidden_channels, hidden_channels)
        self.last_h_adapter = nn.Linear(hidden_channels, graph_size + out_channels)
        self.last_c_adapter = nn.Linear(hidden_channels, graph_size + out_channels)
        self.graph_size = graph_size
        self.out_channels = out_channels
        
    def forward(self, z, m_true=None):
        h_0 = self.estim_h(z)
        c_0 = self.estim_c(z)
        m_hat = []
        m_true = torch.flip(m_true, dims=[1])
        for t in range(m_true.size(1)-1):
            h_0, c_0 = self.rnn_1(m_true[:, t, :], (h_0, c_0))
            h_1 = self.mid_h_adapter(h_0)
            c_1 = self.mid_c_adapter(c_0)
            
            h_1, c_1 = self.rnn_2(h_0, (h_1, c_1))
            h_2 = self.last_h_adapter(h_1)
            c_2 = self.last_c_adapter(c_1)
            
            h_2, c_2 = self.rnn_3(h_1, (h_2, c_2))
            m_hat.append(h_2)
            
        m_hat = torch.stack(m_hat, dim=1)
        return m_hat
    
    def predict(self, z):
        h_0 = self.estim_h(z)
        c_0 = self.estim_c(z)
        pred_node = torch.zeros((z.size(0), 1 , self.graph_size + self.out_channels))
        pred_node[:,:,-1] = 1.
        m_hat = [pred_node]
        for t in range(self.graph_size-1):
            h_0, c_0 = self.rnn_1(pred_node, (h_0, c_0))
            h_1 = self.mid_h_adapter(h_0)
            c_1 = self.mid_c_adapter(c_0)
            
            h_1, c_1 = self.rnn_2(h_0, (h_1, c_1))
            h_2 = self.last_h_adapter(h_1)
            c_2 = self.last_c_adapter(c_1)
            
            h_2, c_2 = self.rnn_3(h_1, (h_2, c_2))
            pred_node = h_2
            m_hat.append(h_2)
            
        m_hat = torch.stack(m_hat, dim=1)
        return m_hat
    
    
"""
Big Auto Regressive Decoder
""" 
class BigARDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, graph_size, out_channels, linear_depth, recurrent_depth, activation=nn.SiLU()):
        super(BigARDecoder, self).__init__()
        self.estim_h = nn.Sequential(*make_mlp(in_channels, hidden_channels, hidden_channels, linear_depth, activation))
        self.estim_c = nn.Sequential(*make_mlp(in_channels, hidden_channels, hidden_channels, linear_depth, activation))
        # we estimate only the coloring of the last node as connections are fixed 
        self.estim_last_x = nn.Sequential(*make_mlp(in_channels, hidden_channels, out_channels, linear_depth, activation)) 
        
        self.rnn_in = nn.LSTMCell(graph_size + out_channels, hidden_channels)
        self.rnns = nn.ModuleList(nn.LSTMCell(hidden_channels, hidden_channels) for _ in range(recurrent_depth))
        self.h_adapters = nn.ModuleList(nn.Linear(hidden_channels, hidden_channels) for _ in range(recurrent_depth))
        self.c_adapters = nn.ModuleList(nn.Linear(hidden_channels, hidden_channels) for _ in range(recurrent_depth))
        self.rnn_out = nn.LSTMCell(hidden_channels, graph_size + out_channels)
        
        self.mid_h_adapter = nn.Linear(hidden_channels, hidden_channels)
        self.mid_c_adapter = nn.Linear(hidden_channels, hidden_channels)
        self.last_h_adapter = nn.Linear(hidden_channels, graph_size + out_channels)
        self.last_c_adapter = nn.Linear(hidden_channels, graph_size + out_channels)
        
        self.graph_size = graph_size
        self.out_channels = out_channels
        
    def forward(self, z, m_true):
        h_0 = self.estim_h(z)
        c_0 = self.estim_c(z)
        x_last = self.estim_last_x(z)
        m_hat_last = torch.cat([torch.zeros_like(m_true[:,0,:self.graph_size]), x_last], dim=-1)
        m_hat = [m_hat_last]
        m_true = torch.flip(m_true, dims=[1])
        for t in range(m_true.size(1)-1):
            h_2, c_2 = self.rnn_in(m_true[:, t, :], (h_0, c_0))
            
            for i in range(len(self.rnns)):
                h_1 = self.h_adapters[i](h_2)
                c_1 = self.c_adapters[i](c_2)
                h_2, c_2 = self.rnns[i](h_2, (h_1, c_1))
            
            h_0, c_0 = h_2, c_2 #<--- this state will be passed to next timesteps
    
            h_1 = h_2 #<--- this is just a renaming to be ok with the rnn_out
            h_2 = self.last_h_adapter(h_1)
            c_2 = self.last_c_adapter(c_1)
            
            h_2, c_2 = self.rnn_out(h_1, (h_2, c_2))
            m_hat.append(h_2)
            
        m_hat = torch.stack(m_hat, dim=1)
        return m_hat
    
    def predict(self, z):
        h_0 = self.estim_h(z)
        c_0 = self.estim_c(z)
        pred_node = torch.zeros((z.size(0), self.graph_size + self.out_channels))
        pred_node[:,-1] = 1.
        m_hat = [pred_node]
        for t in range(self.graph_size-1):
            h_2, c_2 = self.rnn_in(pred_node, (h_0, c_0))
            
            for i in range(len(self.rnns)):
                h_1 = self.h_adapters[i](h_2)
                c_1 = self.c_adapters[i](c_2)
                h_2, c_2 = self.rnns[i](h_2, (h_1, c_1))
            
            h_0, c_0 = h_2, c_2 #<--- this state will be passed to next timesteps
            
            h_1 = h_2
            h_2 = self.last_h_adapter(h_1)
            c_2 = self.last_c_adapter(c_1)
            
            h_2, c_2 = self.rnn_out(h_1, (h_2, c_2))
            pred_node = h_2
            m_hat.append(h_2)
            
        m_hat = torch.stack(m_hat, dim=1)
        return m_hat


    