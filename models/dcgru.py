# models/dcgru.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, GCNConv
from configs import Config as config

class DCGRU(nn.Module):
    """Dual Graph Convolutional GRU Model"""
    
    def __init__(self, input_dim, hidden_dim, cheb_k):
        super().__init__()
        self.num_nodes = len(config.coord_map)
        self.input_dim = input_dim
        
        # Encoder
        self.cheb_conv = ChebConv(input_dim, hidden_dim, cheb_k)
        self.diff_conv = GCNConv(input_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=2 * hidden_dim * self.num_nodes,
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True
        )
        
        # Attention layer (MISSING IN YOUR IMPLEMENTATION)
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),  # Bidirectional GRU output
            nn.Tanh()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, self.num_nodes * input_dim),
            nn.Tanh()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x, spatial_edges, functional_edges, pretrain=False):
        # Ensure proper dimensions
        x = x.float()
        if x.dim() == 3:
            x = x.unsqueeze(2)  # Add time dimension
            
        batch_size, nodes, time_steps, _ = x.shape
        outputs = []
        
        # Process each timestep
        for t in range(time_steps):
            x_t = x[:, :, t, :]
            h_spatial = self.cheb_conv(x_t, spatial_edges[0], spatial_edges[1])
            h_functional = self.diff_conv(x_t, functional_edges[0], functional_edges[1])
            outputs.append(torch.cat([h_spatial, h_functional], dim=-1))
        
        # Temporal processing
        gru_input = torch.stack(outputs, dim=1).view(batch_size, time_steps, -1)
        gru_out, _ = self.gru(gru_input)
        
        if pretrain:
            # Reconstruction output [batch, nodes, features]
            recon = self.decoder(gru_out[:, -1, :])
            return recon.view(batch_size, self.num_nodes, self.input_dim)
        else:
            # Classification output
            attn_weights = F.softmax(self.attn(gru_out), dim=1)
            context = torch.sum(attn_weights * gru_out, dim=1)
            return self.classifier(context)