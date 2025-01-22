import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
from configs import Config as config

class GraphBuilder:
    """Handles graph construction for EEG data"""
    
    def __init__(self):
        self.coords = np.array([config.coord_map[e] for e in config.coord_map.keys()])
        
    def build_spatial_graph(self):
        """Static spatial connectivity graph"""
        dist = np.sqrt(((self.coords[:, None] - self.coords[None, :])**2).sum(-1))
        sigma = dist.std()
        adj = np.exp(-(dist**2) / (sigma**2))
        adj[dist > config.spatial_kappa] = 0
        return self._to_edge_index(adj)
    
    def build_functional_graph(self, batch):
        """Safe functional graph construction with NaN handling"""
        # Input: (batch_size, num_nodes, num_features)
        batch = batch.float()
        
        corr_matrices = []
        for sample in batch:
            # 1. Check for constant values
            std_dev = sample.std(dim=1, keepdim=True)
            zero_std_mask = (std_dev < 1e-6).squeeze()
            
            # 2. Add small noise to constant channels
            if zero_std_mask.any():
                noise = torch.randn_like(sample) * 1e-6
                sample = torch.where(zero_std_mask.unsqueeze(1), sample + noise, sample)
            
            # 3. Compute correlations safely
            cov = torch.mm(sample, sample.t()) 
            std = torch.sqrt(torch.diag(cov))
            eps = 1e-6 * torch.ones_like(std)
            corr = cov / (torch.outer(std, std) + eps)
            
            # 4. Clamp invalid values
            corr = torch.clamp(corr, -1.0, 1.0)
            corr[torch.eye(corr.shape[0], dtype=bool)] = 1.0  # Diagonal
            
            corr_matrices.append(corr)
        
        # 5. Average across batch
        avg_corr = torch.mean(torch.stack(corr_matrices), dim=0)
        
        # 6. Final NaN check and replacement
        avg_corr = torch.where(torch.isnan(avg_corr), torch.zeros_like(avg_corr), avg_corr)
        
        # Threshold connections
        adj = torch.zeros_like(avg_corr)
        topk_values, topk_indices = torch.topk(avg_corr.flatten(), 
                                             k=config.functional_tau)
        adj.view(-1)[topk_indices] = topk_values
        
        return self._to_edge_index(adj.numpy())

    def _to_edge_index(self, adj):
        """Convert dense adjacency matrix to sparse edge index format"""
        edge_index, edge_weight = dense_to_sparse(torch.tensor(adj))
        return edge_index, edge_weight