import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        """
        A simple GCN encoder.
        in_channels: dimension of node features (here equal to number of channels)
        hidden_channels: hidden dimension
        out_channels: final embedding dimension
        """
        super(GCNEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        return x

class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim=128, projection_dim=64):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class SelfSupervisedModel(nn.Module):
    def __init__(self, encoder, projection_head):
        super(SelfSupervisedModel, self).__init__()
        self.encoder = encoder
        self.projection_head = projection_head
    
    def forward(self, data):
        embedding = self.encoder(data)
        projection = self.projection_head(embedding)
        return projection, embedding

class FusionClassifier(nn.Module):
    def __init__(self, encoder, embedding_dim=128, num_classes=2):
        """
        Fusion classifier that concatenates embeddings from two graph views.
        """
        super(FusionClassifier, self).__init__()
        self.encoder = encoder  # Pretrained encoder (shared)
        self.fusion_fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, data_dist, data_corr):
        emb1 = self.encoder(data_dist)
        emb2 = self.encoder(data_corr)
        fusion_emb = torch.cat([emb1, emb2], dim=1)
        logits = self.fusion_fc(fusion_emb)
        return logits
