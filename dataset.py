import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.utils import dense_to_sparse
from torch.utils.data import Dataset
from data_loader import matrix_to_data

class EEGGraphDataset(Dataset):
    def __init__(self, data, labels=None):
        """
        data: list of EEG sequences (each of shape [T, num_channels])
        labels: list of labels (optional)
        For each EEG sample, computes:
           - A distance graph (Euclidean distances between channels)
           - A correlation graph (using np.corrcoef)
        Each graph is converted to a PyTorch Geometric Data object.
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        eeg = self.data[idx]  # shape: (T, num_channels)
        num_channels = eeg.shape[1]
        # Compute distance graph
        dist_graph = np.zeros((num_channels, num_channels), dtype=np.float32)
        for i in range(num_channels):
            for j in range(num_channels):
                dist_graph[i, j] = np.linalg.norm(eeg[:, i] - eeg[:, j])
        # Compute correlation graph
        corr_graph = np.corrcoef(eeg.T).astype(np.float32)
        corr_graph = np.nan_to_num(corr_graph)
        # Convert matrices to Data objects
        data_dist = matrix_to_data(dist_graph)
        data_corr = matrix_to_data(corr_graph)
        if self.labels is not None:
            label = int(self.labels[idx])
            return data_dist, data_corr, label
        else:
            return data_dist, data_corr

def pad_data(data, max_nodes):
    """
    Pads a PyTorch Geometric Data object so that its node features have shape [max_nodes, feature_dim].
    Assumes data.x is of shape [n, feature_dim] where n < max_nodes.
    Also creates a fully connected edge_index for the padded graph.
    """
    n, feat_dim = data.x.shape
    if n < max_nodes:
        padded_x = F.pad(data.x, (0, 0, 0, max_nodes - n), "constant", 0)
        # Create a new fully-connected edge index for max_nodes nodes
        adj = torch.ones((max_nodes, max_nodes), dtype=torch.float)
        edge_index, _ = dense_to_sparse(adj)
        data.x = padded_x
        data.edge_index = edge_index
    return data

def collate_geometric(batch):
    """
    Collates a list of samples into batched PyTorch Geometric objects.
    Pads each graph so that all Data objects have the same number of nodes.
    If labels are provided, returns (batch_dist, batch_corr, labels).
    """
    if len(batch[0]) == 2:
        data_dist_list, data_corr_list = zip(*batch)
    else:
        data_dist_list, data_corr_list, labels = zip(*batch)
    max_nodes = max(
        [data.x.shape[0] for data in data_dist_list] + 
        [data.x.shape[0] for data in data_corr_list]
    )
    data_dist_list = [pad_data(data, max_nodes) for data in data_dist_list]
    data_corr_list = [pad_data(data, max_nodes) for data in data_corr_list]
    batch_dist = Batch.from_data_list(data_dist_list)
    batch_corr = Batch.from_data_list(data_corr_list)
    if len(batch[0]) == 2:
        return batch_dist, batch_corr
    else:
        return batch_dist, batch_corr, torch.tensor(labels)
