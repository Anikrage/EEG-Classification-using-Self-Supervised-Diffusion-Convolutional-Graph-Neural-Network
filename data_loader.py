import os
import glob
import warnings
import mne
import pandas as pd
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse

# List of channels to use (standard 64)
standard_64 = [
    'FP1', 'FP2', 'AF3', 'AF4',
    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO3', 'POZ', 'PO4', 'PO8',
    'O1', 'OZ', 'O2'
]

def load_eeg_dataset(dataset_dir, bdi_threshold=20):
    """
    Loads EEG signals from the dataset directory.
    Removes 'boundary' events and picks only the specified standard 64 channels.
    Returns:
       data_list: list of EEG signals (each as a numpy array of shape [T, num_channels])
       labels_list: list of labels (0/1) based on the BDI score threshold.
    """
    participants_file = os.path.join(dataset_dir, "participants.tsv")
    participants_df = pd.read_csv(participants_file, sep="\t")
    
    data_list = []
    labels_list = []
    subj_dirs = sorted([d for d in os.listdir(dataset_dir) if d.startswith("sub-")])
    
    for subj in subj_dirs:
        eeg_dir = os.path.join(dataset_dir, subj, "eeg")
        if os.path.isdir(eeg_dir):
            set_files = glob.glob(os.path.join(eeg_dir, "*_eeg.set"))
            if len(set_files) == 0:
                print(f"No .set file found for {subj}")
                continue
            set_file = set_files[0]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    raw = mne.io.read_raw_eeglab(set_file, preload=True, verbose=False)
                # Remove boundary annotations
                if raw.annotations is not None:
                    new_onsets, new_durations, new_descriptions = [], [], []
                    for onset, duration, description in zip(raw.annotations.onset,
                                                            raw.annotations.duration,
                                                            raw.annotations.description):
                        if 'boundary' not in description.lower():
                            new_onsets.append(onset)
                            new_durations.append(duration)
                            new_descriptions.append(description)
                    raw.set_annotations(mne.Annotations(onset=new_onsets,
                                                        duration=new_durations,
                                                        description=new_descriptions))
                # Pick only the standard 64 channels.
                available_channels = [ch for ch in standard_64 if ch in raw.ch_names]
                raw.pick(available_channels)
                # Get data with shape (n_times, n_channels)
                eeg_data = raw.get_data().T.astype(np.float32)
            except Exception as e:
                print(f"Error loading {set_file}: {e}")
                continue
            data_list.append(eeg_data)
            subj_info = participants_df[participants_df['participant_id'] == subj]
            if not subj_info.empty:
                try:
                    bdi = float(subj_info.iloc[0]['BDI'])
                except Exception as e:
                    print(f"Error parsing BDI for {subj}: {e}")
                    bdi = 0.0
                label = 1 if bdi >= bdi_threshold else 0
            else:
                label = 0
            labels_list.append(label)
    print(f"Loaded {len(data_list)} subjects.")
    return data_list, labels_list

def matrix_to_data(matrix):
    """
    Converts a square numpy matrix (n x n) into a PyTorch Geometric Data object.
    Uses each row of the matrix as node features and creates a fully connected graph.
    """
    from torch_geometric.data import Data  # local import to avoid circular dependencies
    n = matrix.shape[0]
    # Use the rows of the matrix as node features (each row is a feature vector)
    x = torch.tensor(matrix, dtype=torch.float)  # shape: [n, n]
    # Create a fully connected graph for n nodes
    adj = torch.ones((n, n), dtype=torch.float)
    edge_index, _ = dense_to_sparse(adj)
    data = Data(x=x, edge_index=edge_index)
    return data
