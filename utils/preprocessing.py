import numpy as np
import pandas as pd
import torch
from scipy.signal import welch, firwin, filtfilt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from configs import Config as config

class EEGDataset(Dataset):
    """EEG dataset loader without class balance filtering"""
    def __init__(self, csv_path, condition_filter=None):
        df = pd.read_csv(csv_path)
        self.electrodes = list(config.coord_map.keys())
        
        # Condition filtering
        if condition_filter:
            if not isinstance(condition_filter, list):
                condition_filter = [condition_filter]
            df = df[df['Condition'].isin(condition_filter)]
            
        # Column selection
        required_cols = [f"{e}-LE" for e in self.electrodes] + ['Group', 'Condition', 'Subject ID']
        self.df = df[required_cols]
        
        # Preprocessing
        self.data, self.labels, self.subjects = self._preprocess()
        
        # Debug info
        print(f"\nDataset Info:")
        print(f"- Total samples: {len(self)}")
        print(f"- Subjects: {len(np.unique(self.subjects))}")
        print(f"- Classes: {np.unique(self.labels.numpy(), return_counts=True)}")

    def _preprocess(self):
        scaler = StandardScaler()
        all_epochs, all_labels, all_subjects = [], [], []
        
        for subj in self.df['Subject ID'].unique():
            subj_df = self.df[self.df['Subject ID'] == subj]
            eeg_data = subj_df[[f"{e}-LE" for e in self.electrodes]].values.T
            eeg_data = self._bandpass_filter(eeg_data)
            eeg_data = scaler.fit_transform(eeg_data.T).T
            
            samples_per_epoch = config.sampling_rate * config.epoch_length
            step = int(samples_per_epoch * (1 - config.overlap))
            
            for start in range(0, eeg_data.shape[1] - samples_per_epoch, step):
                epoch = eeg_data[:, start:start+samples_per_epoch]
                features = self._extract_features(epoch, subj_df['Condition'].iloc[0])
                all_epochs.append(features)
                all_labels.append(1 if subj_df['Group'].iloc[0] == 'MDD' else 0)
                all_subjects.append(subj)
                
        return (torch.tensor(np.stack(all_epochs), dtype=torch.float32),
                torch.tensor(all_labels, dtype=torch.float32),
                np.array(all_subjects))

    def _bandpass_filter(self, data, low=0.5, high=45):
        nyq = 0.5 * config.sampling_rate
        taps = firwin(101, [low/nyq, high/nyq], pass_zero=False)
        return filtfilt(taps, 1.0, data)

    def _extract_features(self, epoch, condition):
        # Add this check
        if np.isnan(epoch).any():
            print("NaN in epoch data!")
            epoch = np.nan_to_num(epoch)
        freqs, psd = welch(epoch, fs=config.sampling_rate, nperseg=config.sampling_rate)
        features = []
        for band in config.bands.values():
            mask = (freqs >= band[0]) & (freqs <= band[1])
            # Fix 2: Ensure 2D output for proper stacking
            band_power = np.trapz(psd[:, mask], freqs[mask], axis=1).reshape(-1, 1)
            features.append(band_power)
        
        condition_enc = [1 if condition == c else 0 for c in config.valid_conditions]
        # Fix 3: Ensure consistent feature dimensions
        condition_enc = np.tile(condition_enc, (epoch.shape[0], 1))
        
        return np.hstack([np.concatenate(features, axis=1), condition_enc])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]