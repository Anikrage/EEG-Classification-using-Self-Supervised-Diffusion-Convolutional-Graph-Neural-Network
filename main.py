import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from configs import config
from models.dcgru import DCGRU
from utils.preprocessing import EEGDataset
from utils.graphs import GraphBuilder

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pretrained_dcgru.pth")

def main(csv_path, condition_filter=None):
    # Initialize components
    dataset = EEGDataset(csv_path, condition_filter)
    
    # Check if dataset loaded properly
    if len(dataset) == 0:
        raise ValueError("No valid data loaded! Check your CSV file and filters")
    
    print(f"\nLoaded {len(np.unique(dataset.subjects))} subjects")
    print(f"Class distribution: {np.bincount(dataset.labels.int().numpy())}")
    
    graph_builder = GraphBuilder()
    spatial_edges = graph_builder.build_spatial_graph()
    spatial_edges = (spatial_edges[0].to(torch.long), 
                    spatial_edges[1].to(torch.float32))
    
    # Model setup
    input_dim = len(config.bands) + config.condition_dim
    model = DCGRU(input_dim, config.hidden_dim, config.cheb_k)
    
    # Pretraining
    if not os.path.exists(MODEL_PATH):
        print("\nStarting pretraining...")
        pretrain_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        pretrain_losses = pretrain(model, pretrain_loader, spatial_edges, graph_builder)
        
        plt.figure(figsize=(10, 5))
        plt.plot(pretrain_losses)
        plt.title("Pretraining Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.show()
    else:
        print("\nUsing existing pretrained model")

    # LOSO Validation
    logo = LeaveOneGroupOut()
    metrics = {'auc': [], 'f1': [], 'acc': []}
    subjects = dataset.subjects
    
    for fold, (train_idx, test_idx) in enumerate(logo.split(dataset.data, groups=subjects)):
        print(f"\nFold {fold+1}/{len(np.unique(subjects))}")
        
        # Model setup
        model = DCGRU(input_dim, config.hidden_dim, config.cheb_k)
        
        # Load pretrained weights
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True))
        except Exception as e:
            print(f"Loading error: {e}. Training from scratch")
        
        # Training
        train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), 
                                batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(torch.utils.data.Subset(dataset, test_idx),
                               batch_size=config.batch_size)
        
        preds, labels = finetune(model, train_loader, test_loader, spatial_edges, graph_builder)
        
        # Calculate metrics
        try:
            metrics['auc'].append(roc_auc_score(labels, preds))
            metrics['f1'].append(f1_score(labels, (preds > 0.5).astype(int), zero_division=0))
            metrics['acc'].append(accuracy_score(labels, (preds > 0.5).astype(int)))
        except ValueError as e:
            print(f"Metric calculation failed: {e}")

    # Final results
    if len(metrics['acc']) == 0:
        print("\nNo valid folds for evaluation!")
        print("Check: 1. Class balance 2. Subject filtering 3. Data quality")
    else:
        print("\nCross-Validation Results:")
        print(f"AUC: {np.nanmean(metrics['auc']):.2f} ± {np.nanstd(metrics['auc']):.2f}")
        print(f"F1: {np.mean(metrics['f1']):.2f} ± {np.std(metrics['f1']):.2f}")
        print(f"Accuracy: {np.mean(metrics['acc']):.2f} ± {np.std(metrics['acc']):.2f}")

def pretrain(model, loader, spatial_edges, graph_builder):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    model.train()
    losses = []
    
    for epoch in range(config.pretrain_epochs):
        total_loss = 0
        for x, _ in loader:
            x = x.float()
            mask = torch.rand(x.shape) < config.mask_ratio
            x_masked = x.clone()
            x_masked[mask] = 0
            
            # Functional edges with NaN handling
            func_edges = graph_builder.build_functional_graph(x_masked)
            func_edges = (func_edges[0].to(torch.long),
                         torch.nan_to_num(func_edges[1], 0).to(torch.float32))
            
            pred = model(x_masked, spatial_edges, func_edges, pretrain=True)
            loss = F.mse_loss(pred[mask], x[mask])
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        epoch_loss = total_loss / len(loader)
        losses.append(epoch_loss)
        print(f"Pretrain Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
    
    torch.save(model.state_dict(), MODEL_PATH)
    return losses

def finetune(model, train_loader, test_loader, spatial_edges, graph_builder):
    # Class weighting
    train_labels = torch.cat([y for _, y in train_loader]).float()
    class_counts = torch.bincount(train_labels.long())
    weights = 1. / (class_counts + 1e-6)  # Prevent division by zero
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights[1]/weights[0])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    # Training
    model.train()
    for epoch in range(config.finetune_epochs):
        epoch_loss = 0
        for x, y in train_loader:
            x = x.float()
            func_edges = graph_builder.build_functional_graph(x)
            func_edges = (func_edges[0].to(torch.long),
                         torch.nan_to_num(func_edges[1], 0).to(torch.float32))
            
            pred = model(x, spatial_edges, func_edges, pretrain=False)
            loss = criterion(pred.squeeze(), y.float())
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Finetune Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}")
    
    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.float()
            func_edges = graph_builder.build_functional_graph(x)
            func_edges = (func_edges[0].to(torch.long),
                         torch.nan_to_num(func_edges[1], 0).to(torch.float32))
            
            pred = torch.sigmoid(model(x, spatial_edges, func_edges, pretrain=False))
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)

if __name__ == "__main__":
    main(r"C:\\Users\\Aneek\\Desktop\\research\\data\\eeg_data.csv")