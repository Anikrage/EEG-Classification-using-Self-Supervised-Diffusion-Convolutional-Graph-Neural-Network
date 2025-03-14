import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from data_loader import load_eeg_dataset
from dataset import EEGGraphDataset, collate_geometric
from models import GCNEncoder, ProjectionHead, SelfSupervisedModel, FusionClassifier
from train import pretrain, fine_tune, evaluate_classifier

def main():
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Update the dataset directory as needed
    dataset_dir = r"C:\Users\Aneek\ds003474-download"
    data_list, labels_list = load_eeg_dataset(dataset_dir, bdi_threshold=20)
    
    # Create dataset objects for self-supervised pretraining (unlabeled)
    pretrain_dataset = EEGGraphDataset(data=data_list, labels=None)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=8, shuffle=True, collate_fn=collate_geometric)
    
    # Determine number of channels from the first sample
    num_channels = data_list[0].shape[1]
    hidden_channels = 32
    embedding_dim = 128
    
    # Initialize the GCN encoder and self-supervised model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gcn_encoder = GCNEncoder(
        in_channels=num_channels, 
        hidden_channels=hidden_channels, 
        out_channels=embedding_dim, 
        num_layers=3
    ).to(device)
    
    projection_head = ProjectionHead(embedding_dim=embedding_dim, projection_dim=64).to(device)
    ss_model = SelfSupervisedModel(gcn_encoder, projection_head).to(device)
    ss_optimizer = torch.optim.Adam(ss_model.parameters(), lr=0.001)
    
    print("Starting self-supervised pretraining...")
    pretrain(ss_model, pretrain_loader, ss_optimizer, device, epochs=10)
    
    # Split dataset for fine tuning (train/val/test)
    full_dataset = EEGGraphDataset(data=data_list, labels=labels_list)
    total_samples = len(full_dataset)
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)
    test_size = total_samples - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_geometric)
    val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=collate_geometric)
    test_loader = DataLoader(test_dataset, batch_size=8, collate_fn=collate_geometric)
    
    # Initialize fusion classifier for fine tuning.
    classifier = FusionClassifier(gcn_encoder, embedding_dim=embedding_dim, num_classes=2).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(classifier_optimizer, T_max=10)
    
    print("\nStarting fine tuning...")
    fine_tune(classifier, train_loader, criterion, classifier_optimizer, scheduler, device, epochs=20)
    
    test_acc, test_f1 = evaluate_classifier(classifier, test_loader, device)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    print(f"Final Test Weighted F1 Score: {test_f1:.4f}")

if __name__ == "__main__":
    main()
