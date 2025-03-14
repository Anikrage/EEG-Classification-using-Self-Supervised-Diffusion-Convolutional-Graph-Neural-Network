import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

def nt_xent_loss(z1, z2, temperature=0.5):
    batch_size = z1.shape[0]
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    similarity_matrix = torch.matmul(z, z.T)
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)
    positives = torch.cat([
        torch.diag(torch.matmul(z1, z2.T)),
        torch.diag(torch.matmul(z2, z1.T))
    ])
    logits = torch.cat([positives.unsqueeze(1), similarity_matrix], dim=1)
    labels = torch.zeros(2 * batch_size, dtype=torch.long).to(z.device)
    loss = F.cross_entropy(logits / temperature, labels)
    return loss

def pretrain(model, dataloader, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for data_dist, data_corr in dataloader:
            data_dist = data_dist.to(device)
            data_corr = data_corr.to(device)
            optimizer.zero_grad()
            z1, _ = model(data_dist)
            z2, _ = model(data_corr)
            loss = nt_xent_loss(z1, z2, temperature=0.5)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Pretraining Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f}")

def fine_tune(classifier, dataloader, criterion, optimizer, scheduler, device, epochs=20):
    classifier.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for data_dist, data_corr, labels in dataloader:
            data_dist = data_dist.to(device)
            data_corr = data_corr.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = classifier(data_dist, data_corr)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        print(f"Fine Tuning Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(dataloader):.4f}")

def evaluate_classifier(classifier, dataloader, device):
    classifier.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data_dist, data_corr, labels in dataloader:
            data_dist = data_dist.to(device)
            data_corr = data_corr.to(device)
            labels = labels.to(device)
            outputs = classifier(data_dist, data_corr)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return acc, f1
