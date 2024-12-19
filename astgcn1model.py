# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 17:31:55 2024

@author: Adeka
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import gc

class TrafficDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class Visualizer:
    def __init__(self, save_dir='plots'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def plot_training_metrics(self, train_losses, val_losses, epoch):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Model Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.save_dir}/loss_epoch_{epoch}.png')
        plt.close()
        
    def plot_predictions(self, true_values, predictions, epoch):
        plt.figure(figsize=(12, 6))
        plt.plot(true_values[:100], label='Actual', alpha=0.7)
        plt.plot(predictions[:100], label='Predicted', alpha=0.7)
        plt.title(f'Predictions vs Actual - Epoch {epoch}')
        plt.xlabel('Time Steps')
        plt.ylabel('Traffic Flow')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.save_dir}/predictions_epoch_{epoch}.png')
        plt.close()

class ASTGCN(nn.Module):
    def __init__(self, num_nodes=307, num_features=3, seq_length=12, 
                 hidden_dim=64, num_heads=8, prediction_length=3, dropout=0.1):
        super(ASTGCN, self).__init__()
        
        # Save configuration
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.prediction_length = prediction_length
        
        # Input projection
        self.input_fc = nn.Linear(num_features, hidden_dim)
        
        # Spatial attention
        self.spatial_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.spatial_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # GCN layers
        self.gc1 = nn.Linear(hidden_dim, hidden_dim)
        self.gc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.output_fc = nn.Linear(hidden_dim, prediction_length * num_features)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj):
        batch_size = x.size(0)
        
        # Input projection
        x = self.input_fc(x)  # [batch, seq, nodes, hidden]
        
        # Spatial attention
        spatial_query = self.spatial_fc1(x)
        spatial_key = self.spatial_fc2(x)
        spatial_attention = F.softmax(
            torch.matmul(spatial_query, spatial_key.transpose(-2, -1)) 
            / np.sqrt(self.hidden_dim), dim=-1)
        x = torch.matmul(spatial_attention, x)
        
        # Temporal attention
        x = x.permute(0, 2, 1, 3)  # [batch, nodes, seq, hidden]
        original_shape = x.shape
        x = x.reshape(-1, self.seq_length, self.hidden_dim)  # [batch*nodes, seq, hidden]
        
        x, _ = self.temporal_attention(x, x, x)
        x = x.reshape(*original_shape)  # [batch, nodes, seq, hidden]
        x = x.permute(0, 2, 1, 3)  # [batch, seq, nodes, hidden]
        
        # GCN layers
        x = self.dropout(F.relu(self.gc1(x)))
        x = self.dropout(F.relu(self.gc2(x)))
        
        # Output projection
        x = x[:, -1]  # Take last time step [batch, nodes, hidden]
        x = self.output_fc(x)  # [batch, nodes, pred_len*features]
        
        # Reshape output
        x = x.view(batch_size, self.num_nodes, self.prediction_length, self.num_features)
        x = x.permute(0, 2, 1, 3)  # [batch, pred_len, nodes, features]
        
        return x

class ASTGCNTrainer:
    def __init__(self, model, device='cpu', learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)
        self.criterion = nn.MSELoss()
        self.visualizer = Visualizer()
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader, adj_matrix):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            adj_matrix = adj_matrix.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data, adj_matrix)
            loss = self.criterion(output, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}: Loss {loss.item():.4f}')
                
        return total_loss / len(train_loader)
    
    def validate(self, val_loader, adj_matrix):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                adj_matrix = adj_matrix.to(self.device)
                
                output = self.model(data, adj_matrix)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                all_predictions.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)
        
        # Calculate metrics
        mse = mean_squared_error(targets.flatten(), predictions.flatten())
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets.flatten(), predictions.flatten())
        r2 = r2_score(targets.flatten(), predictions.flatten())
        
        print("\nValidation Metrics:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2 Score: {r2:.4f}")
        
        return total_loss / len(val_loader), predictions, targets
    
    def train(self, train_loader, val_loader, adj_matrix, epochs=20):
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training
            train_loss = self.train_epoch(train_loader, adj_matrix)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, predictions, targets = self.validate(val_loader, adj_matrix)
            self.val_losses.append(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': best_val_loss,
                }, 'best_astgcn_model.pth')
                print(f"Saved new best model with validation loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
            
            # Visualizations
            if (epoch + 1) % 5 == 0:
                self.visualizer.plot_training_metrics(
                    self.train_losses, 
                    self.val_losses, 
                    epoch+1
                )
                self.visualizer.plot_predictions(
                    targets[0, :, 0, 0],
                    predictions[0, :, 0, 0],
                    epoch+1
                )
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    # Load data with confirmed path
    data_path = "C:\\Users\\Adeka\\.spyder-py3\\processed_pems04_astgcn.npz"
    data = np.load(data_path, allow_pickle=True)
    
    # Print data shapes for verification
    train_data = data['train_data'].item()
    val_data = data['val_data'].item()
    adj_matrix = data['physical_adj']
    
    print("Data shapes:")
    print(f"Train X shape: {train_data['X'].shape}")
    print(f"Train y shape: {train_data['y'].shape}")
    print(f"Adjacency matrix shape: {adj_matrix.shape}")
    
    # Create datasets
    train_dataset = TrafficDataset(train_data['X'], train_data['y'])
    val_dataset = TrafficDataset(val_data['X'], val_data['y'])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Load adjacency matrix
    adj_matrix = torch.FloatTensor(adj_matrix)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create and train model
    model = ASTGCN().to(device)
    trainer = ASTGCNTrainer(model, device=device)
    trainer.train(train_loader, val_loader, adj_matrix, epochs=20)


