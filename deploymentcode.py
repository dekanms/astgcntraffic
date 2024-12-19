# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 22:02:34 2024

@author: Adeka
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class TrafficDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

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

def deploy_model(model_path, data_path, device='cpu'):
    # Load the trained model
    model = ASTGCN().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully")
    
    # Load test data
    data = np.load(data_path, allow_pickle=True)
    test_data = data['test_data'].item()
    adj_matrix = torch.FloatTensor(data['physical_adj']).to(device)
    
    print(f"Test data shape: {test_data['X'].shape}")
    
    # Create test dataset and loader
    test_dataset = TrafficDataset(test_data['X'], test_data['y'])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Make predictions
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data, adj_matrix)
            all_predictions.append(output.cpu().numpy())
            all_targets.append(target.numpy())
    
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    
    # Calculate final metrics
    mse = mean_squared_error(targets.flatten(), predictions.flatten())
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets.flatten(), predictions.flatten())
    r2 = r2_score(targets.flatten(), predictions.flatten())
    
    print("\nTest Set Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Save predictions
    np.savez('deployment_results.npz', 
             predictions=predictions, 
             targets=targets,
             metrics={
                 'mse': mse,
                 'rmse': rmse,
                 'mae': mae,
                 'r2': r2
             })
    
    print("\nResults saved to 'deployment_results.npz'")
    return predictions, targets

if __name__ == "__main__":
    # Set paths
    model_path = "best_astgcn_model.pth"
    data_path = "C:\\Users\\Adeka\\.spyder-py3\\processed_pems04_astgcn.npz"
    
    # Deploy model
    predictions, targets = deploy_model(model_path, data_path)