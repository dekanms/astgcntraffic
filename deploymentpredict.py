# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 19:17:04 2024

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
import os
import logging
from datetime import datetime


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

# Add logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='deployment.log'
)

def visualize_deployment_results(predictions, targets, save_dir='deployment_plots'):
    """
    Visualize deployment results with comprehensive plots
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Predictions vs Actual
    plt.figure(figsize=(15, 6))
    plt.plot(targets[0, :, 0, 0][:100], label='Actual', alpha=0.7)
    plt.plot(predictions[0, :, 0, 0][:100], label='Predicted', alpha=0.7)
    plt.title('Predictions vs Actual Traffic Flow')
    plt.xlabel('Time Steps')
    plt.ylabel('Traffic Flow')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dir}/predictions_vs_actual_{timestamp}.png')
    plt.close()
    
    # 2. Error Distribution
    errors = (predictions - targets).flatten()
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.title('Prediction Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Count')
    plt.savefig(f'{save_dir}/error_distribution_{timestamp}.png')
    plt.close()
    
    # 3. Scatter Plot
    plt.figure(figsize=(10, 10))
    plt.scatter(targets.flatten(), predictions.flatten(), alpha=0.1)
    plt.plot([0, 1], [0, 1], 'r--')  # Perfect prediction line
    plt.title('Predicted vs Actual Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.savefig(f'{save_dir}/scatter_plot_{timestamp}.png')
    plt.close()

def deploy_model(model_path, data_path, device='cpu'):
    """
    Deploy model with error handling and real-time prediction capability
    """
    try:
        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = ASTGCN().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logging.info("Model loaded successfully")
        
        # Load test data
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        data = np.load(data_path, allow_pickle=True)
        test_data = data['test_data'].item()
        adj_matrix = torch.FloatTensor(data['physical_adj']).to(device)
        logging.info(f"Test data loaded: shape {test_data['X'].shape}")
        
        # Create test dataset and loader
        test_dataset = TrafficDataset(test_data['X'], test_data['y'])
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Make predictions with progress tracking
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                try:
                    data = data.to(device)
                    output = model(data, adj_matrix)
                    all_predictions.append(output.cpu().numpy())
                    all_targets.append(target.numpy())
                    
                    if batch_idx % 10 == 0:
                        logging.info(f"Processed batch {batch_idx}/{len(test_loader)}")
                        
                except Exception as e:
                    logging.error(f"Error in batch {batch_idx}: {str(e)}")
                    continue
        
        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)
        
        # Calculate metrics
        metrics = calculate_metrics(targets, predictions)
        log_metrics(metrics)
        
        # Visualize results
        visualize_deployment_results(predictions, targets)
        
        # Save results
        save_results(predictions, targets, metrics)
        
        return predictions, targets, metrics
        
    except Exception as e:
        logging.error(f"Deployment failed: {str(e)}")
        raise

def real_time_predict(model, input_data, adj_matrix, device='cpu'):
    """
    Make real-time predictions on new data
    """
    try:
        model.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(device)
            prediction = model(input_tensor, adj_matrix)
            return prediction.cpu().numpy()
    except Exception as e:
        logging.error(f"Real-time prediction failed: {str(e)}")
        raise

def calculate_metrics(targets, predictions):
    """Calculate comprehensive metrics"""
    return {
        'mse': mean_squared_error(targets.flatten(), predictions.flatten()),
        'rmse': np.sqrt(mean_squared_error(targets.flatten(), predictions.flatten())),
        'mae': mean_absolute_error(targets.flatten(), predictions.flatten()),
        'r2': r2_score(targets.flatten(), predictions.flatten())
    }

def log_metrics(metrics):
    """Log metrics with proper formatting"""
    logging.info("\nTest Set Metrics:")
    for metric, value in metrics.items():
        logging.info(f"{metric.upper()}: {value:.4f}")

def save_results(predictions, targets, metrics):
    """Save deployment results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.savez(f'deployment_results_{timestamp}.npz',
             predictions=predictions,
             targets=targets,
             metrics=metrics)
    logging.info(f"Results saved to deployment_results_{timestamp}.npz")


if __name__ == "__main__":
    # Set paths
    model_path = "best_astgcn_model.pth"
    data_path = "C:\\Users\\Adeka\\.spyder-py3\\processed_pems04_astgcn.npz"
    
    try:
        # Deploy model and get results
        predictions, targets, metrics = deploy_model(model_path, data_path)
        
        # Print metrics clearly
        print("\nTest Set Metrics:")
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"R2 Score: {metrics['r2']:.4f}")
        
        # Create visualizations
        visualize_deployment_results(predictions, targets)
        
        print(f"\nTest data shape: {targets.shape}")
        print("Results and visualizations saved in deployment_plots directory")
        
    except Exception as e:
        print(f"Deployment error: {str(e)}")


