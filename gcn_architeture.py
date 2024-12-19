# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:07:33 2024

@author: Adeka
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import os
import json
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import gc

gc.enable()

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        batch_size, seq_length, num_nodes, _ = x.size()
        # Feature transformation
        x = x.contiguous().view(-1, self.in_features)  
        x = torch.matmul(x, self.weight)  
        
        # Reshape
        x = x.view(batch_size * seq_length, num_nodes, self.out_features)
        
        # Adjacency for all sequences in batch
        adj_batch = adj.unsqueeze(0).expand(batch_size * seq_length, -1, -1)
        
        # Graph convolution
        output = torch.bmm(adj_batch, x)
        
        # Reshape to original dims
        output = output.view(batch_size, seq_length, num_nodes, self.out_features)
        
        # Add bias
        output = output + self.bias.view(1, 1, 1, -1)
        
        return output


class GCNTransformer(nn.Module):
    def __init__(self, num_nodes=307, num_features=3, seq_length=12, hidden_dim=64, prediction_length=3):
        super(GCNTransformer, self).__init__()
        
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.prediction_length = prediction_length
        
        # GCN layers
        self.gc1 = GraphConvolution(num_features, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim)
        
        # Transformer prep
        self.pre_transformer = nn.Linear(num_nodes * hidden_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, num_nodes * num_features)
        
        self.hparams = {
            'num_nodes': num_nodes,
            'num_features': num_features,
            'seq_length': seq_length,
            'hidden_dim': hidden_dim,
            'prediction_length': prediction_length
        }
    
    def forward(self, x, adj):
        batch_size = x.size(0)
        
        # GCN extraction
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, 0.1, training=self.training)
        x = F.relu(self.gc2(x, adj))
        
        # Reshape for Transformer
        x = x.reshape(batch_size, self.seq_length, -1)
        x = self.pre_transformer(x)
        
        # Transformer
        x = self.transformer(x)
        
        # Select last prediction_length steps
        x = x[:, -self.prediction_length:, :]
        
        # Output projection
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(batch_size, self.prediction_length, self.num_nodes, self.num_features)
        
        return x


class TrafficDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
        assert self.X.dim() == 4, f"Expected X to have 4 dimensions, got {self.X.dim()}"
        assert self.y.dim() == 4, f"Expected y to have 4 dimensions, got {self.y.dim()}"
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TrafficPredictor:
    def __init__(self, data_path, batch_size=16, hidden_dim=32, prediction_length=3):
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.prediction_length = prediction_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_data(data_path)
        self.setup_model()
        self.setup_training()
    
    def normalize_adjacency(self, adj_matrix):
        adj_matrix = adj_matrix + np.eye(adj_matrix.shape[0], dtype=adj_matrix.dtype)
        rowsum = adj_matrix.sum(1)
        degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5).flatten())
        adj_matrix = degree_mat_inv_sqrt @ adj_matrix @ degree_mat_inv_sqrt
        return adj_matrix

    def load_data(self, data_path):
        data = np.load(data_path, allow_pickle=True)
        train_data = data['train_data'].item()
        val_data = data['val_data'].item()
        test_data = data['test_data'].item()
        
        self.train_loader = DataLoader(
            TrafficDataset(train_data['X'], train_data['y']),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        self.val_loader = DataLoader(
            TrafficDataset(val_data['X'], val_data['y']),
            batch_size=self.batch_size
        )
        
        self.test_loader = DataLoader(
            TrafficDataset(test_data['X'], test_data['y']),
            batch_size=self.batch_size
        )
        
        self.num_nodes = train_data['X'].shape[2]
        self.num_features = train_data['X'].shape[3]
        self.seq_length = train_data['X'].shape[1]
        
        adj_matrix = data['physical_adj'].astype(np.float32)
        self.adj_matrix = torch.FloatTensor(self.normalize_adjacency(adj_matrix)).to(self.device)
        
        del data
        gc.collect()
    
    def setup_model(self):
        self.model = GCNTransformer(
            num_nodes=self.num_nodes,
            num_features=self.num_features,
            seq_length=self.seq_length,
            hidden_dim=self.hidden_dim,
            prediction_length=self.prediction_length
        ).to(self.device)
    
    def setup_training(self):
        self.criterion = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=3,
            factor=0.5
        )
    
    def train(self, epochs=20, early_stopping_patience=5):
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for X, y in self.train_loader:
                self.optimizer.zero_grad()
                X = X.to(self.device)
                y = y.to(self.device)
                
                output = self.model(X, self.adj_matrix)
                loss = self.criterion(output, y)
                
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                
                del X, y, output, loss
                torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
            
            train_loss /= len(self.train_loader)
            train_losses.append(train_loss)
            
            val_loss = self.validate()
            val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model(epoch, val_loss)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print("Early stopping triggered")
                    break
            
            if (epoch + 1) % 5 == 0:
                self.plot_losses(train_losses, val_losses)
            
            gc.collect()
    
    def validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in self.val_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                
                output = self.model(X, self.adj_matrix)
                val_loss += self.criterion(output, y).item()
                
                del X, y, output
                torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
        
        return val_loss / len(self.val_loader)
    
    def save_model(self, epoch, loss):
        os.makedirs('checkpoints', exist_ok=True)
        save_path = 'checkpoints/best_model.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'hparams': self.model.hparams
        }, save_path)
    
    def plot_losses(self, train_losses, val_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Model Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'loss_plot_{len(train_losses)}.png')
        plt.close()

    def evaluate_test(self):
        self.model.eval()
        preds = []
        targets = []
        with torch.no_grad():
            for X, y in self.test_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                output = self.model(X, self.adj_matrix)
                preds.append(output.cpu().numpy())
                targets.append(y.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)

        # Compute MSE
        mse = mean_squared_error(targets.reshape(-1), preds.reshape(-1))
        # Compute RMSE
        rmse = math.sqrt(mse)
        # Compute R2
        r2 = r2_score(targets.reshape(-1), preds.reshape(-1))

        print(f"Test MSE: {mse:.4f}")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test R²: {r2:.4f}")


if __name__ == "__main__":
    predictor = TrafficPredictor(
        data_path="C:\\Users\\Adeka\\.spyder-py3\\processed_pems04_astgcn.npz",
        batch_size=16,
        hidden_dim=32,
        prediction_length=3
    )
    predictor.train(epochs=20, early_stopping_patience=5)
    predictor.evaluate_test()  # Call evaluation after training completes
