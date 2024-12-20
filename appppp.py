# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 23:49:37 2024

@author: Adeka
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict
import uvicorn

# ASTGCN Model definition
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

# Data model for API input
class TrafficData(BaseModel):
    sequence: List[List[List[float]]]  # [timesteps, nodes, features]
    adj_matrix: List[List[float]]      # [nodes, nodes]

app = FastAPI(title="Traffic Prediction API",
             description="API for real-time traffic prediction using ASTGCN model")

# Load the trained model
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ASTGCN().to(device)
    checkpoint = torch.load('best_astgcn_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# Initialize model
print("Loading model...")
model = load_model()
print("Model loaded successfully")

@app.post("/predict", response_model=Dict[str, List])
async def predict_traffic(data: TrafficData):
    try:
        # Convert input data to tensors
        sequence = torch.FloatTensor(data.sequence).unsqueeze(0)  # Add batch dimension
        adj_matrix = torch.FloatTensor(data.adj_matrix)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(sequence, adj_matrix)
            
        return {
            "prediction": prediction.squeeze(0).numpy().tolist(),
            "shape": list(prediction.shape[1:])  # Return predicted timesteps, nodes, features
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def get_model_info():
    return {
        "model_type": "ASTGCN",
        "input_sequence_length": 12,
        "prediction_horizon": 3,
        "features": ["flow", "occupancy", "speed"],
        "performance_metrics": {
            "RMSE": 0.0424,
            "R2": 0.9812,
            "MAE": 0.0278
        }
    }

# API health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    print("Starting API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)