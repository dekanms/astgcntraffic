# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 02:00:54 2024

@author: Adeka
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from scipy.sparse.csgraph import floyd_warshall
from scipy.spatial.distance import cdist

class ASTGCNPreprocessor:
    def __init__(self, data_path, seq_length=12, pred_length=3):
        """
        Preprocessor specifically designed for ASTGCN requirements
        Args:
            data_path: Path to PeMS04 dataset
            seq_length: Input sequence length
            pred_length: Prediction horizon
        """
        self.data_path = data_path
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.load_data()

    def load_data(self):
        """Load and structure the PeMS04 dataset"""
        loaded_data = np.load(self.data_path)
        self.traffic_data = loaded_data['data']
        self.num_timestamps, self.num_nodes, self.num_features = self.traffic_data.shape
        print(f"Loaded data shape: {self.traffic_data.shape}")

    def create_dynamic_adjacency(self):
        """
        Create dynamic adjacency matrices for spatial attention
        Incorporates both physical proximity and feature similarity
        """
        # Physical distance-based adjacency
        node_coords = np.arange(self.num_nodes).reshape(-1, 1)
        physical_dist = cdist(node_coords, node_coords)
        physical_adj = np.exp(-physical_dist ** 2 / (2 * np.std(physical_dist) ** 2))

        # Feature similarity-based adjacency
        feature_adj = np.zeros((self.num_timestamps, self.num_nodes, self.num_nodes))
        
        for t in range(self.num_timestamps):
            features = self.traffic_data[t]
            similarity = cdist(features, features, metric='correlation')
            feature_adj[t] = np.exp(-similarity ** 2 / (2 * np.std(similarity) ** 2))

        return physical_adj, feature_adj

    def create_temporal_attention_features(self):
        """
        Create features for temporal attention mechanism
        Includes time-based and pattern-based features
        """
        # Time-based features
        timestamps_per_day = 288  # 5-minute intervals
        time_of_day = np.tile(np.arange(timestamps_per_day), 
                             self.num_timestamps // timestamps_per_day + 1)[:self.num_timestamps]
        time_of_day = time_of_day / timestamps_per_day  # Normalize to [0,1]

        # Day of week (one-hot encoded)
        day_of_week = np.tile(np.arange(7), 
                             self.num_timestamps // (timestamps_per_day * 7) + 1)[:self.num_timestamps]
        day_of_week_onehot = np.eye(7)[day_of_week]

        # Pattern-based features (rolling statistics)
        pattern_features = np.zeros((self.num_timestamps, self.num_nodes, 3))
        
        for t in range(self.seq_length, self.num_timestamps):
            window = self.traffic_data[t-self.seq_length:t]
            pattern_features[t, :, 0] = np.mean(window[:, :, 0], axis=0)  # Mean flow
            pattern_features[t, :, 1] = np.std(window[:, :, 0], axis=0)   # Flow variability
            pattern_features[t, :, 2] = np.gradient(np.mean(window[:, :, 0], axis=1))[-1] # Trend

        return time_of_day, day_of_week_onehot, pattern_features

    def normalize_features(self):
        """
        Normalize features while preserving temporal and spatial patterns
        Uses separate scalers for different feature types
        """
        self.scalers = {}
        normalized_data = np.zeros_like(self.traffic_data)

        # Normalize each feature separately
        for i in range(self.num_features):
            self.scalers[f'feature_{i}'] = MinMaxScaler()
            reshaped_data = self.traffic_data[:, :, i].reshape(-1, 1)
            normalized_data[:, :, i] = self.scalers[f'feature_{i}'].fit_transform(
                reshaped_data).reshape(self.num_timestamps, self.num_nodes)

        return normalized_data

    def create_attention_masked_sequences(self):
        """
        Create sequences with attention masking for transformer
        """
        X, y = [], []
        attention_masks = []

        for i in range(len(self.traffic_data) - self.seq_length - self.pred_length):
            # Input sequence
            X.append(self.traffic_data[i:i+self.seq_length])
            # Target sequence
            y.append(self.traffic_data[i+self.seq_length:i+self.seq_length+self.pred_length])
            # Create attention mask (1 for valid positions, 0 for padding)
            mask = np.ones((self.seq_length, self.seq_length))
            mask = np.triu(mask)  # Upper triangular for causal attention
            attention_masks.append(mask)

        return np.array(X), np.array(y), np.array(attention_masks)

    def preprocess(self):
        """
        Execute complete preprocessing pipeline for ASTGCN
        """
        print("Starting ASTGCN-specific preprocessing...")

        # 1. Normalize features
        self.traffic_data = self.normalize_features()

        # 2. Create dynamic adjacency matrices
        self.physical_adj, self.feature_adj = self.create_dynamic_adjacency()

        # 3. Create temporal attention features
        self.time_features, self.day_features, self.pattern_features = \
            self.create_temporal_attention_features()

        # 4. Create sequences with attention masking
        self.X, self.y, self.attention_masks = self.create_attention_masked_sequences()

        # 5. Split data while preserving temporal order
        train_size = int(0.7 * len(self.X))
        val_size = int(0.1 * len(self.X))

        self.train_data = {
            'X': self.X[:train_size],
            'y': self.y[:train_size],
            'attention_masks': self.attention_masks[:train_size],
            'time_features': self.time_features[self.seq_length:train_size+self.seq_length],
            'day_features': self.day_features[self.seq_length:train_size+self.seq_length],
            'pattern_features': self.pattern_features[self.seq_length:train_size+self.seq_length]
        }

        self.val_data = {
            'X': self.X[train_size:train_size+val_size],
            'y': self.y[train_size:train_size+val_size],
            'attention_masks': self.attention_masks[train_size:train_size+val_size],
            'time_features': self.time_features[train_size+self.seq_length:train_size+val_size+self.seq_length],
            'day_features': self.day_features[train_size+self.seq_length:train_size+val_size+self.seq_length],
            'pattern_features': self.pattern_features[train_size+self.seq_length:train_size+val_size+self.seq_length]
        }

        self.test_data = {
            'X': self.X[train_size+val_size:],
            'y': self.y[train_size+val_size:],
            'attention_masks': self.attention_masks[train_size+val_size:],
            'time_features': self.time_features[train_size+val_size+self.seq_length:],
            'day_features': self.day_features[train_size+val_size+self.seq_length:],
            'pattern_features': self.pattern_features[train_size+val_size+self.seq_length:]
        }

        # Save processed data
        np.savez('processed_pems04_astgcn.npz',
                 train_data=self.train_data,
                 val_data=self.val_data,
                 test_data=self.test_data,
                 physical_adj=self.physical_adj,
                 feature_adj=self.feature_adj)
        
        print("ASTGCN preprocessing completed")

# Usage
preprocessor = ASTGCNPreprocessor("C:\\Users\\Adeka\\.spyder-py3\\pems04.npz")
preprocessor.preprocess()