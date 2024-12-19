# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 05:20:25 2024

@author: Adeka
"""


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import requests

# Function to retrieve predictions from the API
def get_predictions(api_url, sequence, adj_matrix):
    data = {
        "sequence": sequence,
        "adj_matrix": adj_matrix.tolist()
    }
    response = requests.post(api_url, json=data)
    response.raise_for_status()  # Raises an error for bad status
    return response.json()['prediction']

# Function to create and visualize network graph
def visualize_network_graph(adj_matrix, predictions, timestep, feature_idx=0):
    # Use from_numpy_array instead of from_numpy_matrix
    G = nx.from_numpy_array(adj_matrix)

    # Assign feature values to nodes
    for node in G.nodes():
        G.nodes[node]['feature'] = predictions[timestep, node, feature_idx]

    # Normalize feature values for color mapping
    features = [G.nodes[node]['feature'] for node in G.nodes()]
    max_feature = max(features)
    min_feature = min(features)
    norm_features = [
        (f - min_feature) / (max_feature - min_feature) if max_feature != min_feature else 0.5
        for f in features
    ]

    # Define node colors based on normalized feature values
    cmap = plt.cm.viridis
    colors = cmap(norm_features)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 12))

    # Create layout
    pos = nx.spring_layout(G, seed=42)  # Positions for all nodes

    # Draw nodes
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=50,
        node_color=norm_features,  # Use normalized features directly
        cmap=cmap,
        ax=ax
    )

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)

    # Create colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_feature, vmax=max_feature))
    sm.set_array([])  # Only needed for older versions of Matplotlib
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Flow')  # Change label based on feature

    plt.title(f'Network Graph Visualization - Timestep {timestep+1}')
    plt.axis('off')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Replace with your actual API endpoint
    API_URL = "http://localhost:8000/predict"

    # Example input data (replace with your actual data)
    sequence = [
        [
            [
                np.random.uniform(0.0, 1.0),    # Flow
                np.random.uniform(0.0, 100.0),  # Occupancy (%)
                np.random.uniform(30.0, 100.0)  # Speed (km/h)
            ]
            for _ in range(307)
        ]
        for _ in range(12)  # Assuming 12 past timesteps
    ]

    adj_matrix = np.zeros((307, 307))
    for i in range(307):
        if i > 0:
            adj_matrix[i][i-1] = 1
        if i < 306:
            adj_matrix[i][i+1] = 1

    # Optional: Verify adjacency matrix symmetry
    assert np.array_equal(adj_matrix, adj_matrix.T), "Adjacency matrix is not symmetric!"

    # Retrieve predictions
    prediction = get_predictions(API_URL, sequence, adj_matrix)
    predictions = np.array(prediction)  # Shape: (3, 307, 3)

    # Select a timestep to visualize (e.g., first timestep)
    timestep = 0  # 0-based index

    # Feature index: 0=Flow, 1=Occupancy, 2=Speed
    feature_idx = 0  # Change based on which feature you want to visualize

    # Visualize network graph
    visualize_network_graph(adj_matrix, predictions, timestep, feature_idx)
