# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 01:43:58 2024

@author: Adeka
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the data
data_path = "C:\\Users\\Adeka\\.spyder-py3\\pems04.npz"
loaded_data = np.load(data_path)
traffic_data = loaded_data['data']

# Calculate basic statistics across all sensors
mean_values = np.mean(traffic_data, axis=(0,1))
std_values = np.std(traffic_data, axis=(0,1))
min_values = np.min(traffic_data, axis=(0,1))
max_values = np.max(traffic_data, axis=(0,1))

print("\nFeature Statistics:")
print(f"Flow - Mean: {mean_values[0]:.2f}, Std: {std_values[0]:.2f}, Min: {min_values[0]:.2f}, Max: {max_values[0]:.2f}")
print(f"Occupancy - Mean: {mean_values[1]:.4f}, Std: {std_values[1]:.4f}, Min: {min_values[1]:.4f}, Max: {max_values[1]:.4f}")
print(f"Speed - Mean: {mean_values[2]:.2f}, Std: {std_values[2]:.2f}, Min: {min_values[2]:.2f}, Max: {max_values[2]:.2f}")

# Time series analysis for a single sensor (using first sensor)
plt.figure(figsize=(15, 10))

# Plot flow over time
plt.subplot(3, 1, 1)
plt.plot(traffic_data[:, 0, 0])
plt.title('Traffic Flow Over Time (Sensor 0)')
plt.ylabel('Flow')

# Plot occupancy over time
plt.subplot(3, 1, 2)
plt.plot(traffic_data[:, 0, 1])
plt.title('Occupancy Over Time (Sensor 0)')
plt.ylabel('Occupancy')

# Plot speed over time
plt.subplot(3, 1, 3)
plt.plot(traffic_data[:, 0, 2])
plt.title('Speed Over Time (Sensor 0)')
plt.ylabel('Speed')

plt.tight_layout()
plt.show()

# Distribution analysis
plt.figure(figsize=(15, 5))

# Flow distribution
plt.subplot(1, 3, 1)
sns.histplot(traffic_data[:, :, 0].flatten(), kde=True)
plt.title('Flow Distribution')
plt.xlabel('Flow')

# Occupancy distribution
plt.subplot(1, 3, 2)
sns.histplot(traffic_data[:, :, 1].flatten(), kde=True)
plt.title('Occupancy Distribution')
plt.xlabel('Occupancy')

# Speed distribution
plt.subplot(1, 3, 3)
sns.histplot(traffic_data[:, :, 2].flatten(), kde=True)
plt.title('Speed Distribution')
plt.xlabel('Speed')

plt.tight_layout()
plt.show()

# Correlation analysis
# Reshape data for correlation analysis
reshaped_data = traffic_data.reshape(-1, 3)
correlation_matrix = np.corrcoef(reshaped_data.T)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, 
            annot=True, 
            xticklabels=['Flow', 'Occupancy', 'Speed'],
            yticklabels=['Flow', 'Occupancy', 'Speed'],
            cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

# Daily patterns (assuming 5-minute intervals)
samples_per_day = 288  # 24 hours * 12 (5-minute intervals)
days = traffic_data.shape[0] // samples_per_day

# Calculate average daily pattern
daily_pattern = traffic_data[:days*samples_per_day].reshape(days, samples_per_day, 307, 3)
mean_daily_pattern = np.mean(daily_pattern, axis=(0,2))  # Average across days and sensors

# Plot average daily patterns
plt.figure(figsize=(15, 5))
time_points = np.linspace(0, 24, samples_per_day)

plt.subplot(1, 3, 1)
plt.plot(time_points, mean_daily_pattern[:, 0])
plt.title('Average Daily Flow Pattern')
plt.xlabel('Hour of Day')
plt.ylabel('Flow')

plt.subplot(1, 3, 2)
plt.plot(time_points, mean_daily_pattern[:, 1])
plt.title('Average Daily Occupancy Pattern')
plt.xlabel('Hour of Day')
plt.ylabel('Occupancy')

plt.subplot(1, 3, 3)
plt.plot(time_points, mean_daily_pattern[:, 2])
plt.title('Average Daily Speed Pattern')
plt.xlabel('Hour of Day')
plt.ylabel('Speed')

plt.tight_layout()
plt.show()

# Spatial analysis
sensor_means = np.mean(traffic_data, axis=0)  # Average across time for each sensor

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(sensor_means[:, 0], bins=30)
plt.title('Distribution of Mean Flow Across Sensors')
plt.xlabel('Mean Flow')

plt.subplot(1, 3, 2)
plt.hist(sensor_means[:, 1], bins=30)
plt.title('Distribution of Mean Occupancy Across Sensors')
plt.xlabel('Mean Occupancy')

plt.subplot(1, 3, 3)
plt.hist(sensor_means[:, 2], bins=30)
plt.title('Distribution of Mean Speed Across Sensors')
plt.xlabel('Mean Speed')

plt.tight_layout()
plt.show()