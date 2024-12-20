import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import math


class TraditionalModels:
    def __init__(self, preprocessed_data_path):
        """Initialize with preprocessed data"""
        self.load_data(preprocessed_data_path)
        self.models = {}
        self.results = {}

    def load_data(self, data_path):
        """Load the preprocessed data"""
        try:
            data = np.load(data_path, allow_pickle=True)
            self.train_data = data['train_data'].item()
            self.val_data = data['val_data'].item()
            self.test_data = data['test_data'].item()
            print("Data loaded successfully")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def plot_predictions(self, model_name, actual, predictions):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(10, 6))
        plt.plot(actual, label='Actual', color='blue', alpha=0.7)
        plt.plot(predictions, label='Predicted', color='orange', alpha=0.7)
        plt.title(f"{model_name} Predictions")
        plt.xlabel("Time Steps")
        plt.ylabel("Traffic Flow")
        plt.legend()
        plt.grid()
        plt.show()

    def train_sarima(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24)):
        """Train SARIMA model"""
        print("\nTraining SARIMA model...")
        
        try:
            # Use flow data from the first sensor
            train_flow = self.train_data['X'][:, 0, 0, 0]
            test_flow = self.test_data['X'][:, 0, 0, 0]
            
            # Fit SARIMA model
            model = SARIMAX(train_flow, 
                            order=order,
                            seasonal_order=seasonal_order)
            self.models['sarima'] = model.fit(disp=False)
            
            # Make predictions
            predictions = self.models['sarima'].forecast(len(test_flow))
            
            # Calculate metrics
            mse = mean_squared_error(test_flow, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(test_flow, predictions)
            r2 = r2_score(test_flow, predictions)
            
            self.results['sarima'] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
            
            print("SARIMA Training completed")
            print(f"MSE: {self.results['sarima']['mse']:.4f}")
            print(f"RMSE: {self.results['sarima']['rmse']:.4f}")
            print(f"MAE: {self.results['sarima']['mae']:.4f}")
            print(f"R²: {self.results['sarima']['r2']:.4f}")
            
            # Plot results
            self.plot_predictions('SARIMA', test_flow, predictions)
            
            return predictions
        
        except Exception as e:
            print(f"Error in SARIMA training: {str(e)}")

    def train_linear_regression(self, node_idx=0, feature_idx=0):
        """Train a linear regression model as a baseline"""
        print("\nTraining Linear Regression model...")
        
        try:
            # Extract single node and feature
            X_train = self.train_data['X'][:, :, node_idx, feature_idx]
            y_train = self.train_data['y'][:, :, node_idx, feature_idx]
            X_test = self.test_data['X'][:, :, node_idx, feature_idx]
            y_test = self.test_data['y'][:, :, node_idx, feature_idx]

            # Flatten data for linear regression
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            y_train_flat = y_train.reshape(y_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            y_test_flat = y_test.reshape(y_test.shape[0], -1)

            # Train linear regression model
            model = LinearRegression()
            model.fit(X_train_flat, y_train_flat)
            self.models['linear_regression'] = model

            # Make predictions
            y_pred = model.predict(X_test_flat)

            # Calculate metrics
            y_pred_1d = y_pred.flatten()
            y_test_1d = y_test_flat.flatten()

            mse = mean_squared_error(y_test_1d, y_pred_1d)
            rmse = math.sqrt(mse)
            r2 = r2_score(y_test_1d, y_pred_1d)

            self.results['linear_regression'] = {
                'mse': mse,
                'rmse': rmse,
                'r2': r2
            }

            print("Linear Regression Training completed")
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"R²: {r2:.4f}")

            # Plot predictions for the first future step
            self.plot_predictions("Linear Regression", y_test[:, 0], y_pred[:, 0])

        except Exception as e:
            print(f"Error in Linear Regression training: {str(e)}")

    def train_lstm(self, node_idx=0, feature_idx=0, epochs=20, batch_size=32):
        """Train an LSTM model"""
        print("\nTraining LSTM model...")
        
        try:
            # Extract single node and feature
            X_train = self.train_data['X'][:, :, node_idx, feature_idx]
            y_train = self.train_data['y'][:, :, node_idx, feature_idx]
            X_test = self.test_data['X'][:, :, node_idx, feature_idx]
            y_test = self.test_data['y'][:, :, node_idx, feature_idx]

            # Reshape for LSTM input
            X_train = X_train[:, :, np.newaxis]
            X_test = X_test[:, :, np.newaxis]
            y_train = y_train[:, :, np.newaxis]
            y_test = y_test[:, :, np.newaxis]

            # Build LSTM model
            model = Sequential([
                LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
                Dropout(0.2),
                Dense(y_train.shape[1])
            ])
            model.compile(optimizer='adam', loss='mse')
            self.models['lstm'] = model

            # Train model
            model.fit(X_train, y_train[:, :, 0], 
                      validation_split=0.2, 
                      epochs=epochs, 
                      batch_size=batch_size, 
                      verbose=1)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            mse = mean_squared_error(y_test[:, :, 0].flatten(), y_pred.flatten())
            rmse = math.sqrt(mse)
            r2 = r2_score(y_test[:, :, 0].flatten(), y_pred.flatten())

            self.results['lstm'] = {
                'mse': mse,
                'rmse': rmse,
                'r2': r2
            }

            print("LSTM Training completed")
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"R²: {r2:.4f}")

            # Plot predictions
            self.plot_predictions("LSTM", y_test[:, 0, 0], y_pred[:, 0])

        except Exception as e:
            print(f"Error in LSTM training: {str(e)}")


if __name__ == "__main__":
    # Use the preprocessed data path
    data_path = "C:\\Users\\Adeka\\.spyder-py3\\processed_pems04_astgcn.npz"
    
    try:
        # Initialize the class
        print("Initializing traditional models...")
        traditional_models = TraditionalModels(data_path)
        
        # Train SARIMA model
        traditional_models.train_sarima()

        # Train Linear Regression model
        traditional_models.train_linear_regression()

        # Train LSTM model
        traditional_models.train_lstm()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
