

README.md for Hybrid ASTGCN-Transformer Model

Hybrid ASTGCN-Transformer Model for Traffic Flow Prediction

This repository contains the implementation of a hybrid Attention-based Spatio-Temporal Graph Convolutional Network (ASTGCN) combined with Transformer Networks to enhance traffic flow prediction. The project addresses challenges in spatio-temporal dependencies and introduces advanced modeling techniques for Intelligent Transportation Systems (ITS).

Features
	•	Hybrid Model: Combines Graph Convolutional Networks (GCNs) for spatial dependency modeling and Transformers for temporal dependency handling.
	•	Dynamic Adjacency Matrices: Captures evolving relationships between sensors in real-time.
	•	Attention Mechanisms: Leverages spatial and temporal attention for improved accuracy and interpretability.
	•	Comprehensive Analysis: Includes preprocessing, exploratory data analysis, and benchmarking against traditional models (e.g., SARIMA, LSTM).
	•	Real-World Testing: Validates the model using the PeMS04 dataset from California’s highway traffic system.

Dataset

The project uses the PeMS04 dataset, which includes:
	•	Data from 307 sensors.
	•	Measurements every 5 minutes over three months (January-March 2018).
	•	Key metrics:
	•	Traffic flow (number of vehicles).
	•	Occupancy (road usage percentage).
	•	Speed (vehicle velocity).

Note: Due to file size constraints, the dataset is not stored in this repository. You can download it from Google Drive and place it in the data/ directory.

Installation
	1.	Clone this repository:

git clone https://github.com/dekanms/astgcntraffic.git


	2.	Navigate to the project directory:

cd your-repo-name


	3.	Install the required dependencies:

pip install -r requirements.txt

Preprocessing

The preprocessing pipeline includes:
	•	Temporal Features:
	•	Time-of-day encoding (288 intervals/day).
	•	Day-of-week one-hot encoding.
	•	Rolling statistics for pattern detection.
	•	Spatial Features:
	•	Physical distance-based adjacency computation.
	•	Feature similarity-based dynamic adjacency matrices.
	•	Gaussian kernel application for edge weights.

Model Architecture

GCN-Transformer

The GCN-Transformer serves as an intermediate step, integrating:
	•	Spatial dependencies using dual GCN layers.
	•	Temporal patterns using Transformer encoders.

ASTGCN

The ASTGCN model builds upon the GCN-Transformer by introducing:
	•	Dual Attention Mechanisms:
	•	Spatial attention for dynamic node weighting.
	•	Temporal attention for identifying critical time steps.

Performance Metrics:
	•	Mean Squared Error (MSE): 0.0020
	•	Root Mean Squared Error (RMSE): 0.0444
	•	R² Score: 0.9456

Usage

Training

Train the ASTGCN model using the preprocessed dataset:

python train.py --config config/train_config.json

Testing

Evaluate the model on the test dataset:

python test.py --config config/test_config.json

Results

Key Findings:
	•	72% RMSE reduction compared to SARIMA.
	•	Improved handling of peak traffic periods.
	•	Enhanced scalability for real-world ITS applications.

Visualization Examples:
	•	Temporal and spatial attention heatmaps.
	•	Predictions vs actual traffic flows.
	•	Training and validation loss convergence curves.

Real-World Deployment

The ASTGCN model has been tested in real-world scenarios, demonstrating:
	•	Interactive Dashboards: Node-level traffic monitoring with refresh rates of 5-60 seconds.
	•	Scalability: Robust performance across 307 sensors with distributed processing.

Contributing

Contributions are welcome! Please fork the repository and submit a pull request. For major changes, open an issue first to discuss what you’d like to contribute.

License

This project is licensed under the MIT License.

Contact

For questions or collaborations, please reach out to:
	•	Name: adekanmi oluwaseun
	•	Email: adekanmi000@gmail.com
	•	GitHub: https://github.com/dekanms/astgcntraffic

