LLM_GPS: Multi-Sensor Data Fusion for Trajectory Prediction Using Deep Learning
Table of Contents
Introduction
Data
Model Architecture
Installation
Usage
Results
Contributing
License
Contact
Introduction
Welcome to the LLM_GPS project! This project aims to develop a deep learning-based multi-sensor fusion model to improve trajectory prediction accuracy by combining Global Navigation Satellite System (GNSS) data with Inertial Measurement Unit (IMU) data. This model is particularly useful in environments where GNSS signals are compromised, such as urban canyons.

The project explores different deep learning architectures, including Convolutional Neural Networks (CNNs), Long Short-Term Memory (LSTM) networks, and Transformers, to achieve robust and accurate navigation solutions.

Data
This project uses both synthetic and real-world data, which are provided as CSV files in the data directory of this repository:

GNSS Data: Provides absolute position information, but can be unreliable in certain environments.
IMU Data: Provides relative positioning data, including acceleration and angular velocity, which complements GNSS data.
Ground Truth Data: Contains the actual trajectories for validation and comparison.
Data Structure
Input Features: A total of 9 features representing sensor data inputs.
Output Labels: 3 dimensions representing Longitude, Latitude, and Altitude.
Data Files
train_data.csv: Training dataset containing GNSS, IMU data, and ground truth.
validation_data.csv: Validation dataset for evaluating model performance.
test_data.csv: Test dataset to assess the model's generalization ability.
ground_truth.csv: Ground truth data used for comparison with the predicted trajectories.
Model Architecture
The model architecture involves:

Spatial Encoders (CNNs): Extracts spatial features from IMU data.
Temporal Encoders (LSTM/Transformers): Captures temporal dependencies in the sequence data.
Fully Connected Layers: Outputs the predicted 3D trajectory (Longitude, Latitude, Altitude).
Key Features
Transformer Integration: Enables advanced sequence modeling.
Residual Connections: Helps prevent vanishing gradients and improves model training.
Data Normalization: Ensures consistent input feature scaling for accurate predictions.
Installation
Follow these steps to set up the project on your local machine:

Clone the Repository:

bash
Copy code
git clone https://github.com/suryakiran594/LLM_GPS.git
cd LLM_GPS
Create a Virtual Environment:

bash
Copy code
python3 -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
Install Dependencies:

bash
Copy code
pip install -r requirements.txt
Install Additional Libraries:
If using Conda, you may also need:

bash
Copy code
conda install -c conda-forge lightgbm
Usage
Here’s how to run the project using the provided data files:

Prepare the Data:
Ensure that the CSV data files (train_data.csv, validation_data.csv, test_data.csv, ground_truth.csv) are in the data directory.

Train the Model:

python
Copy code
python train.py --config configs/train_config.yaml --data_path data/train_data.csv
Evaluate the Model:

python
Copy code
python evaluate.py --model_path saved_models/model.pth --data_path data/validation_data.csv
Generate Trajectory Plots:

python
Copy code
python plot_trajectory.py --model_path saved_models/model.pth --data_path data/validation_data.csv --ground_truth data/ground_truth.csv
Synthetic Data Generation:
Generate synthetic data for testing:

python
Copy code
python generate_synthetic_data.py --output_path data/synthetic_data.csv
Results
The model's performance is evaluated using the following metrics:

Root Mean Squared Error (RMSE)
Mean Absolute Error (MAE)
Position Error (PE)
Plots generated during evaluation show the comparison between ground truth trajectories and predicted trajectories, both in 2D and 3D spaces.

Sample Outputs
3D Trajectory Plot: Visualizes the predicted vs. actual trajectories.
Coordinate Plots: Compare ground truth vs. predicted Longitude, Latitude, and Altitude over time.
Contributing
We welcome contributions! If you’d like to contribute, please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Commit your changes (git commit -am 'Add new feature').
Push to the branch (git push origin feature-branch).
Open a Pull Request.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Contact
For questions or suggestions, feel free to reach out:

Name: Surya Kiran Pilli
Email: suryakiran594@yahoo.com
LinkedIn: linkedin.com/in/suryakirandatascientist

