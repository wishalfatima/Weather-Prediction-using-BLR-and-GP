Weather Prediction using Bayesian Linear Regression and Gaussian Processes
 Overview

This project implements **Bayesian Linear Regression (BLR)** and **Gaussian Process (GP)** from scratch to predict temperature across Austria based on weather station data. The models provide probabilistic predictions to issue ice warnings when there is a high likelihood of freezing temperatures.

 Features

- Bayesian Linear Regression (implemented from scratch)
- Gaussian Process Regression with RBF kernel (implemented from scratch)
- Active Learning for optimal weather station placement
- Probabilistic Ice Warning System (80% probability threshold)
- Model Comparison (BLR vs GP with different bandwidths)
- Uncertainty Analysis and visualization

Dataset

- 199 weather stations across Austria
- Training period: March 9-23, 2026 (2017 timesteps, 10-minute intervals)
- Test period: March 23 - April 1, 2026 (1439 timesteps)
- Prediction grid: 100 × 100 = 10,000 points covering Austria
- Features: Longitude, Latitude, Altitude

 Results Summary

| Model | MAE (°C) | Best For |
|-------|----------|----------|
| BLR | 2.85 | Accuracy + Speed |
| GP (Spatial Only) | 3.03 | - |
| GP (Full Features) | 2.97 | Uncertainty Estimation |

 Key Findings

- BLR outperformed GP in accuracy (MAE: 2.85°C vs 2.97°C) due to:
  - Small dataset (199 stations) favoring simpler model
  - Approximately linear temperature-altitude relationship
  - BLR's regularization preventing overfitting

- GP provides more realistic uncertainty:
  - BLR uncertainty: uniform (0.20°C everywhere)
  - GP uncertainty: location-dependent (0.75°C near stations to 5.05°C in remote Alps)

- Active Learning selected 100 optimal station locations:
  - Well spread across Austria (average distance: 380 km)
  - Covers 106% longitude, 140% latitude of grid

- Ice Warning System (80% probability threshold):
  - 13.9% of Austria requires ice warnings
  - Warning zones are 892m higher than safe zones

- Computational Trade-offs:
  - BLR: O(N·D²) - linear scaling, 0.00004s for 199 stations
  - GP: O(N³) - cubic scaling, 0.0069s for 199 stations (186x slower)

 Requirements
 numpy
h5py
matplotlib
scikit-learn
scipy
pandas
geopandas


 Installation

```bash
Clone the repository
git clone https://github.com/wishalfatima/Weather-Prediction-using-BLR-and-GP.git

Navigate to project directory
cd Weather-Prediction-using-BLR-and-GP

 Install dependencies
pip install -r requirements.txt


Usage
Place weather_data.h5 in the project folder

Run the Jupyter notebook:

bash
jupyter notebook weather_assignment.ipynb
Execute cells in order (Cell 1 to Cell 20)

Visualizations
The notebook generates the following visualizations:

Weather station locations on Austria map

Temperature trend over 2 days

BLR mean prediction and uncertainty maps

GP mean prediction and uncertainty maps (two bandwidths)

Ice warning zones (probability and risk levels)

Active learning optimal station locations

Predictions vs ground truth scatter plots

Uncertainty vs altitude and distance correlations

Author
Fatima Wishal

License
This project is for educational purposes as part of the Probabilistic AI course at University of Vienna.
