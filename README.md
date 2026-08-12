Probabilistic Weather Forecasting with Bayesian Linear Regression and Gaussian Processes

A probabilistic weather forecasting project implementing Bayesian Linear Regression (BLR) and Gaussian Process Regression (GP) from scratch using NumPy. The project predicts temperature across Austria from weather-station observations and uses predictive uncertainty for ice-warning detection and active learning for optimal weather-station placement.

This project was completed as part of the Probabilistic Artificial Intelligence course at the University of Vienna.

Overview

The project investigates how different probabilistic regression models can be used for spatial temperature prediction across Austria.

The main objectives are:

Implement Bayesian Linear Regression from scratch.
Implement Gaussian Process Regression with an RBF kernel from scratch.
Predict temperature at 10,000 locations across Austria.
Compare BLR and GP in terms of accuracy and uncertainty.
Develop a probabilistic ice-warning system.
Use active learning to identify informative locations for new weather stations.
Analyze how spatial distance and altitude affect predictive uncertainty.
Evaluate the computational trade-off between BLR and GP.

The project uses observations from 199 weather stations across Austria.

Project Structure
Weather-Prediction-using-BLR-and-GP/
│
├── weather_assignment.ipynb
├── requirements.txt
├── weather_data.h5
└── README.md

Note: The dataset is not included in the GitHub repository if its size or distribution restrictions prevent it from being uploaded. See the Dataset section below.

Dataset

The dataset contains temperature observations from 199 weather stations across Austria.

Training Data
2,017 time steps
199 weather stations
10-minute temporal resolution
Training period: 9 March – 23 March 2026
Test Data
1,439 time steps
199 weather stations
Test period: 23 March – 1 April 2026
Spatial Prediction Grid

Predictions are generated over a regular grid covering Austria:

Property	Value
Grid size	100 × 100
Total grid points	10,000
Longitude range	9.000° – 17.500°
Latitude range	46.000° – 49.500°

For each grid point, three features are used:

Longitude
Latitude
Altitude
Methodology

The project consists of four main components.

1. Bayesian Linear Regression

Bayesian Linear Regression is implemented from scratch using NumPy.

The model assumes a Gaussian prior over the weights:

w∼N(0,α
−1
I)

and a Gaussian likelihood:

y∼N(Xw,β
−1
I)

The posterior parameters are:

S
N
	​

=(αI+βX
T
X)
−1
m
N
	​

=βS
N
	​

X
T
y

For each prediction location, the model provides both:

Predictive mean
Predictive variance

The input features are standardized using StandardScaler.

2. Gaussian Process Regression

A Gaussian Process is implemented from scratch using NumPy with an RBF kernel:

k(x,x
′
)=σ
f
2
	​

exp(−
2ℓ
2
∥x−x
′
∥
2
	​

)

Two bandwidths were investigated:

Bandwidth	Behaviour
ℓ = 0.3	Local, highly sensitive spatial predictions
ℓ = 1.5	Smoother, more global spatial predictions

The GP provides:

Predictive mean
Predictive variance
Location-dependent uncertainty

The larger bandwidth, ℓ = 1.5, produced smoother and more spatially coherent predictions and was selected for the final ice-warning and active-learning analyses.

Bayesian Linear Regression vs Gaussian Process

The models were evaluated on the test data using Mean Squared Error (MSE) and Mean Absolute Error (MAE).

Model	MSE (°C²)	MAE (°C)
BLR	12.54	2.85
GP — Spatial Only	16.35	3.16
GP — Full Features	13.00	2.99
Main finding

BLR achieved the lowest MAE in this experiment.

This is likely related to the approximately linear relationship between temperature and the geographic features used by the model, together with the relatively small number of training stations.

However, accuracy was not the only important consideration.

Predictive Uncertainty

A major difference between BLR and GP is how they represent uncertainty.

Model	Average Uncertainty
BLR	~0.20°C
GP — Spatial Only	~0.73°C
GP — Full Features	~0.77°C

BLR produced nearly uniform uncertainty across Austria.

In contrast, GP uncertainty varied spatially according to the availability of nearby observations.

Regions close to weather stations generally had lower uncertainty, while remote areas had substantially higher uncertainty.

For the GP analysis, uncertainty ranged approximately from 0.72°C to 4.76°C.

The correlation between distance to the nearest weather station and predictive uncertainty was approximately 0.92, while altitude had only a weak correlation of approximately −0.11.

This indicates that, in this experiment, distance to existing observations was a much stronger driver of uncertainty than altitude.

Ice Warning System

A probabilistic ice-warning system was developed using Gaussian Process predictions.

For each grid location, the GP produces a Gaussian temperature distribution:

T∼N(μ,σ
2
)

The probability of freezing is then calculated as:

P(T≤0
∘
C)

A warning is issued when:

P(T≤0
∘
C)≥0.8
Warning Results

Both GP bandwidths were evaluated.

GP bandwidth	Warning grid points	Percentage
ℓ = 0.3	361	3.6%
ℓ = 1.5	2,156	21.6%

The ℓ = 1.5 model was selected for the final warning analysis because it produced more spatially coherent warning regions.

Final Risk Classification
Risk level	Probability	Grid points	Proportion
High	> 80%	2,156	21.6%
Medium	20–80%	792	7.9%
Low	≤ 20%	7,052	70.5%

The high-risk areas were concentrated mainly in colder and higher-altitude regions, particularly mountainous areas.

Active Learning for Weather Station Placement

An active-learning strategy was implemented to identify 100 informative locations for potential weather-station placement.

The initial candidate set consisted of the 10,000 grid points. Only locations inside Austria were considered for realistic station placement.

The greedy procedure repeatedly:

Computes the GP predictive variance at candidate locations.
Selects the location with the highest variance.
Adds that location to the selected set.
Updates the GP posterior.
Repeats until 100 locations have been selected.

The objective is to place new stations where additional observations would provide the most information.

Bandwidth Comparison
Metric	ℓ = 0.3	ℓ = 1.5
Average station distance	276 km	316 km
Average altitude	1,271 m	1,239 m
Spatial coverage	Moderate	High
Sensitivity to local noise	Higher	Lower
National-scale suitability	Moderate	Better

The ℓ = 1.5 model was selected for the final active-learning result because it produced more uniform and coherent spatial coverage across Austria.

The selected locations were distributed across western mountainous regions, central Austria, and eastern lowlands.

Effect of Altitude

Altitude was investigated as an important geographic feature because of the strong topographic variation across Austria.

For mountainous regions above 1,500 m:

Model	Predictive uncertainty
GP Spatial Only	1.0550°C
GP Full Features	1.3485°C

Including altitude improved the mean prediction, but in this experiment it increased predictive variance in mountainous regions by approximately 27.8%.

This can occur because adding altitude changes the geometry of the feature space used by the RBF kernel. Large altitude differences can make locations appear less similar under the kernel, reducing the amount of information transferred between observations.

Computational Comparison

BLR and GP have significantly different computational costs.

Bayesian Linear Regression

BLR primarily operates on the feature dimension and scales approximately as:

O(ND
2
)

where:

N = number of observations
D = number of features
Gaussian Process

GP requires operations involving the N×N kernel matrix and therefore has approximately cubic complexity:

O(N
3
)

In the timing experiment, GP was approximately 135× slower than BLR with 199 weather stations.

Model	Computational cost	Main advantage
BLR	Lower	Fast and accurate in this experiment
GP	Higher	Flexible predictions and spatial uncertainty
Key Findings
1. BLR achieved the best predictive accuracy

BLR achieved an MAE of 2.85°C, outperforming both GP configurations tested.

2. GP provides more informative uncertainty

Although GP had higher average uncertainty, its uncertainty was spatially varying and strongly related to the availability of nearby observations.

3. The larger GP bandwidth performed better spatially

The ℓ = 1.5 configuration produced smoother and more physically plausible spatial predictions than ℓ = 0.3.

4. Distance to observations strongly affects uncertainty

The correlation between nearest-station distance and uncertainty was approximately 0.92, indicating that data coverage is a major determinant of prediction confidence.

5. Altitude improves the mean prediction but can increase uncertainty

Adding altitude helped represent Austria's topography, but it increased predictive variance in high-altitude regions in the tested GP configuration.

6. Active learning identifies high-value locations

The active-learning procedure placed candidate stations in areas where the GP was most uncertain, producing broad spatial coverage rather than simply clustering stations around existing observations.

Visualizations

The notebook produces visualizations covering:

Weather-station locations across Austria
Temperature trends at individual stations
BLR temperature prediction maps
BLR predictive uncertainty
GP mean predictions for different bandwidths
GP uncertainty maps
Ice-warning probability maps
Ice-risk classification
Observed station temperatures
Active-learning station locations
Model predictions versus observations
Uncertainty versus altitude
Uncertainty versus distance to the nearest station
How to Run
1. Clone the repository
git clone https://github.com/wishalfatima/Weather-Prediction-using-BLR-and-GP.git
2. Enter the project directory
cd Weather-Prediction-using-BLR-and-GP
3. Install dependencies
pip install -r requirements.txt
4. Add the dataset

Place the required HDF5 dataset in the project directory using the expected filename:

weather_data.h5
5. Run the notebook
jupyter notebook weather_assignment.ipynb

Execute the notebook cells in order.

Implementation

The main models were implemented from scratch rather than relying on ready-made BLR or GP regression implementations.

Implemented components
Bayesian Linear Regression
Gaussian Process Regression
RBF kernel
Predictive mean and variance calculations
Probabilistic ice-warning system
Greedy active-learning algorithm
Spatial uncertainty analysis
Model comparison and evaluation

The project uses common scientific Python libraries for data handling, preprocessing, numerical computation, and visualization.

Technologies
Python
NumPy
SciPy
Pandas
Scikit-learn
Matplotlib
GeoPandas
h5py
Jupyter Notebook
Repository

GitHub:
https://github.com/wishalfatima/Weather-Prediction-using-BLR-and-GP

Author

Wishal Fatima

University of Vienna
Master's Programme in Computer Science

This project was completed as part of the Probabilistic Artificial Intelligence course at the University of Vienna.

License

This project is intended for educational and academic purposes.
