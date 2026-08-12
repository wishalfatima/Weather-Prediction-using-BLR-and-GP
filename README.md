# Probabilistic Weather Forecasting with Bayesian Linear Regression and Gaussian Processes

**By Wishal Fatima**
University of Vienna — Probabilistic Artificial Intelligence

---

## Overview

This project investigates **probabilistic temperature prediction across Austria** using two Bayesian machine learning approaches implemented from scratch:

* **Bayesian Linear Regression (BLR)**
* **Gaussian Process Regression (GP)** with an RBF kernel

The models use geographical information from weather stations — **longitude, latitude, and altitude** — to predict temperature at unseen locations across Austria.

In addition to temperature prediction, the project investigates:

* Predictive uncertainty
* Gaussian Process bandwidth selection
* Probabilistic ice-warning detection
* Active learning for optimal weather-station placement
* The effect of altitude on spatial predictions
* Accuracy and computational trade-offs between BLR and GP

The project was completed as part of the **Probabilistic Artificial Intelligence course at the University of Vienna**.

---

# 1. Dataset

The dataset contains weather observations from **199 weather stations across Austria**.

### Training Data

* **2,017 time steps**
* **199 weather stations**
* 10-minute measurement intervals
* Training period: **9 March – 23 March 2026**

### Test Data

* **1,439 time steps**
* **199 weather stations**
* Test period: **23 March – 1 April 2026**

### Prediction Grid

Spatial predictions were generated over a regular grid covering Austria:

* Grid shape: **100 × 100**
* Total prediction locations: **10,000**
* Longitude range: **9.000° – 17.500°**
* Latitude range: **46.000° – 49.500°**

### Weather Station Features

Three geographical features were extracted:

| Feature   | Description               |
| --------- | ------------------------- |
| Longitude | East–west position        |
| Latitude  | North–south position      |
| Altitude  | Elevation above sea level |

The station feature matrix therefore has shape:

$$
199 \times 3
$$

The prediction grid has shape:

$$
10000 \times 3
$$

---

# 2. Data Exploration

## Weather Station Distribution

The weather stations were visualized geographically across Austria.

The stations are broadly distributed throughout the country, although station density varies between regions.

This spatial distribution is important because the availability of nearby observations directly influences the uncertainty of Gaussian Process predictions.

---

## Temperature Example — Freistadt

A 48-hour temperature window was examined for **station 5 (Freistadt)**.

The observed temperature showed clear daily cycles:

* Temperatures increase during the daytime.
* Temperatures decrease during the night.
* Temperatures occasionally fall below 0°C.
* Freezing periods are therefore relevant for the ice-warning task.

### Temperature Statistics

| Metric                    |     Value |
| ------------------------- | --------: |
| Minimum temperature       |   -0.60°C |
| Maximum temperature       |   16.40°C |
| Average temperature       |    6.37°C |
| Temperature range         |   17.00°C |
| Time below 0°C            | 3.2 hours |
| Percentage below freezing |      6.6% |

The average temperature was almost identical on the two examined days:

| Day   | Average Temperature |
| ----- | ------------------: |
| Day 1 |              6.35°C |
| Day 2 |              6.39°C |

Difference:

$$
0.04^\circ C
$$

---

# 3. Feature Engineering

## Spatial Features

The three input features used by the models were:

* Longitude
* Latitude
* Altitude

The observed station ranges were:

| Feature   | Minimum | Maximum |
| --------- | ------: | ------: |
| Longitude |  9.610° | 16.845° |
| Latitude  | 46.444° | 48.955° |
| Altitude  |   116 m |  3437 m |

---

## Feature Normalization

The input features were standardized using `StandardScaler`.

The scaler was fitted using the weather-station features and then applied to the prediction grid.

This ensures that longitude, latitude, and altitude are placed on comparable numerical scales before being used by the models.

---

## Prediction Grid

A regular grid containing 10,000 locations was created to generate spatial temperature predictions across Austria.

For every grid point, the following features were extracted:

$$
X_* =
[\text{longitude},\text{latitude},\text{altitude}]
$$

The resulting grid feature matrix has shape:

$$
10000 \times 3
$$

---

## Historical Temperature Features

Although the dataset contains temperature time series, explicit historical temperature features were **not** engineered for the main spatial prediction task.

Instead, a single time step was used as the target variable.

Therefore, the models primarily learn the spatial relationship between:

* Longitude
* Latitude
* Altitude
* Temperature

---

# 4. Bayesian Linear Regression

## Model

Bayesian Linear Regression (BLR) was implemented **from scratch using NumPy**.

A zero-mean isotropic Gaussian prior was placed over the model weights:

$$
w \sim \mathcal{N}(0,\alpha^{-1}I)
$$

where:

* $w$ is the weight vector.
* $\alpha$ controls the strength of the prior.
* $I$ is the identity matrix.

The likelihood is modeled as:

$$
y \sim \mathcal{N}(Xw,\beta^{-1}I)
$$

where:

* $X$ is the feature matrix.
* $y$ is the temperature target.
* $\beta$ is the noise precision.

---

## Posterior Distribution

The posterior covariance matrix is:

$$
S_N =
\left(
\alpha I+\beta X^T X
\right)^{-1}
$$

The posterior mean is:

$$
m_N =
\beta S_N X^T y
$$

---

## Training Setup

The BLR model was trained using:

| Property              | Value                         |
| --------------------- | ----------------------------- |
| Input features        | Longitude, Latitude, Altitude |
| Training samples      | 199 stations                  |
| Target                | Temperature                   |
| Target timestep       | 9 March 2026, 00:00           |
| Feature normalization | StandardScaler                |

---

## Prediction

For a new location $x_*$, the predictive mean is:

$$
\mu(x_*) = x_*^T m_N
$$

The predictive variance is:

$$
\sigma^2(x_*) =
\frac{1}{\beta}
+
x_*^T S_N x_*
$$

Therefore, BLR provides both:

* Expected temperature
* Predictive uncertainty

---

## BLR Prediction Characteristics

BLR produces smooth temperature predictions across Austria because it assumes a global linear relationship between the geographical features and temperature.

The resulting predictions show:

* Lower temperatures in mountainous regions.
* Higher temperatures in lower-elevation regions.
* A relatively smooth spatial temperature field.

However, BLR predictive uncertainty is nearly uniform across the prediction grid.

The observed uncertainty was approximately:

$$
0.20^\circ C
$$

This occurs because BLR uses a global linear model and does not explicitly model spatial distance from observations.

---

# 5. Gaussian Process Regression

## Model

Gaussian Process Regression was implemented **from scratch using NumPy**.

An RBF (Radial Basis Function) kernel was used:

$$
k(x,x') =
\sigma_f^2
\exp\left(
-\frac{|x-x'|^2}{2\ell^2}
\right)
$$

where:

* $\sigma_f^2$ is the signal variance.
* $\ell$ is the length scale or bandwidth.
* $x$ and $x'$ are two input locations.

Two bandwidths were investigated:

| Bandwidth    | Description                                   |
| ------------ | --------------------------------------------- |
| $\ell = 0.3$ | Small bandwidth; highly local predictions     |
| $\ell = 1.5$ | Large bandwidth; smoother spatial predictions |

---

## GP Predictive Mean

The GP predictive mean is:

$$
\mu(x_*) =
K(x_*,X)
\left[
K(X,X)+\sigma_n^2 I
\right]^{-1}
y
$$

where:

* $K(X,X)$ is the training kernel matrix.
* $K(x_*,X)$ is the kernel between the new point and training locations.
* $\sigma_n^2$ is the noise variance.
* $y$ is the vector of observed temperatures.

---

## GP Predictive Variance

The predictive variance is:

$$
\sigma^2(x_*) =
k(x_*,x_*)
----------

K(x_*,X)
\left[
K(X,X)+\sigma_n^2 I
\right]^{-1}
K(X,x_*)
$$

Unlike BLR, the GP predictive variance depends on the relationship between the prediction location and the observed training locations.

This allows GP uncertainty to vary spatially.

---

# 6. Gaussian Process with Small Bandwidth

## $\ell = 0.3$

The small bandwidth produces highly localized predictions.

### Mean Prediction

The resulting temperature map contains many small warm and cold regions.

This occurs because each weather station has influence over only a relatively small spatial neighborhood.

The model can capture local variations, but it is also more sensitive to noise and sparse observations.

### Uncertainty

The uncertainty increases strongly in areas that are far from weather stations.

In remote mountainous areas, uncertainty can reach approximately:

$$
5^\circ C
$$

This indicates that the model has little information available for those locations.

---

# 7. Gaussian Process with Large Bandwidth

## $\ell = 1.5$

The larger bandwidth produces smoother and more spatially coherent predictions.

### Mean Prediction

The predicted temperature map shows:

* Colder mountainous regions, especially in the Alps.
* Warmer lowland regions.
* Smooth transitions between neighboring locations.

The larger bandwidth therefore provides a better balance between local flexibility and spatial smoothness.

### Uncertainty

The uncertainty varies spatially:

* Lower uncertainty near weather stations.
* Higher uncertainty in data-sparse regions.
* Higher uncertainty near remote borders and mountainous areas.

The large-bandwidth GP therefore provides more interpretable uncertainty estimates than BLR.

---

# 8. BLR vs Gaussian Process

## Prediction Flexibility

| Property               | BLR          | GP ($\ell=0.3$) | GP ($\ell=1.5$) |
| ---------------------- | ------------ | --------------- | --------------- |
| Prediction flexibility | Low          | High            | Medium          |
| Spatial smoothness     | High         | Low             | High            |
| Local variation        | Limited      | Strong          | Moderate        |
| Uncertainty variation  | Near-uniform | Strong          | Moderate        |

---

## Uncertainty Comparison

BLR produces approximately constant uncertainty:

$$
\approx 0.20^\circ C
$$

The GP models provide location-dependent uncertainty.

For example, the small-bandwidth model can produce very high uncertainty in areas far from observations, whereas the larger bandwidth produces smoother uncertainty patterns.

The observed average uncertainties in the final model comparison were:

| Model            | Average Test Uncertainty |
| ---------------- | -----------------------: |
| BLR              |                   0.20°C |
| GP Spatial Only  |                   0.73°C |
| GP Full Features |                   0.77°C |

---

# 9. Probabilistic Ice Warning System

## Method

The Gaussian Process was used to build a probabilistic ice-warning system.

The model predicts a Gaussian temperature distribution for every grid point:

$$
T \sim \mathcal{N}(\mu,\sigma^2)
$$

The probability of freezing is calculated as:

$$
P(T \leq 0^\circ C)
$$

A warning is issued when:

$$
P(T \leq 0^\circ C) \geq 0.8
$$

Therefore, the system does not simply ask whether the predicted temperature is below zero.

Instead, it considers the **uncertainty of the prediction**.

---

## Test-Time Observation

The ice-warning experiment used:

`test[0, :]`

corresponding to:

**23 March 2026 at 00:10**

The 199 weather-station observations were used as evidence for the GP prediction.

---

## Bandwidth Comparison

Both bandwidths were tested:

| Metric              | $\ell=0.3$ |    $\ell=1.5$ |
| ------------------- | ---------: | ------------: |
| Warning grid points |        361 |         2,156 |
| Warning proportion  |       3.6% |         21.6% |
| Warning zones       | Fragmented | More coherent |

The $\ell=1.5$ model was selected for the final warning map because it produced smoother and more spatially coherent warning zones.

---

## Final Risk Classification

Using $\ell=1.5$:

| Risk Level | Probability        | Grid Points | Proportion |
| ---------- | ------------------ | ----------: | ---------: |
| High       | $P > 80%$          |       2,156 |      21.6% |
| Medium     | $20% < P \leq 80%$ |         792 |       7.9% |
| Low        | $P \leq 20%$       |       7,052 |      70.5% |

The high-risk areas are concentrated mainly in colder and higher-altitude regions.

The low-risk areas are mainly located in warmer lowland regions.

---

# 10. Active Learning for Weather Station Placement

## Objective

The project also investigates how a limited number of new weather stations could be placed optimally.

Instead of selecting locations randomly, an **active learning** strategy was used to identify locations where additional observations would provide the most information.

---

## Candidate Locations

The initial candidate set consisted of the 10,000 grid points.

Only points falling within Austria's geographical boundary were retained as realistic station locations.

---

## Greedy Selection Strategy

The algorithm selects 100 locations iteratively.

At each iteration:

1. Compute the GP predictive variance at all remaining candidate locations.
2. Select the location with the highest predictive variance.
3. Add that location to the selected station set.
4. Update the GP posterior.
5. Repeat until 100 locations have been selected.

The strategy therefore prioritizes locations where the model is currently most uncertain.

---

## Bandwidth Comparison

Two GP models were evaluated.

| Property                   | $\ell=0.3$ | $\ell=1.5$   |
| -------------------------- | ---------- | ------------ |
| Spatial coverage           | Moderate   | High         |
| Station spacing            | Tighter    | More uniform |
| Sensitivity to local noise | Higher     | Lower        |
| National-scale suitability | Moderate   | Better       |

The larger bandwidth was selected for the final station-placement experiment.

---

## Quantitative Results

| Metric                   | $\ell=0.3$ | $\ell=1.5$ |
| ------------------------ | ---------: | ---------: |
| Average station distance |     276 km |     316 km |
| Average altitude         |    1,271 m |    1,239 m |

The selected locations are distributed across:

* Western mountainous regions
* Central Austria
* Eastern lowlands

The resulting distribution demonstrates the main principle of uncertainty-based active learning: once a location is selected, nearby locations become less uncertain, encouraging subsequent selections elsewhere.

---

# 11. Model Comparison

Predictive performance was evaluated using:

* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* Predictive uncertainty

The final comparison was:

| Model              | MSE (°C²) | MAE (°C) |
| ------------------ | --------: | -------: |
| BLR                |     12.54 |     2.85 |
| GP — Spatial Only  |     16.35 |     3.16 |
| GP — Full Features |     13.00 |     2.99 |

---

## Accuracy Findings

BLR achieved the lowest MAE:

$$
MAE = 2.85^\circ C
$$

The GP with all geographical features achieved:

$$
MAE = 2.99^\circ C
$$

The spatial-only GP achieved:

$$
MAE = 3.16^\circ C
$$

Although GP is more flexible, BLR performed better for this particular prediction setting.

A likely explanation is that the spatial temperature pattern at the evaluated timestep is approximately linear with respect to longitude, latitude, and altitude.

The GP hyperparameters were also manually selected rather than optimized automatically.

---

# 12. Effect of Altitude

Altitude was investigated because Austria contains large mountainous regions where elevation strongly affects temperature.

The analysis compared predictive uncertainty in regions above:

$$
1500,m
$$

The observed uncertainties were:

| Model            | Mountainous-region uncertainty |
| ---------------- | -----------------------------: |
| GP Spatial Only  |                       1.0550°C |
| GP Full Features |                       1.3485°C |

Including altitude therefore increased predictive variance by approximately:

$$
27.8%
$$

This does **not** mean that altitude is harmful to the mean prediction.

Instead, adding altitude increases the dimensionality of the input space. In mountainous regions, altitude can change significantly over relatively short geographic distances, causing the RBF kernel to assign lower similarity between observations and prediction locations.

Thus:

* Altitude improves the model's ability to represent temperature variation.
* However, under the chosen GP configuration, it can also increase predictive uncertainty in mountainous regions.

---

# 13. Computational Trade-offs

BLR and GP have significantly different computational requirements.

## Bayesian Linear Regression

For feature dimension $D$ and number of observations $N$, the main computational cost is approximately:

$$
O(ND^2)
$$

With only three input features, BLR is extremely efficient.

---

## Gaussian Process

GP regression requires operations on the $N \times N$ kernel matrix.

The standard computational complexity is approximately:

$$
O(N^3)
$$

for training due to matrix factorization/inversion.

---

## Experimental Comparison

The timing experiment showed that GP was substantially more expensive than BLR.

At 199 weather stations, GP was approximately **135 times slower** than BLR in the measured experiment.

Therefore:

| Aspect                         | BLR       | GP        |
| ------------------------------ | --------- | --------- |
| Computational cost             | Low       | High      |
| Scaling with observations      | Efficient | Expensive |
| Non-linear relationships       | Limited   | Yes       |
| Spatial uncertainty            | Limited   | Strong    |
| Location-dependent uncertainty | No        | Yes       |

---

# 14. Uncertainty Analysis

The GP uncertainty analysis showed substantial spatial variation.

Observed uncertainty values ranged approximately from:

$$
0.72^\circ C
$$

to:

$$
4.76^\circ C
$$

High uncertainty was mainly observed in western Austria, while lower uncertainty was concentrated in regions with better observational coverage.

---

## Distance to Weather Stations

The correlation between uncertainty and distance to the nearest weather station was:

$$
r = 0.92
$$

This indicates a strong relationship between observational coverage and predictive uncertainty.

In contrast, the correlation between uncertainty and altitude was:

$$
r = -0.11
$$

This indicates only a weak relationship between altitude and predictive uncertainty in this analysis.

These results support the active-learning strategy, which prioritizes locations where the GP is most uncertain.

---

# 15. Key Findings

### 1. BLR achieved the best numerical accuracy

BLR achieved the lowest MAE:

$$
2.85^\circ C
$$

compared with:

* GP Spatial Only: $3.16^\circ C$
* GP Full Features: $2.99^\circ C$

---

### 2. GP provides more informative uncertainty

BLR produces nearly uniform uncertainty of approximately:

$$
0.20^\circ C
$$

GP uncertainty varies according to the spatial availability of observations.

This makes GP uncertainty more useful for identifying regions where additional measurements may be required.

---

### 3. A larger GP bandwidth produced better spatial behavior

The $\ell=0.3$ model was highly local and produced fragmented predictions.

The $\ell=1.5$ model produced smoother and more spatially coherent predictions.

For national-scale prediction across Austria, $\ell=1.5$ provided the more useful spatial behavior.

---

### 4. The ice-warning system identified high-risk areas

Using:

$$
P(T\leq0^\circ C)\geq0.8
$$

the $\ell=1.5$ model classified:

**21.6% of the prediction grid as high risk.**

The warning areas were concentrated mainly in colder and mountainous regions.

---

### 5. Active learning concentrates on uncertain areas

The active-learning algorithm selected new station locations in areas where the GP had high predictive uncertainty.

The selected locations were distributed across Austria rather than concentrated around existing stations.

---

### 6. Distance is strongly related to uncertainty

The uncertainty-distance correlation was:

$$
r=0.92
$$

This indicates that spatial observation coverage is a major factor affecting prediction confidence.

---

### 7. Altitude improves physical representation but can increase GP uncertainty

Including altitude allows the model to account for elevation-related temperature differences.

However, in mountainous regions, the addition of altitude increased GP predictive variance under the tested configuration.

---

# 16. Limitations

Several limitations should be considered when interpreting the results.

### Limited temporal modeling

The main spatial prediction task uses a selected timestep rather than explicitly modeling the complete temporal dynamics of temperature.

### Manual GP bandwidth selection

The GP length scales were selected manually:

* $\ell=0.3$
* $\ell=1.5$

Automatic hyperparameter optimization could potentially improve GP performance.

### Limited input features

Only longitude, latitude, and altitude were used.

Additional geographical or meteorological features could potentially improve prediction.

### Computational cost

Standard GP regression scales cubically with the number of training observations, limiting scalability to much larger station networks.

### Limited seasonal coverage

The analyzed period represents a relatively short period in March 2026. A longer dataset covering multiple seasons and years would provide a more comprehensive evaluation.

---

# 17. Potential Improvements

Possible extensions of this project include:

* Automatic optimization of the GP length scale.
* Testing additional kernel functions.
* Incorporating additional meteorological variables.
* Adding terrain characteristics such as slope and land cover.
* Using longer historical datasets.
* Evaluating the models across different seasons.
* Testing sparse or approximate Gaussian Processes for larger station networks.
* Comparing additional active-learning acquisition functions.
* Combining spatial and temporal information in a spatio-temporal GP.
* Evaluating calibration of predictive uncertainty in addition to MAE and MSE.

---

# 18. Project Structure

```text
Weather-Prediction-using-BLR-and-GP/
│
├── weather_assignment.ipynb
├── requirements.txt
├── README.md
└── weather_data.h5
```

> The dataset file may not be included in the GitHub repository because of its size. Place the provided `weather_data.h5` file in the project directory before running the notebook.

---

# 19. Installation

Clone the repository:

```bash
git clone https://github.com/wishalfatima/Weather-Prediction-using-BLR-and-GP.git
```

Navigate to the project:

```bash
cd Weather-Prediction-using-BLR-and-GP
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

# 20. Requirements

The project uses the following Python libraries:

```text
numpy
h5py
matplotlib
scikit-learn
scipy
pandas
geopandas
jupyter
```

---

# 21. Running the Project

Place the dataset:

```text
weather_data.h5
```

in the project directory.

Then start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
weather_assignment.ipynb
```

Execute the notebook cells in order.

The notebook contains the implementation and experiments for:

1. Data loading and exploration
2. Feature engineering
3. Bayesian Linear Regression
4. Gaussian Process Regression
5. GP bandwidth comparison
6. Probabilistic ice-warning detection
7. Active learning
8. Model comparison
9. Uncertainty analysis
10. Visualization

---

# 22. Visualizations

The notebook generates visualizations including:

* Weather station locations across Austria
* Temperature trends
* BLR mean temperature prediction
* BLR predictive uncertainty
* GP mean prediction
* GP predictive uncertainty
* Comparison of GP bandwidths
* Ice-warning probability maps
* Ice-risk classification maps
* Observed station temperatures
* Active-learning station locations
* Prediction versus ground-truth plots
* Uncertainty versus altitude
* Uncertainty versus distance to the nearest weather station

---

# 23. Main Results at a Glance

| Model / Experiment                 | Main Result              |
| ---------------------------------- | ------------------------ |
| BLR                                | MAE = **2.85°C**         |
| GP Spatial Only                    | MAE = **3.16°C**         |
| GP Full Features                   | MAE = **2.99°C**         |
| BLR uncertainty                    | ~**0.20°C**              |
| GP uncertainty                     | ~**0.73–0.77°C average** |
| GP $\ell=0.3$ ice warnings         | **3.6%** of grid         |
| GP $\ell=1.5$ ice warnings         | **21.6%** of grid        |
| Best GP bandwidth                  | **$\ell=1.5$**           |
| Active-learning stations           | **100**                  |
| $\ell=1.5$ average station spacing | **316 km**               |
| Uncertainty-distance correlation   | **0.92**                 |

---

# 24. Conclusion

This project demonstrates the application of probabilistic machine learning to spatial temperature prediction across Austria.

Bayesian Linear Regression achieved the best numerical prediction accuracy for the evaluated test setting, with an MAE of **2.85°C**. Its main advantage is computational efficiency and simplicity.

Gaussian Processes provided a more flexible spatial model and, importantly, produced **location-dependent predictive uncertainty**. This allowed uncertainty to be used for both probabilistic ice-warning detection and active learning for weather-station placement.

Among the tested GP configurations, the larger bandwidth:

$$
\ell=1.5
$$

provided the best balance between smooth spatial predictions and meaningful uncertainty.

The ice-warning experiment demonstrated how probabilistic predictions can be transformed into practical risk information, while the active-learning experiment showed how predictive uncertainty can guide the placement of additional weather stations.

Overall, the project highlights an important distinction between **prediction accuracy** and **uncertainty quality**: although BLR achieved the lowest MAE in this experiment, Gaussian Processes provided substantially richer information about where predictions are reliable and where additional observations are needed.

---

# Author

**Wishal Fatima**

University of Vienna
Probabilistic Artificial Intelligence

---

# License

This project was developed for educational purposes as part of the **Probabilistic Artificial Intelligence course at the University of Vienna**.
