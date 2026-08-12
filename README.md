

**Probabilistic Weather Forecasting with Bayesian Linear Regression (BLR) and Gaussian Processes (GP)**

This project implements probabilistic temperature prediction across Austria using **Bayesian Linear Regression** and **Gaussian Process Regression**, both implemented from scratch using NumPy.

The project was completed as part of the **Probabilistic Artificial Intelligence** course at the **University of Vienna**.

The main goals are to:

* Predict temperature across Austria from weather-station observations.
* Compare Bayesian Linear Regression and Gaussian Processes.
* Quantify predictive uncertainty.
* Develop a probabilistic ice-warning system.
* Use active learning to identify informative locations for additional weather stations.
* Investigate the relationship between spatial coverage, altitude, and predictive uncertainty.
* Compare the computational cost of BLR and GP.

---

## Overview

The project uses temperature observations from **199 weather stations across Austria**.

The models use three geographic features:

* Longitude
* Latitude
* Altitude

A spatial prediction grid containing **10,000 locations** is then used to generate temperature predictions across Austria.

Unlike conventional deterministic regression, both models provide probabilistic predictions, allowing the project to estimate not only the expected temperature but also the uncertainty associated with each prediction.

---

# Dataset

The dataset contains temperature measurements from **199 weather stations across Austria**.

## Training Data

* **2,017 time steps**
* **199 weather stations**
* 10-minute temporal resolution
* Training period: **9 March – 23 March 2026**

## Test Data

* **1,439 time steps**
* **199 weather stations**
* Test period: **23 March – 1 April 2026**

## Spatial Prediction Grid

Predictions are generated over a regular grid covering Austria.

| Property          |             Value |
| ----------------- | ----------------: |
| Grid size         |         100 × 100 |
| Total grid points |            10,000 |
| Longitude range   |  9.000° – 17.500° |
| Latitude range    | 46.000° – 49.500° |

The resulting grid contains **10,000 prediction locations**.

For each location, the following features are available:

| Feature   | Description                     |
| --------- | ------------------------------- |
| Longitude | East–west geographic position   |
| Latitude  | North–south geographic position |
| Altitude  | Elevation above sea level       |

The station feature matrix has shape:

```text
199 × 3
```

and the prediction-grid feature matrix has shape:

```text
10,000 × 3
```

---

# Project Structure

```text
Weather-Prediction-using-BLR-and-GP/
│
├── weather_assignment.ipynb
├── requirements.txt
├── weather_data.h5
└── README.md
```

> **Note:** The dataset may not be included in the GitHub repository because of its size or distribution restrictions. If `weather_data.h5` is not included, it must be obtained separately before running the notebook.

---

# Methodology

The project consists of four main components:

1. Data exploration and feature engineering
2. Bayesian Linear Regression
3. Gaussian Process Regression
4. Ice-warning and active-learning applications

---

# 1. Data Exploration and Feature Engineering

## Weather Station Distribution

The 199 weather stations are distributed across Austria with varying spatial density.

The station locations are visualized using their longitude and latitude coordinates to understand the geographic coverage of the available observations.

## Temperature Analysis

A two-day temperature period at **station 5 (Freistadt)** was examined.

The observed temperature statistics were:

| Statistic                 |     Value |
| ------------------------- | --------: |
| Minimum temperature       |   -0.60°C |
| Maximum temperature       |   16.40°C |
| Average temperature       |    6.37°C |
| Temperature range         |   17.00°C |
| Time below freezing       | 3.2 hours |
| Percentage below freezing |      6.6% |

The time series shows clear **diurnal temperature cycles**, with temperatures generally increasing during the day and decreasing at night.

Periods below 0°C are particularly relevant for the ice-warning application.

---

# 2. Feature Engineering

Three geographic features were extracted from the weather stations:

* Longitude
* Latitude
* Altitude

The observed ranges were:

| Feature   |             Range |
| --------- | ----------------: |
| Longitude |  9.610° – 16.845° |
| Latitude  | 46.444° – 48.955° |
| Altitude  |   116 m – 3,437 m |

The features were standardized using `StandardScaler`.

The scaler was fitted using the weather-station features and then applied to the prediction grid.

### Feature Summary

| Property               | Value                         |
| ---------------------- | ----------------------------- |
| Input features         | Longitude, Latitude, Altitude |
| Station feature matrix | 199 × 3                       |
| Grid feature matrix    | 10,000 × 3                    |
| Target                 | Temperature (°C)              |
| Normalization          | StandardScaler                |

Although the dataset contains temporal temperature measurements, the regression models in this experiment use the geographic features to learn the spatial relationship between location and temperature at a selected time step.

---

# 3. Bayesian Linear Regression

## Model

Bayesian Linear Regression (BLR) was implemented **from scratch using NumPy**.

A zero-mean isotropic Gaussian prior was placed over the model weights:

$$
w \sim \mathcal{N}(0,\alpha^{-1}I)
$$

where $\alpha$ controls the strength of the prior.

The likelihood is modeled as:

$$
y \sim \mathcal{N}(Xw,\beta^{-1}I)
$$

where $\beta$ is the noise precision.

The posterior covariance is:

$$
S_N =
(\alpha I + \beta X^T X)^{-1}
$$

and the posterior mean is:

$$
m_N =
\beta S_N X^T y
$$

## Training Setup

The model was trained using:

* Longitude
* Latitude
* Altitude
* 199 weather stations
* A selected temperature timestep as the target
* Standardized input features

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

# 4. Gaussian Process Regression

## Model

Gaussian Process Regression was also implemented **from scratch using NumPy**.

An RBF (Radial Basis Function) kernel was used:

$$
k(x,x') =
\sigma_f^2
\exp
\left(
-\frac{|x-x'|^2}{2\ell^2}
\right)
$$

where:

* $\sigma_f^2$ is the signal variance.
* $\ell$ is the length scale or bandwidth.
* $x$ and $x'$ are two input locations.

The GP predictive mean is:

$$
\mu(x_*) =
K(x_*,X)
\left[
K(X,X)+\sigma_n^2I
\right]^{-1}
y
$$

The predictive variance is:

$$
\sigma^2(x_*) =
k(x_*,x_*)
----------

K(x_*,X)
\left[
K(X,X)+\sigma_n^2I
\right]^{-1}
K(X,x_*)
$$

where $\sigma_n^2$ represents the observation-noise variance.

---

# 5. GP Bandwidth Experiments

Two RBF bandwidths were investigated.

| Bandwidth    | Behaviour                  |
| ------------ | -------------------------- |
| $\ell = 0.3$ | Local and highly sensitive |
| $\ell = 1.5$ | Smooth and more global     |

## GP with $\ell = 0.3$

The small bandwidth produced highly localized temperature predictions.

The resulting temperature map contained many small warm and cold regions, indicating strong local variation.

This allows the GP to capture fine-grained spatial differences but also makes it more sensitive to noise and sparse observations.

The uncertainty increases substantially in areas far from weather stations.

## GP with $\ell = 1.5$

The larger bandwidth produced smoother and more continuous temperature predictions.

The spatial temperature pattern was more structured, with:

* Colder regions in mountainous areas.
* Warmer temperatures in lowland areas.
* Smoother transitions between neighbouring locations.

The uncertainty was also more spatially coherent.

For this project, $\ell = 1.5$ was selected for the final ice-warning and active-learning analyses because it provided a better balance between smoothness and spatial flexibility.

---

# 6. BLR vs GP

## Predictive Accuracy

The models were evaluated using Mean Squared Error (MSE) and Mean Absolute Error (MAE).

| Model              | MSE (°C²) | MAE (°C) |
| ------------------ | --------: | -------: |
| BLR                | **12.54** | **2.85** |
| GP — Spatial Only  |     16.35 |     3.16 |
| GP — Full Features |     13.00 |     2.99 |

### Interpretation

BLR achieved the lowest MAE in this experiment.

This suggests that the temperature relationship represented by the available geographic features was sufficiently close to a linear relationship for BLR to perform very well.

The GP models provide greater flexibility but require appropriate kernel hyperparameters.

The GP using only spatial coordinates performed worst because it excluded altitude, which is an important physical factor for temperature variation in Austria.

---

# 7. Predictive Uncertainty

One of the most important differences between BLR and GP is their treatment of uncertainty.

| Model              | Average Uncertainty |
| ------------------ | ------------------: |
| BLR                |             ~0.20°C |
| GP — Spatial Only  |             ~0.73°C |
| GP — Full Features |             ~0.77°C |

BLR produced nearly uniform uncertainty across the prediction grid.

This occurs because BLR assumes a global linear relationship and learns one posterior distribution over the model weights.

Gaussian Processes produce **location-dependent uncertainty**.

Regions close to observations tend to have lower uncertainty, while regions far from weather stations have higher uncertainty.

The GP uncertainty analysis produced values ranging approximately from:

**0.72°C to 4.76°C**

The correlation between distance to the nearest weather station and predictive uncertainty was approximately:

**0.92**

The correlation between altitude and predictive uncertainty was approximately:

**-0.11**

This indicates that, in this experiment, **distance to available observations was a much stronger driver of uncertainty than altitude**.

---

# 8. Effect of Altitude

Altitude was specifically investigated because Austria contains large differences in elevation.

For mountainous regions above 1,500 m, predictive uncertainty was measured as:

| Model            | Uncertainty |
| ---------------- | ----------: |
| GP Spatial Only  |    1.0550°C |
| GP Full Features |    1.3485°C |

Including altitude improved the mean prediction, but it increased predictive variance in these mountainous regions by approximately **27.8%**.

This can occur because altitude changes the geometry of the input space used by the RBF kernel. Large differences in altitude can increase the effective distance between observations and prediction locations, reducing their similarity under the kernel.

Therefore, in this experiment:

* Altitude improved the spatial temperature representation.
* Altitude did not reduce GP uncertainty in mountainous regions.
* The effect of altitude on uncertainty was more complex than its effect on the predictive mean.

---

# 9. Probabilistic Ice-Warning System

A probabilistic ice-warning system was implemented using the Gaussian Process model.

For every prediction location, the GP produces a Gaussian temperature distribution:

$$
T \sim \mathcal{N}(\mu,\sigma^2)
$$

The probability that the temperature is at or below freezing is:

$$
P(T \leq 0^\circ C)
$$

A warning is issued when:

$$
P(T \leq 0^\circ C) \geq 0.8
$$

This means a location is classified as high risk when the model estimates at least an **80% probability of freezing temperatures**.

---

# 10. Ice-Warning Results

Both GP bandwidths were tested.

| GP Bandwidth | Warning Points | Percentage |
| ------------ | -------------: | ---------: |
| $\ell = 0.3$ |            361 |       3.6% |
| $\ell = 1.5$ |          2,156 |      21.6% |

The $\ell = 1.5$ model was selected for the final warning map because it produced more spatially coherent warning zones.

## Final Risk Classification

| Risk Level | Probability | Grid Points | Proportion |
| ---------- | ----------- | ----------: | ---------: |
| High       | > 80%       |       2,156 |      21.6% |
| Medium     | 20% – 80%   |         792 |       7.9% |
| Low        | ≤ 20%       |       7,052 |      70.5% |

The high-risk areas were concentrated mainly in colder and higher-altitude regions, particularly mountainous areas.

The resulting spatial pattern is physically reasonable because temperature generally decreases with increasing altitude.

---

# 11. Active Learning for Weather Station Placement

An active-learning strategy was developed to identify **100 informative candidate locations** for potential additional weather stations.

The initial candidate set consisted of the 10,000 grid locations.

Only candidate points within Austria were retained for realistic station placement.

## Selection Strategy

The algorithm starts with an empty set of selected locations.

At each iteration:

1. Compute the GP predictive variance at all remaining candidate locations.
2. Select the location with the highest predictive variance.
3. Add that location to the selected set.
4. Update the GP posterior.
5. Repeat until 100 locations have been selected.

The next location is selected according to:

$$
x_{\text{next}}
===============

\arg\max_{x \in \mathcal{X}_{\text{candidate}}}
\sigma^2(x)
$$

This strategy prioritizes areas where the model is most uncertain.

---

# 12. Active Learning Results

Two GP bandwidths were compared.

| Metric                     | $\ell = 0.3$ | $\ell = 1.5$ |
| -------------------------- | -----------: | -----------: |
| Average station distance   |       276 km |   **316 km** |
| Average altitude           |      1,271 m |      1,239 m |
| Spatial coverage           |     Moderate |     **High** |
| Sensitivity to local noise |       Higher |    **Lower** |
| National-scale suitability |     Moderate |   **Better** |

The $\ell = 1.5$ model was selected for the final station-placement analysis.

The selected locations were distributed across:

* Western mountainous regions
* Central Austria
* Eastern lowlands

The selected stations were generally spread out rather than concentrated in a single region.

This occurs because once a location is selected, the predictive variance around that location decreases, making nearby locations less informative.

---

# 13. Computational Trade-Off

BLR and GP have substantially different computational requirements.

## Bayesian Linear Regression

BLR scales approximately with:

$$
O(ND^2)
$$

where:

* $N$ = number of observations
* $D$ = number of features

Because the number of features is small, BLR is computationally efficient.

## Gaussian Process

GP requires operations involving the $N \times N$ kernel matrix and therefore has approximately cubic computational complexity:

$$
O(N^3)
$$

In the timing experiment, GP was approximately **135× slower than BLR** for the 199 weather stations.

| Model | Computational Cost | Main Advantage                               |
| ----- | ------------------ | -------------------------------------------- |
| BLR   | Low                | Fast and accurate                            |
| GP    | High               | Flexible predictions and spatial uncertainty |

---

# 14. Key Findings

## Finding 1 — BLR achieved the best accuracy

BLR achieved an MAE of **2.85°C**, which was lower than both GP configurations tested.

## Finding 2 — GP provides richer uncertainty information

Although GP produced higher average uncertainty than BLR, its uncertainty varied according to the spatial distribution of observations.

This makes GP uncertainty more informative for identifying poorly observed regions.

## Finding 3 — The larger GP bandwidth performed better spatially

The $\ell = 1.5$ model generated smoother and more spatially coherent temperature predictions than $\ell = 0.3$.

## Finding 4 — Distance to observations strongly affects uncertainty

The correlation between nearest-station distance and predictive uncertainty was approximately **0.92**.

This indicates that data availability is a major factor determining prediction confidence.

## Finding 5 — Altitude affects the GP differently from spatial distance

Adding altitude improved the representation of temperature variation but increased predictive uncertainty in the high-altitude region investigated.

## Finding 6 — Active learning identifies informative locations

The active-learning approach selected locations where the GP was most uncertain, producing broader spatial coverage and reducing redundant station placement.

---

# 15. Overall Model Comparison

| Aspect                           | BLR            | GP — Spatial Only | GP — Full Features |
| -------------------------------- | -------------- | ----------------- | ------------------ |
| Accuracy                         | **Best**       | Lowest            | Medium             |
| Flexibility                      | Low            | High              | High               |
| Uncertainty                      | Nearly uniform | Spatially varying | Spatially varying  |
| Altitude                         | Yes            | No                | Yes                |
| Computational cost               | **Low**        | High              | High               |
| Suitable for uncertainty mapping | Limited        | Good              | **Good**           |

### Overall Interpretation

BLR performed best in terms of predictive accuracy for this particular dataset and experiment.

However, Gaussian Processes provide a major advantage in **spatially varying uncertainty estimation**.

Therefore, the choice between the models depends on the application:

* **BLR** is attractive when computational efficiency and predictive accuracy are the main priorities.
* **GP** is more useful when understanding where predictions are uncertain is important.

For applications such as ice-warning systems and weather-station placement, the spatial uncertainty information provided by GP is particularly valuable.

---

# 16. Visualizations

The notebook generates visualizations covering:

* Weather station locations across Austria
* Temperature trends at individual stations
* BLR mean temperature predictions
* BLR predictive uncertainty
* GP mean predictions
* GP uncertainty maps
* Comparison of GP bandwidths
* Ice-warning probability maps
* Ice-risk classification
* Observed station temperatures
* Active-learning station locations
* Predictions versus ground truth
* Uncertainty versus altitude
* Uncertainty versus distance to the nearest station

---

# 17. How to Run

## 1. Clone the repository

```bash
git clone https://github.com/wishalfatima/Weather-Prediction-using-BLR-and-GP.git
```

## 2. Enter the project directory

```bash
cd Weather-Prediction-using-BLR-and-GP
```

## 3. Install the dependencies

```bash
pip install -r requirements.txt
```

## 4. Add the dataset

Place the required HDF5 dataset in the project directory:

```text
weather_data.h5
```

If the dataset is not included in the repository, obtain it separately and place it in the expected location.

## 5. Start Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
weather_assignment.ipynb
```

Execute the notebook cells in order.

---

# 18. Technologies

The project uses:

* **Python**
* **NumPy**
* **SciPy**
* **Pandas**
* **Scikit-learn**
* **Matplotlib**
* **GeoPandas**
* **h5py**
* **Jupyter Notebook**

The BLR and GP regression models were implemented from scratch using NumPy.

---

# 19. Implementation Highlights

### Bayesian Linear Regression

Implemented manually using:

* Gaussian prior
* Posterior mean
* Posterior covariance
* Predictive mean
* Predictive variance

### Gaussian Process

Implemented manually using:

* RBF kernel
* Kernel matrix construction
* GP posterior mean
* GP posterior variance
* Multiple bandwidth experiments

### Probabilistic Ice Warning

Implemented using:

* GP predictive mean
* GP predictive variance
* Gaussian freezing probability
* 80% warning threshold

### Active Learning

Implemented using:

* Predictive variance
* Greedy maximum-uncertainty selection
* Iterative GP posterior updates
* Selection of 100 candidate station locations

---

# 20. Repository

**GitHub Repository:**

[https://github.com/wishalfatima/Weather-Prediction-using-BLR-and-GP](https://github.com/wishalfatima/Weather-Prediction-using-BLR-and-GP)

---

# 21. Author

**Wishal Fatima**

University of Vienna
Master's Programme in Computer Science

This project was completed as part of the **Probabilistic Artificial Intelligence** course at the University of Vienna.

---

# License

This project is intended for **educational and academic purposes**.
:::
