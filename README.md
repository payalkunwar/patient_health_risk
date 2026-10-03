# 🩺 Patient Health Risk Segmentation Using Clustering

> **Academic B.Tech Machine Learning Project**  
> Unsupervised Learning · K-Means · Hierarchical Clustering · PCA · Streamlit  

---

## ⚠️ Important Disclaimer

This project uses **entirely synthetic, artificially generated data**.  
All results are for **academic and educational purposes only**.  
Nothing in this project constitutes medical advice, clinical risk prediction, or insurance decisions.

---

## 📋 Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Objectives](#2-objectives)
3. [Dataset Generation](#3-dataset-generation)
4. [Features](#4-features)
5. [Synthetic Data Methodology](#5-synthetic-data-methodology)
6. [Missing Value Handling](#6-missing-value-handling)
7. [Feature Scaling](#7-feature-scaling)
8. [K-Means Clustering](#8-k-means-clustering)
9. [Hierarchical Clustering](#9-hierarchical-clustering)
10. [PCA](#10-pca---dimensionality-reduction)
11. [Elbow Method & WCSS](#11-elbow-method--wcss)
12. [Silhouette Score](#12-silhouette-score)
13. [Dendrogram](#13-dendrogram)
14. [Cluster Profiling](#14-cluster-profiling)
15. [Model Comparison](#15-model-comparison)
16. [Deployment](#16-deployment)
17. [Ethical Considerations](#17-ethical-considerations)
18. [Limitations](#18-limitations)
19. [Installation](#19-installation)
20. [How to Run](#20-how-to-run)
21. [Project Structure](#21-project-structure)

---

## 1. Problem Statement

A health insurance provider wants to group policyholders according to health-related indicators —
such as age, BMI, blood pressure, cholesterol, physical activity level, and number of previous claims —
to support the design of **preventive health programmes**.

Since there is no pre-labelled dataset, this is framed as an **unsupervised learning** problem.
The model must discover natural groupings in the data without any prior labels.

---

## 2. Objectives

- Generate a realistic synthetic dataset of 1 200 policyholders.
- Introduce and handle realistic missing clinical values.
- Discover natural health-indicator groups using K-Means and Hierarchical Clustering.
- Select the optimal number of clusters using quantitative metrics.
- Visualise cluster structure in 2-D using PCA.
- Profile each cluster and assign descriptive labels.
- Deploy an interactive dashboard for real-time cluster assignment.

---

## 3. Dataset Generation

**File**: `data/health_policyholders.csv`  
**Records**: 1 200  
**Random seed**: 42 (for reproducibility)  

There is no standard public dataset containing all required attributes.  
The data was generated using NumPy with realistic clinical distributions embedded around three latent groups.

---

## 4. Features

| Feature | Description | Range |
|---|---|---|
| `age` | Policyholder age in years | 18–85 |
| `bmi` | Body Mass Index (kg/m²) | 15–45 |
| `systolic_bp` | Systolic blood pressure (mmHg) | 90–190 |
| `diastolic_bp` | Diastolic blood pressure (mmHg) | 55–120 |
| `cholesterol` | Total cholesterol (mg/dL) | 110–330 |
| `physical_activity` | Exercise hours per week | 0–12 |
| `previous_claims` | Number of prior insurance claims | 0–12 |

---

## 5. Synthetic Data Methodology

Three latent health-indicator profiles were embedded:

**Group A — Lower-indicator profile**
- Younger policyholders (18–50), lower BMI (~15–29), normal BP and cholesterol, high physical activity.

**Group B — Moderate-indicator profile**
- Middle-aged (35–67), moderate BMI (~23–36), moderate BP and cholesterol, moderate activity.

**Group C — Higher-indicator profile**
- Older policyholders (52–85), higher BMI (~27–45), elevated BP and cholesterol, low activity.

Each group was sampled from normal distributions. All records were shuffled together.  
**The group labels are NOT stored in the CSV** — the ML algorithm must discover them.

```python
SEED = 42
np.random.seed(SEED)
```

---

## 6. Missing Value Handling

Approximately **3% of values** in selected clinical columns (`bmi`, `systolic_bp`, `diastolic_bp`, `cholesterol`) were randomly set to `NaN` to simulate incomplete real-world records.

### Why not delete missing rows?

Deleting rows with missing values reduces the dataset and can introduce bias if missingness is not completely random. In clinical settings, it is preferable to **impute** plausible values.

### Imputation

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X_raw)
```

- **Median imputation** is used because health measurements are often skewed and the median is robust to outliers.
- The imputer is fitted once on the training data and the **same fitted imputer** is used in the deployment app. This prevents data leakage.

---

## 7. Feature Scaling

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
```

### Why scaling is essential

Both K-Means and Hierarchical Clustering are **distance-based algorithms**.  
Without scaling, features with large numeric ranges (e.g., `cholesterol`: 110–330) dominate features with smaller ranges (e.g., `previous_claims`: 0–12), causing biased cluster assignments.

StandardScaler transforms each feature to **zero mean and unit variance**, giving all features equal influence on distance calculations.

---

## 8. K-Means Clustering

```python
from sklearn.cluster import KMeans
KMeans(n_clusters=selected_k, random_state=42, n_init=20)
```

K-Means is a partition-based algorithm that:
1. Randomly initialises K cluster centroids.
2. Assigns each point to the nearest centroid.
3. Recalculates centroids as the mean of assigned points.
4. Repeats steps 2–3 until centroids stabilise (convergence).

`n_init=20` runs the algorithm 20 times with different random initialisations, keeping the best result to avoid local minima.

---

## 9. Hierarchical Clustering

```python
from sklearn.cluster import AgglomerativeClustering
AgglomerativeClustering(n_clusters=selected_k, linkage="ward")
```

Agglomerative hierarchical clustering is a **bottom-up** approach:
- Starts with every point in its own cluster.
- Iteratively merges the two closest clusters.
- **Ward linkage** merges clusters that minimise the total within-cluster variance.

Used as a **comparison model** only. Not deployed for prediction because it does not expose a `.predict()` method for new records.

---

## 10. PCA – Dimensionality Reduction

```python
from sklearn.decomposition import PCA
PCA(n_components=2, random_state=42)
```

PCA (Principal Component Analysis) reduces the 7-dimensional scaled feature space to 2 dimensions, preserving the maximum variance. The resulting two principal components are used **only for visualisation** — not as clustering features.

The PCA scatter plot colours points by their K-Means cluster label to visually verify cluster separation.

---

## 11. Elbow Method & WCSS

**WCSS** (Within-Cluster Sum of Squares), also called **inertia**, measures the total squared distance between each point and its cluster centroid.

As K increases:
- WCSS always decreases.
- The rate of decrease slows after the optimal K.
- The "elbow" — where the curve bends — indicates the optimal K.

Saved at: `reports/plots/elbow_curve.png`

---

## 12. Silhouette Score

The silhouette score for a point measures:

```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

Where:
- `a(i)` = mean distance to other points in the same cluster (cohesion)
- `b(i)` = mean distance to points in the nearest other cluster (separation)

Range: **−1 to +1**. Higher is better. A score near 1 means the point is well-matched to its cluster.

Saved at: `reports/plots/silhouette_curve.png`

---

## 13. Dendrogram

A dendrogram visualises the hierarchical merging process:
- The **height** of each horizontal bar represents the linkage distance at which two clusters merged.
- A **horizontal cut** at any height yields a specific number of clusters.

The dendrogram is truncated to the last 30 merges for readability.  
A red dashed line indicates the cut threshold for the selected K.

Saved at: `reports/plots/dendrogram.png`

---

## 14. Cluster Profiling

After K-Means clustering, the mean of each feature is computed **on original (unscaled) values** for each cluster.

A composite indicator score `(age + bmi + systolic_bp + cholesterol)` is used to rank clusters from lowest to highest indicator level. Descriptive names are then assigned:

| Rank | Name |
|---|---|
| 1 (lowest) | Lower-indicator profile |
| 2 | Moderate-indicator profile |
| 3 (highest) | Higher-indicator profile |

**Note**: Cluster numbers (0, 1, 2 …) are assigned by K-Means arbitrarily. The profiling step determines which cluster corresponds to which descriptive name.

---

## 15. Model Comparison

| Criterion | K-Means (Selected Model) | Hierarchical (Ward Comparison) |
|---|---|---|
| Algorithm type | Centroid-based Partitioning | Agglomerative Bottom-Up Hierarchical |
| Requires K upfront | ✅ Yes (tested K=2..8) | ✅ Yes (cut at selected K) |
| Selected Clusters (K) | **3** | **3** |
| Silhouette Score | **0.4987** | **0.4985** |
| WCSS / Inertia | **1707.00** (sharp elbow) | N/A |
| Assigns new records | ✅ Direct `.predict()` via centroids | ❌ Requires tree recomputation |
| Deployed in App | ✅ **Yes** (Production deployment) | ❌ Comparison baseline only |
| Visualisation | Elbow Curve + Silhouette Curve | Truncated Dendrogram (Ward Linkage) |

**Why K-Means is selected for deployment:**  
K-Means exposes a `.predict()` method that assigns any incoming policyholder record to the nearest learned cluster centroid in $O(K \cdot d)$ time without retraining. Scikit-learn's `AgglomerativeClustering` requires refitting the entire dendrogram matrix, making it computationally impractical for real-time web deployment. Furthermore, K-Means achieved an outstanding Silhouette Score of **0.4987** with a distinct elbow at $K=3$.

---

## 16. Deployment

The Streamlit application (`app/app.py`) loads `models/model_bundle.joblib` which contains:

- Fitted `KMeans` model
- Fitted `SimpleImputer`
- Fitted `StandardScaler`
- Fitted `PCA` transformer
- Feature list
- Selected K
- Cluster profiles DataFrame
- Cluster profile name map
- Random seed

**The model is never retrained during the app session.** It loads once and serves all predictions.

---

## 17. Ethical Considerations

### Synthetic Data
The dataset is entirely artificial. No real patient data was used.

### Not Medical Advice
Cluster assignments are statistical groupings only. They are **not** diagnoses, clinical risk scores, or insurance-related decisions.

### Potential Bias
Real-world health datasets may contain demographic, socioeconomic, measurement, or reporting biases that could skew clustering results.

### Privacy
In a real-world application, patient data would require strict privacy controls (HIPAA, GDPR, etc.).

### Insurance Fairness
Health-indicator clusters must **not** be used to increase premiums, deny coverage, or restrict access to healthcare services.

### Stigmatisation
Labels such as "higher-indicator profile" must be communicated with care to avoid stigmatising individuals.

### Self-Reported Data
Many clinical measurements (especially BMI and activity levels) may be self-reported and therefore incomplete or inaccurate.

### Causality
Clustering identifies statistical associations but does **not** establish causality between features and health outcomes.

---

## 18. Limitations

- Synthetic data may not capture the full complexity of real-world health populations.
- K-Means assumes roughly spherical clusters of similar size, which may not hold for all health datasets.
- Results depend on the choice of K and random seed.
- Median imputation may introduce small biases when missing data is not completely random.
- PCA visualisation captures only a fraction of total variance; full cluster structure exists in 7 dimensions.

---

## 19. Installation

```bash
# Clone or download the project
cd Patient_Health_Risk_Segmentation

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 20. How to Run

### Step 1 – Train the model

```bash
python train_model.py
```

This will:
- Generate `data/health_policyholders.csv`
- Run EDA and save plots to `reports/plots/`
- Fit K-Means (K=2..8) and Hierarchical Clustering
- Save `models/model_bundle.joblib`
- Save `reports/cluster_profiles.csv`, `cluster_profiles_named.csv`, `cluster_sizes.csv`, `model_metrics.csv`

### Step 2 – Launch the dashboard

```bash
streamlit run app/app.py
```

Open your browser at `http://localhost:8501`.

### Step 2 – Run the Jupyter Notebook (Main ML Deliverable)

Open and run the primary notebook from top to bottom (`Run All`):

```bash
jupyter notebook notebooks/Patient_Health_Risk_Segmentation.ipynb
# or open directly in VS Code / IDE
```

This comprehensive notebook walks through all 35 steps of the Machine Learning lifecycle:
- Synthetic data generation (1,200 records, SEED=42)
- Missing value injection and median imputation
- Exploratory data analysis, distributions, and correlation heatmap
- Standardization with `StandardScaler`
- K-Means evaluation across $K=2..8$ (WCSS, Elbow method, Silhouette analysis)
- Agglomerative Hierarchical Clustering & Ward dendrogram
- PCA dimensionality reduction and 2D visualization
- Cluster profiling and dynamic descriptive naming
- Serialization to `models/model_bundle.joblib`
- Live inference demonstration on a test policyholder record

### Step 3 – Launch the Streamlit Interactive Dashboard

```bash
streamlit run app/app.py
```

Open your browser at `http://localhost:8501` to use the interactive risk assignment tool.

---

## 21. Project Structure

```text
Patient_Health_Risk_Segmentation/
│
├── notebooks/
│   ├── Patient_Health_Risk_Segmentation.ipynb   ← MAIN ML WORK (Executable 35-step analysis)
│   └── analysis.py                             ← EDA utility script
│
├── app/
│   └── app.py                                  ← Streamlit deployment dashboard
│
├── data/
│   └── health_policyholders.csv                ← Synthetic dataset (1 200 records)
│
├── models/
│   └── model_bundle.joblib                     ← Saved model bundle for deployment
│
├── reports/
│   ├── methodology.md                          ← Comprehensive methodology report
│   ├── cluster_profiles.csv                    ← Cluster centroids in original physical units
│   ├── cluster_profiles_named.csv              ← Centroids with dynamic descriptive profile names
│   ├── cluster_sizes.csv                       ← Policyholder distribution per cluster
│   ├── model_metrics.csv                       ← WCSS & silhouette scores across K=2..8
│   └── plots/
│       ├── elbow_curve.png
│       ├── silhouette_curve.png
│       ├── dendrogram.png
│       ├── pca_clusters.png
│       ├── distributions.png
│       ├── boxplots.png
│       └── correlation_heatmap.png
│
├── train_model.py                              ← Pipeline script
├── requirements.txt                            ← Complete project dependencies
├── README.md                                   ← Full project documentation & viva guide
└── .gitignore
```

---

## 🛠️ Technology Stack

| Library | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core language |
| pandas | 2.2.2 | Data manipulation |
| numpy | 1.26.4 | Numerical computation & data generation |
| scikit-learn | 1.5.1 | KMeans, AgglomerativeClustering, SimpleImputer, StandardScaler, PCA, silhouette_score |
| scipy | 1.13.1 | Linkage matrix & dendrogram |
| matplotlib | 3.9.1 | Plotting |
| seaborn | 0.13.2 | Statistical visualisation |
| joblib | 1.4.2 | Model serialisation |
| streamlit | 1.37.1 | Interactive dashboard |

---

*B.Tech Machine Learning Project · Academic use only · Synthetic data · Not for clinical or commercial use*
