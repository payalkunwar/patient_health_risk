# Methodology – Patient Health Risk Segmentation

## 1. Overview

This document describes the full methodology used in the **Patient Health Risk Segmentation** project.  
The project uses **unsupervised machine learning** (K-Means and Agglomerative Hierarchical Clustering) on a **synthetic dataset** of 1 200 health insurance policyholders.

> **Important**: This is an academic/educational project. The dataset is entirely artificial.  
> Results must not be interpreted as medical diagnoses, clinical risk predictions, or insurance decisions.

---

## 2. Synthetic Data Generation

Because no standard public dataset contains all the required attributes in a single source, the dataset was generated programmatically using **NumPy** with a fixed random seed (`SEED = 42`) for reproducibility.

### Latent Structure

Three broad latent groups were embedded in the data to make clustering results interpretable:

| Latent Group | Age Range | BMI | Blood Pressure | Cholesterol | Physical Activity | Previous Claims |
|---|---|---|---|---|---|---|
| **A – Lower-indicator** | ~18–50 | ~15–29 | Low-normal | Low-normal | High (≥5 hrs/wk) | Low (0–4) |
| **B – Moderate-indicator** | ~35–67 | ~23–36 | Moderate | Moderate | Moderate (2–8 hrs/wk) | Moderate (0–8) |
| **C – Higher-indicator** | ~52–85 | ~27–45 | Elevated-high | Elevated-high | Low (0–4 hrs/wk) | High (2–12) |

Each group was sampled from multivariate normal distributions with realistic clinical ranges. All groups were shuffled together. **No group labels were stored in the CSV**; the algorithm discovers them.

### Dataset Size

- **Total records**: 1 200 policyholders
- **Split**: ~35% Group A · 35% Group B · 30% Group C

---

## 3. Missing Value Handling

Approximately **3% missing values** were randomly introduced into four clinical measurement columns:

- `bmi`
- `systolic_bp`
- `diastolic_bp`
- `cholesterol`

### Imputation Strategy

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
```

- **Why median?** Medical measurements are often skewed. The median is more robust to extreme values than the mean.
- The imputer was **fitted on the training dataset** and the **same fitted imputer** is applied in deployment (Streamlit app).
- Rows with missing values were **not deleted** — preserving data is especially important in healthcare contexts.

---

## 4. Feature Scaling

### Why scaling is required

K-Means and Agglomerative Hierarchical Clustering are both **distance-based algorithms**. Without scaling, features with naturally larger numeric ranges (e.g., `cholesterol`: 110–330 mg/dL) would dominate features with smaller ranges (e.g., `previous_claims`: 0–12), distorting cluster boundaries.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
```

StandardScaler transforms each feature to have **zero mean and unit variance**.

---

## 5. K-Means Clustering

### Configuration

```python
from sklearn.cluster import KMeans
KMeans(n_clusters=selected_k, random_state=42, n_init=20)
```

- `n_init=20` runs the algorithm 20 times with different initialisations and selects the best result, reducing the chance of converging to a local minimum.

### Cluster Selection

K was tested from **2 to 8**. For each K:

- **WCSS (inertia)** — sum of squared distances from each point to its cluster centroid
- **Silhouette score** — measures how similar a point is to its own cluster vs neighbouring clusters (range: −1 to 1; higher is better)

The optimal K was selected using:
1. Elbow curve (WCSS) to identify the point of diminishing returns
2. Silhouette score (primary criterion; highest score for K ≤ 5)
3. Interpretability in a healthcare context

---

## 6. Agglomerative Hierarchical Clustering

```python
from sklearn.cluster import AgglomerativeClustering
AgglomerativeClustering(n_clusters=selected_k, linkage="ward")
```

- **Ward linkage** minimises the total within-cluster variance at each merge step.
- Used the same number of clusters as selected for K-Means to enable a fair comparison.
- A **dendrogram** was plotted using `scipy.cluster.hierarchy` to visually determine the natural number of clusters.

### Deployment Limitation

Hierarchical clustering (as implemented in scikit-learn) does **not** provide a `.predict()` method for assigning new records to existing clusters. Therefore:

> **K-Means is used for deployment. Hierarchical clustering is used for comparison only.**

---

## 7. PCA – Dimensionality Reduction

```python
from sklearn.decomposition import PCA
PCA(n_components=2, random_state=42)
```

- PCA was applied **after** imputation and scaling.
- The two principal components are used **only for 2-D visualisation** of cluster structure.
- PCA is **not** used as the clustering algorithm.

The proportion of variance explained by each component is reported in the training output.

---

## 8. Cluster Profiling

After fitting K-Means, each cluster's mean values are computed on the **original (unscaled) feature values** to produce human-interpretable profiles.

A composite score `(age + bmi + systolic_bp + cholesterol)` is used to rank clusters by overall health indicator level, and descriptive names are assigned:

| Rank (lowest → highest composite) | Descriptive Name |
|---|---|
| 1 | Lower-indicator profile |
| 2 | Moderate-indicator profile |
| 3 | Higher-indicator profile |

---

## 9. Model Comparison & Empirical Results

| Metric / Feature | K-Means (Selected Model) | Hierarchical Clustering (Ward) |
|---|---|---|
| **Algorithm Type** | Centroid-based partitioning | Agglomerative hierarchical bottom-up |
| **Number of Clusters (K)** | **3** | **3** |
| **Silhouette Score** | **0.4987** | **0.4985** |
| **WCSS / Inertia** | **1707.00** (sharp elbow point) | N/A |
| **New Sample Inference** | ✅ Direct `.predict()` via nearest centroid | ❌ Requires refitting entire tree |
| **Deployed for Inference** | ✅ **Yes** (Production deployment) | ❌ Comparison baseline only |
| **Visualisation Tool** | Elbow Curve + Silhouette Curve | Truncated Dendrogram |

### Dimensionality Reduction (PCA)
- **Component 1 (PC1)**: 81.95% explained variance
- **Component 2 (PC2)**: 5.55% explained variance
- **Total Explained Variance**: **87.49%** in 2 dimensions

### Final Cluster Distribution
- **Cluster 0 — Lower-indicator profile**: 420 policyholders (35.0%)
- **Cluster 2 — Moderate-indicator profile**: 421 policyholders (35.1%)
- **Cluster 1 — Higher-indicator profile**: 359 policyholders (29.9%)

---

## 10. Deployment Artefacts

The following are saved by `train_model.py` and loaded by `app/app.py`:

```
models/model_bundle.joblib
  ├── model           – fitted KMeans
  ├── imputer         – fitted SimpleImputer
  ├── scaler          – fitted StandardScaler
  ├── pca             – fitted PCA(n_components=2)
  ├── features        – ordered feature list
  ├── selected_k      – number of clusters
  ├── cluster_profiles – DataFrame of cluster means + sizes
  ├── profile_name_map – {cluster_id: descriptive_name}
  └── seed            – 42
```

---

## 11. Ethical Considerations

- **Synthetic data**: All results are educational artefacts, not real-world insights.
- **No medical advice**: Cluster assignments are statistical groupings only.
- **Bias**: Real-world data may carry demographic, socioeconomic, or measurement biases.
- **Privacy**: Real patient data requires strict privacy and security controls.
- **Insurance fairness**: Clustering must not be used to set premiums or deny coverage.
- **Stigmatisation**: Health labels can negatively affect individuals if misused.

---

*Generated automatically by the Patient Health Risk Segmentation training pipeline.*
