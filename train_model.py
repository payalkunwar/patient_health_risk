"""
=============================================================================
train_model.py
Patient Health Risk Segmentation – Complete Training Pipeline
=============================================================================

This script:
  1. Generates a synthetic dataset of policyholders.
  2. Introduces ~3% missing values.
  3. Performs Exploratory Data Analysis (EDA) and saves plots.
  4. Pre-processes data (imputation + scaling).
  5. Runs K-Means for K = 2..8, produces elbow + silhouette curves.
  6. Runs Agglomerative Hierarchical Clustering (Ward linkage) + dendrogram.
  7. Reduces dimensionality with PCA for a 2-D cluster visualization.
  8. Profiles each cluster using original feature values.
  9. Saves all artefacts (model bundle, CSVs, plots).

Run:
    python train_model.py
=============================================================================
"""

import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")            # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# GLOBAL SETTINGS
# ─────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

N_RECORDS   = 1200          # Number of synthetic policyholders
K_RANGE     = range(2, 9)   # K-Means: test K = 2 … 8
MISSING_PCT = 0.03          # ~3 % missing values

FEATURES = [
    "age",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "cholesterol",
    "physical_activity",
    "previous_claims",
]

COLS_WITH_MISSING = ["bmi", "systolic_bp", "diastolic_bp", "cholesterol"]

# Output directories
os.makedirs("data",          exist_ok=True)
os.makedirs("models",        exist_ok=True)
os.makedirs("reports/plots", exist_ok=True)


# =============================================================================
# STEP 1 – SYNTHETIC DATA GENERATION
# =============================================================================

def generate_synthetic_data(n: int = N_RECORDS, seed: int = SEED) -> pd.DataFrame:
    """
    Generate a realistic synthetic dataset with 3 latent health-indicator groups.

    The three latent profiles are:
        Group A – Lower-indicator  : younger, lower BMI/BP/cholesterol, more active
        Group B – Moderate-indicator : middle-range values across all indicators
        Group C – Higher-indicator : older, higher BMI/BP/cholesterol, less active

    These labels are NOT stored in the CSV; the ML algorithm discovers them.
    """
    rng = np.random.default_rng(seed)

    # Approximate proportions for the three latent groups
    sizes = [420, 420, n - 840]

    records = []

    # --- Group A: Lower-indicator profile ---
    # Younger, active, healthy BMI, optimal BP & cholesterol, low claim history
    nA = sizes[0]
    records.append({
        "age":               rng.normal(30, 6,   nA).clip(18, 48),
        "bmi":               rng.normal(21.5, 2.2, nA).clip(16, 26),
        "systolic_bp":       rng.normal(110, 6,  nA).clip(92, 126),
        "diastolic_bp":      rng.normal(70, 5,   nA).clip(56, 82),
        "cholesterol":       rng.normal(165, 16, nA).clip(115, 205),
        "physical_activity": rng.normal(8.5, 1.8, nA).clip(5.0, 12.0),
        "previous_claims":   rng.poisson(0.8,    nA).clip(0, 3),
    })

    # --- Group B: Moderate-indicator profile ---
    # Middle-aged, moderate BMI, borderline BP & cholesterol, moderate activity
    nB = sizes[1]
    records.append({
        "age":               rng.normal(49, 7,   nB).clip(34, 64),
        "bmi":               rng.normal(27.8, 2.4, nB).clip(22, 34),
        "systolic_bp":       rng.normal(132, 7,  nB).clip(116, 148),
        "diastolic_bp":      rng.normal(84, 5,   nB).clip(72, 94),
        "cholesterol":       rng.normal(218, 18, nB).clip(175, 260),
        "physical_activity": rng.normal(4.2, 1.2, nB).clip(1.8, 7.0),
        "previous_claims":   rng.poisson(2.6,    nB).clip(0, 6),
    })

    # --- Group C: Higher-indicator profile ---
    # Older, elevated BMI, hypertensive BP, higher cholesterol, sedentary, frequent claims
    nC = sizes[2]
    records.append({
        "age":               rng.normal(68, 7,   nC).clip(52, 85),
        "bmi":               rng.normal(34.5, 3.2, nC).clip(28, 45),
        "systolic_bp":       rng.normal(156, 9,  nC).clip(136, 190),
        "diastolic_bp":      rng.normal(98, 6,   nC).clip(84, 120),
        "cholesterol":       rng.normal(275, 20, nC).clip(230, 330),
        "physical_activity": rng.normal(1.2, 0.8, nC).clip(0.0, 3.5),
        "previous_claims":   rng.poisson(6.2,    nC).clip(2, 12),
    })

    # Concatenate all groups and shuffle
    df_parts = []
    for group_data in records:
        df_parts.append(pd.DataFrame(group_data))

    df = pd.concat(df_parts, ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Round to sensible precision
    df["age"]               = df["age"].round(0).astype(int)
    df["bmi"]               = df["bmi"].round(1)
    df["systolic_bp"]       = df["systolic_bp"].round(0).astype(int)
    df["diastolic_bp"]      = df["diastolic_bp"].round(0).astype(int)
    df["cholesterol"]       = df["cholesterol"].round(0).astype(int)
    df["physical_activity"] = df["physical_activity"].round(1)
    df["previous_claims"]   = df["previous_claims"].astype(int)

    # Prepend policyholder IDs
    df.insert(0, "policyholder_id", [f"PH{str(i+1).zfill(4)}" for i in range(len(df))])

    return df


def introduce_missing_values(df: pd.DataFrame,
                              cols: list,
                              pct: float = MISSING_PCT,
                              seed: int = SEED) -> pd.DataFrame:
    """
    Randomly set ~pct of values to NaN in the specified columns.
    This simulates missing clinical measurements in real data.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    for col in cols:
        n_missing = max(1, int(len(df) * pct))
        miss_idx  = rng.choice(df.index, size=n_missing, replace=False)
        df.loc[miss_idx, col] = np.nan
    return df


# =============================================================================
# STEP 2 – EXPLORATORY DATA ANALYSIS
# =============================================================================

def run_eda(df: pd.DataFrame) -> None:
    """
    Print summary statistics and save EDA plots to reports/plots/.
    """
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)

    print(f"\nDataset shape     : {df.shape}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nDescriptive statistics:\n{df[FEATURES].describe().round(2)}")

    numeric_cols = FEATURES  # exclude id

    # ── 1. Distribution plots (histograms + KDE) ──────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col].dropna(), bins=35, color="#4C72B0", edgecolor="white",
                     alpha=0.85)
        axes[i].set_title(col.replace("_", " ").title(), fontsize=12, fontweight="bold")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Count")
        axes[i].grid(axis="y", linestyle="--", alpha=0.5)
    # Hide unused subplot
    for j in range(len(numeric_cols), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Feature Distributions", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("reports/plots/distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n[EDA] Saved: reports/plots/distributions.png")

    # ── 2. Boxplots ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        axes[i].boxplot(df[col].dropna(), patch_artist=True,
                        boxprops=dict(facecolor="#4C72B0", color="#333"),
                        medianprops=dict(color="white", linewidth=2))
        axes[i].set_title(col.replace("_", " ").title(), fontsize=12, fontweight="bold")
        axes[i].grid(axis="y", linestyle="--", alpha=0.5)
    for j in range(len(numeric_cols), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Feature Boxplots", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("reports/plots/boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[EDA] Saved: reports/plots/boxplots.png")

    # ── 3. Correlation heatmap ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 9})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig("reports/plots/correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[EDA] Saved: reports/plots/correlation_heatmap.png")


# =============================================================================
# STEP 3 – PREPROCESSING
# =============================================================================

def preprocess_features(df: pd.DataFrame):
    """
    1. Extract the feature matrix X (raw values).
    2. Impute missing values with median (fit on this dataset).
    3. Standardise with StandardScaler (fit on this dataset).

    Returns:
        X_raw     – original feature values (for profiling)
        X_scaled  – imputed + scaled values (for clustering)
        imputer   – fitted SimpleImputer
        scaler    – fitted StandardScaler
    """
    X_raw = df[FEATURES].copy()

    # Median imputation – fit and transform on the full dataset
    # (acceptable for unsupervised learning; no leakage concern)
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_raw)

    # StandardScaler – required because K-Means and hierarchical clustering
    # are distance-based algorithms; unscaled features with larger ranges
    # (e.g., cholesterol: 110–330) would dominate smaller-range features
    # (e.g., previous_claims: 0–12).
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_raw, X_scaled, imputer, scaler


# =============================================================================
# STEP 4 – K-MEANS CLUSTERING
# =============================================================================

def run_kmeans_analysis(X_scaled: np.ndarray):
    """
    Test K-Means for K = 2 … 8.
    Calculate WCSS (inertia) and silhouette score for each K.
    Return metrics dict and the selected K with the fitted model.
    """
    print("\n" + "="*60)
    print("K-MEANS ANALYSIS  (K = 2 … 8)")
    print("="*60)

    wcss        = {}
    silhouettes = {}

    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=20)
        labels = km.fit_predict(X_scaled)
        wcss[k]        = km.inertia_
        silhouettes[k] = silhouette_score(X_scaled, labels)
        print(f"  K={k}  |  WCSS={wcss[k]:>10.2f}  |  Silhouette={silhouettes[k]:.4f}")

    # ── Select optimal K ──────────────────────────────────────────────────
    # Use silhouette score as the primary criterion (higher = better separation),
    # guided by the elbow curve for interpretability.
    # We also cap at K=5 to keep clusters interpretable for a healthcare context.
    candidate_scores = {k: silhouettes[k] for k in K_RANGE if k <= 5}
    selected_k = max(candidate_scores, key=candidate_scores.get)

    print(f"\n  → Selected K = {selected_k}  "
          f"(highest silhouette among K=2..5: {silhouettes[selected_k]:.4f})")

    # ── Final K-Means model with selected K ───────────────────────────────
    km_final = KMeans(n_clusters=selected_k, random_state=SEED, n_init=20)
    km_labels = km_final.fit_predict(X_scaled)

    return wcss, silhouettes, selected_k, km_final, km_labels


def plot_elbow_curve(wcss: dict, selected_k: int) -> None:
    """Save the WCSS elbow curve with the selected K highlighted."""
    ks   = list(wcss.keys())
    vals = list(wcss.values())

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ks, vals, "o-", color="#4C72B0", linewidth=2.5, markersize=8,
            label="WCSS")
    ax.axvline(selected_k, color="#DD4444", linestyle="--", linewidth=1.8,
               label=f"Selected K = {selected_k}")
    ax.scatter([selected_k], [wcss[selected_k]], color="#DD4444", s=150, zorder=5)
    ax.set_xlabel("Number of Clusters (K)", fontsize=12)
    ax.set_ylabel("WCSS  (Within-Cluster Sum of Squares)", fontsize=12)
    ax.set_title("Elbow Curve – K-Means", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("reports/plots/elbow_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[K-Means] Saved: reports/plots/elbow_curve.png")


def plot_silhouette_curve(silhouettes: dict, selected_k: int) -> None:
    """Save the silhouette score curve with the selected K highlighted."""
    ks   = list(silhouettes.keys())
    vals = list(silhouettes.values())

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ks, vals, "s-", color="#55A868", linewidth=2.5, markersize=8,
            label="Silhouette Score")
    ax.axvline(selected_k, color="#DD4444", linestyle="--", linewidth=1.8,
               label=f"Selected K = {selected_k}")
    ax.scatter([selected_k], [silhouettes[selected_k]],
               color="#DD4444", s=150, zorder=5)
    ax.set_xlabel("Number of Clusters (K)", fontsize=12)
    ax.set_ylabel("Silhouette Score", fontsize=12)
    ax.set_title("Silhouette Score – K-Means", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("reports/plots/silhouette_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[K-Means] Saved: reports/plots/silhouette_curve.png")


# =============================================================================
# STEP 5 – HIERARCHICAL CLUSTERING
# =============================================================================

def run_hierarchical_clustering(X_scaled: np.ndarray, selected_k: int):
    """
    Fit AgglomerativeClustering with Ward linkage using selected_k clusters.
    Also compute a linkage matrix for the dendrogram.
    Returns: hierarchical labels, linkage matrix, silhouette score.
    """
    print("\n" + "="*60)
    print("HIERARCHICAL CLUSTERING  (Ward linkage)")
    print("="*60)

    # For the dendrogram we need the linkage matrix (scipy)
    Z = linkage(X_scaled, method="ward")

    # Fit AgglomerativeClustering
    agg = AgglomerativeClustering(n_clusters=selected_k, linkage="ward")
    hc_labels = agg.fit_predict(X_scaled)

    hc_sil = silhouette_score(X_scaled, hc_labels)
    print(f"  Hierarchical Silhouette Score (K={selected_k}): {hc_sil:.4f}")

    return hc_labels, Z, hc_sil


def plot_dendrogram(Z: np.ndarray, selected_k: int) -> None:
    """
    Save a dendrogram truncated to the last 30 merges for readability.
    A horizontal line is drawn at the cut threshold that would yield selected_k clusters.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot truncated dendrogram
    dend = dendrogram(Z, ax=ax, truncate_mode="lastp", p=30,
                      leaf_rotation=45, leaf_font_size=9,
                      color_threshold=0,
                      above_threshold_color="#4C72B0")

    # Determine the threshold height for selected_k clusters
    # The linkage distances in Z[:,2] are sorted ascending.
    # To get n_clusters, we cut at the (n-selected_k)th merge from bottom,
    # i.e. just below the (N - selected_k + 1)-th largest distance.
    distances = sorted(Z[:, 2], reverse=True)
    # threshold sits between the (selected_k-1)-th and selected_k-th largest distances
    threshold = (distances[selected_k - 2] + distances[selected_k - 1]) / 2
    ax.axhline(y=threshold, color="#DD4444", linestyle="--", linewidth=1.8,
               label=f"Cut for K={selected_k} clusters")

    ax.set_title(f"Dendrogram – Hierarchical Clustering (Ward Linkage)\n"
                 f"Truncated to last 30 merges", fontsize=13, fontweight="bold")
    ax.set_xlabel("Sample Index / Cluster", fontsize=11)
    ax.set_ylabel("Linkage Distance", fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("reports/plots/dendrogram.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Hierarchical] Saved: reports/plots/dendrogram.png")


# =============================================================================
# STEP 6 – PCA VISUALIZATION
# =============================================================================

def run_pca_and_plot(X_scaled: np.ndarray, km_labels: np.ndarray,
                     selected_k: int):
    """
    Fit PCA(n_components=2), project scaled data, and plot coloured by K-Means label.
    Returns: fitted PCA transformer, 2-D projection array.
    """
    print("\n" + "="*60)
    print("PCA – Dimensionality Reduction")
    print("="*60)

    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_scaled)

    ev = pca.explained_variance_ratio_
    print(f"  PC1 explained variance : {ev[0]*100:.2f}%")
    print(f"  PC2 explained variance : {ev[1]*100:.2f}%")
    print(f"  Total (PC1+PC2)        : {(ev[0]+ev[1])*100:.2f}%")

    # Colour palette
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    colours = [palette[i % len(palette)] for i in km_labels]

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                         c=km_labels, cmap="tab10",
                         s=35, alpha=0.75, edgecolors="white", linewidths=0.3)

    # Cluster centroids in PCA space
    for k in range(selected_k):
        cx = X_pca[km_labels == k, 0].mean()
        cy = X_pca[km_labels == k, 1].mean()
        ax.scatter(cx, cy, marker="X", s=220, c="black", zorder=10)
        ax.annotate(f"C{k}", (cx, cy), fontsize=10, fontweight="bold",
                    textcoords="offset points", xytext=(8, 5))

    handles = [mpatches.Patch(color=scatter.cmap(scatter.norm(k)),
                              label=f"Cluster {k}") for k in range(selected_k)]
    ax.legend(handles=handles, title="K-Means Cluster", fontsize=10,
              title_fontsize=10, loc="best")
    ax.set_xlabel(f"PC1  ({ev[0]*100:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2  ({ev[1]*100:.1f}% variance)", fontsize=12)
    ax.set_title("PCA – 2D Cluster Visualization (K-Means Labels)",
                 fontsize=14, fontweight="bold")
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("reports/plots/pca_clusters.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[PCA] Saved: reports/plots/pca_clusters.png")

    return pca, X_pca


# =============================================================================
# STEP 7 – CLUSTER PROFILING
# =============================================================================

def profile_clusters(df: pd.DataFrame,
                     km_labels: np.ndarray,
                     hc_labels: np.ndarray,
                     selected_k: int):
    """
    Compute per-cluster mean statistics on ORIGINAL (unscaled) feature values.
    Assign descriptive profile names based on the composite health indicator ranking.

    Returns:
        cluster_profiles      – DataFrame with mean values per cluster
        cluster_profiles_named– same + descriptive name column
        cluster_sizes         – DataFrame with cluster sizes
        profile_name_map      – dict {cluster_id: descriptive_name}
    """
    print("\n" + "="*60)
    print("CLUSTER PROFILING")
    print("="*60)

    df_profile = df[FEATURES].copy()
    df_profile["km_cluster"] = km_labels
    df_profile["hc_cluster"] = hc_labels

    # ── K-Means profiles ──────────────────────────────────────────────────
    km_profile = (
        df_profile.groupby("km_cluster")[FEATURES]
        .mean()
        .round(2)
        .reset_index()
        .rename(columns={"km_cluster": "cluster"})
    )

    # Cluster sizes
    sizes = pd.Series(km_labels).value_counts().reset_index()
    sizes.columns = ["cluster", "size"]
    sizes = sizes.sort_values("cluster").reset_index(drop=True)

    km_profile = km_profile.merge(sizes, on="cluster")

    # ── Assign descriptive names ──────────────────────────────────────────
    # Composite score: we rank clusters by (age + bmi + systolic_bp + cholesterol)
    # Higher composite → "Higher-indicator"; lower → "Lower-indicator"
    km_profile["composite_score"] = (
        km_profile["age"]
        + km_profile["bmi"]
        + km_profile["systolic_bp"]
        + km_profile["cholesterol"]
    )

    km_profile_sorted = km_profile.sort_values("composite_score").reset_index(drop=True)

    if selected_k == 2:
        profile_labels = ["Lower-indicator profile", "Higher-indicator profile"]
    elif selected_k == 3:
        profile_labels = ["Lower-indicator profile",
                          "Moderate-indicator profile",
                          "Higher-indicator profile"]
    else:
        # Generalise for any K
        n = selected_k
        if n == 4:
            profile_labels = ["Lower-indicator profile",
                              "Lower-moderate profile",
                              "Upper-moderate profile",
                              "Higher-indicator profile"]
        elif n == 5:
            profile_labels = ["Lower-indicator profile",
                              "Lower-moderate profile",
                              "Moderate-indicator profile",
                              "Upper-moderate profile",
                              "Higher-indicator profile"]
        else:
            profile_labels = [f"Profile-{i+1}" for i in range(n)]

    # Map cluster id → descriptive name
    profile_name_map = {}
    for rank, row in km_profile_sorted.iterrows():
        profile_name_map[int(row["cluster"])] = profile_labels[rank]

    km_profile["profile_name"] = km_profile["cluster"].map(profile_name_map)
    km_profile.drop(columns=["composite_score"], inplace=True)

    # Print profiles
    for _, row in km_profile.iterrows():
        print(f"\n  Cluster {int(row['cluster'])}  ({row['profile_name']})")
        print(f"    Size              : {int(row['size'])}")
        print(f"    Avg Age           : {row['age']:.1f}")
        print(f"    Avg BMI           : {row['bmi']:.1f}")
        print(f"    Avg Systolic BP   : {row['systolic_bp']:.1f}")
        print(f"    Avg Diastolic BP  : {row['diastolic_bp']:.1f}")
        print(f"    Avg Cholesterol   : {row['cholesterol']:.1f}")
        print(f"    Avg Physical Act. : {row['physical_activity']:.1f} hrs/week")
        print(f"    Avg Prev. Claims  : {row['previous_claims']:.1f}")

    # ── Save CSVs ─────────────────────────────────────────────────────────
    # cluster_profiles.csv  – without profile name column
    cluster_profiles = km_profile.drop(columns=["profile_name"])
    cluster_profiles.to_csv("reports/cluster_profiles.csv", index=False)
    print("\n[Profile] Saved: reports/cluster_profiles.csv")

    # cluster_profiles_named.csv – with profile name column
    km_profile.to_csv("reports/cluster_profiles_named.csv", index=False)
    print("[Profile] Saved: reports/cluster_profiles_named.csv")

    # cluster_sizes.csv
    sizes_out = sizes.copy()
    sizes_out["profile_name"] = sizes_out["cluster"].map(profile_name_map)
    sizes_out.to_csv("reports/cluster_sizes.csv", index=False)
    print("[Profile] Saved: reports/cluster_sizes.csv")

    return km_profile, sizes_out, profile_name_map


# =============================================================================
# STEP 8 – MODEL METRICS REPORT
# =============================================================================

def save_model_metrics(wcss: dict,
                       silhouettes: dict,
                       selected_k: int,
                       hc_sil: float) -> None:
    """Save a summary CSV of all K-Means + Hierarchical metrics."""
    rows = []
    for k in K_RANGE:
        rows.append({
            "model":           "K-Means",
            "n_clusters":      k,
            "wcss":            round(wcss[k], 4),
            "silhouette":      round(silhouettes[k], 4),
            "selected":        (k == selected_k),
        })
    rows.append({
        "model":      "Hierarchical (Ward)",
        "n_clusters": selected_k,
        "wcss":       None,
        "silhouette": round(hc_sil, 4),
        "selected":   True,
    })
    pd.DataFrame(rows).to_csv("reports/model_metrics.csv", index=False)
    print("[Metrics] Saved: reports/model_metrics.csv")


# =============================================================================
# STEP 9 – SAVE MODEL BUNDLE
# =============================================================================

def save_model_bundle(km_model,
                      imputer,
                      scaler,
                      pca,
                      selected_k: int,
                      cluster_profiles: pd.DataFrame,
                      profile_name_map: dict) -> None:
    """
    Save all deployment artefacts in a single Joblib bundle so the
    Streamlit app can load them without retraining.
    """
    bundle = {
        "model":           km_model,          # fitted KMeans
        "imputer":         imputer,            # fitted SimpleImputer
        "scaler":          scaler,             # fitted StandardScaler
        "pca":             pca,                # fitted PCA(n_components=2)
        "features":        FEATURES,           # ordered feature list
        "selected_k":      selected_k,
        "cluster_profiles": cluster_profiles,  # DataFrame (means + size)
        "profile_name_map": profile_name_map,  # {cluster_id: name}
        "seed":            SEED,
    }
    joblib.dump(bundle, "models/model_bundle.joblib")
    print("[Bundle] Saved: models/model_bundle.joblib")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("PATIENT HEALTH RISK SEGMENTATION – TRAINING PIPELINE")
    print("="*60)

    # 1. Generate data
    print("\n[1/9] Generating synthetic dataset …")
    df = generate_synthetic_data(N_RECORDS, SEED)
    df = introduce_missing_values(df, COLS_WITH_MISSING, MISSING_PCT, SEED)
    df.to_csv("data/health_policyholders.csv", index=False)
    print(f"      Saved: data/health_policyholders.csv  ({len(df)} records)")

    # 2. EDA
    print("\n[2/9] Running EDA …")
    run_eda(df)

    # 3. Preprocess
    print("\n[3/9] Preprocessing features …")
    X_raw, X_scaled, imputer, scaler = preprocess_features(df)
    print(f"      Feature matrix shape: {X_scaled.shape}")

    # 4. K-Means
    print("\n[4/9] K-Means clustering …")
    wcss, silhouettes, selected_k, km_model, km_labels = run_kmeans_analysis(X_scaled)
    plot_elbow_curve(wcss, selected_k)
    plot_silhouette_curve(silhouettes, selected_k)

    # 5. Hierarchical clustering
    print("\n[5/9] Hierarchical clustering …")
    hc_labels, Z, hc_sil = run_hierarchical_clustering(X_scaled, selected_k)
    plot_dendrogram(Z, selected_k)

    # 6. PCA
    print("\n[6/9] PCA …")
    pca, X_pca = run_pca_and_plot(X_scaled, km_labels, selected_k)

    # 7. Cluster profiling
    print("\n[7/9] Profiling clusters …")
    cluster_profiles, cluster_sizes, profile_name_map = profile_clusters(
        df, km_labels, hc_labels, selected_k
    )

    # 8. Save metrics
    print("\n[8/9] Saving model metrics …")
    save_model_metrics(wcss, silhouettes, selected_k, hc_sil)

    # 9. Save model bundle
    print("\n[9/9] Saving model bundle …")
    save_model_bundle(km_model, imputer, scaler, pca,
                      selected_k, cluster_profiles, profile_name_map)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"  Selected K          : {selected_k}")
    print(f"  KM Silhouette Score : {silhouettes[selected_k]:.4f}")
    print(f"  HC Silhouette Score : {hc_sil:.4f}")
    print("\nAll outputs saved. Run: streamlit run app/app.py")


if __name__ == "__main__":
    main()
