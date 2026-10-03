"""
=============================================================================
notebooks/analysis.py
Patient Health Risk Segmentation – EDA & Analysis Script
=============================================================================

This script mirrors a Jupyter notebook workflow:
  • Loads the generated dataset
  • Prints summaries and statistics
  • Generates all EDA plots

Run after train_model.py has created data/health_policyholders.csv:
    python notebooks/analysis.py
=============================================================================
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add parent directory to path so we can import constants from train_model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "health_policyholders.csv")
PLOTS_DIR  = os.path.join(BASE_DIR, "reports", "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

FEATURES = [
    "age", "bmi", "systolic_bp", "diastolic_bp",
    "cholesterol", "physical_activity", "previous_claims",
]


def load_data(path: str) -> pd.DataFrame:
    """Load the CSV dataset."""
    df = pd.read_csv(path)
    print(f"Loaded: {path}  ({df.shape[0]} rows × {df.shape[1]} cols)")
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print basic dataset information."""
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Shape        : {df.shape}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nDescriptive Statistics:\n{df[FEATURES].describe().round(2)}")


def plot_distributions(df: pd.DataFrame) -> None:
    """Histograms for all features."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()
    for i, col in enumerate(FEATURES):
        axes[i].hist(df[col].dropna(), bins=40, color="#4C72B0",
                     edgecolor="white", alpha=0.85)
        axes[i].set_title(col.replace("_", " ").title(), fontsize=12, fontweight="bold")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")
        axes[i].grid(axis="y", linestyle="--", alpha=0.5)
    for j in range(len(FEATURES), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Feature Distributions", fontsize=15, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "distributions.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_boxplots(df: pd.DataFrame) -> None:
    """Boxplots for all features."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()
    for i, col in enumerate(FEATURES):
        axes[i].boxplot(df[col].dropna(), patch_artist=True,
                        boxprops=dict(facecolor="#4C72B0", color="#333"),
                        medianprops=dict(color="white", linewidth=2))
        axes[i].set_title(col.replace("_", " ").title(), fontsize=12, fontweight="bold")
        axes[i].grid(axis="y", linestyle="--", alpha=0.5)
    for j in range(len(FEATURES), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Feature Boxplots", fontsize=15, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "boxplots.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_correlation(df: pd.DataFrame) -> None:
    """Correlation heatmap."""
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df[FEATURES].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.5, ax=ax,
                annot_kws={"size": 9})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_pairplot_subset(df: pd.DataFrame) -> None:
    """Pairplot for a subset of features."""
    subset_features = ["age", "bmi", "systolic_bp", "cholesterol", "physical_activity"]
    fig = sns.pairplot(df[subset_features].dropna(), diag_kind="kde",
                       plot_kws={"alpha": 0.5, "color": "#4C72B0"},
                       diag_kws={"color": "#4C72B0"})
    fig.fig.suptitle("Pairplot – Subset of Features", y=1.02, fontsize=13, fontweight="bold")
    out = os.path.join(PLOTS_DIR, "pairplot.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found.")
        print("Please run:  python train_model.py  first.")
        return

    df = load_data(DATA_PATH)
    print_summary(df)
    plot_distributions(df)
    plot_boxplots(df)
    plot_correlation(df)
    plot_pairplot_subset(df)
    print("\nAnalysis complete. All plots saved to reports/plots/")


if __name__ == "__main__":
    main()
