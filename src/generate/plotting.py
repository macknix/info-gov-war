"""
Plotting functions for data visualisation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def plot_regression_stats(X, y, n_features, random_state=None, categorical_col=None, feature_num=10, title=None):
    """Plot regression statistics including optional categorical feature distribution.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target values
    n_features : int
        Number of features (excluding categorical if present)
    random_state : int or None
        Random seed for PCA
    categorical_col : int or None
        Column index of categorical feature (e.g., -1 for last column), or None if no categorical
    feature_num : int, default=10
        Number of features to show in distribution plot
    title : str or None
        Optional title for the figure (e.g., "Original Data" or "Mondrian Anonymised (k=10)")
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    if title is not None:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    # 1) Distribution of first few features
    max_feats = min(feature_num, n_features)
    for j in range(max_feats):
        axes[0, 0].hist(X[:, j], bins=30, alpha=0.5, label=f"Feature {j}")
    axes[0, 0].set_title("Distribution of first features")
    axes[0, 0].set_xlabel("Value")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].legend()

    # 2) Compressibility via PCA explained variance
    n_pca = min(20, n_features)
    pca = PCA(n_components=n_pca, random_state=random_state)
    pca.fit(X[:, :n_features])  # Only use feature columns for PCA
    evr = pca.explained_variance_ratio_
    cum_evr = evr.cumsum()

    axes[0, 1].bar(range(1, n_pca + 1), evr, alpha=0.7, label="per-component")
    axes[0, 1].plot(range(1, n_pca + 1), cum_evr, marker="o", color="C1", label="cumulative")
    axes[0, 1].set_xlabel("Principal component")
    axes[0, 1].set_ylabel("Explained variance ratio")
    axes[0, 1].set_title("PCA spectrum (compressibility)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3) Correlation heatmap of features (subset)
    n_corr_feats = min(20, n_features)
    corr = np.corrcoef(X[:, :n_corr_feats].T)
    sns.heatmap(
        corr,
        ax=axes[1, 0],
        cmap="coolwarm",
        center=0,
        cbar_kws={"label": "Correlation"},
    )
    axes[1, 0].set_title(f"Feature correlation (first {n_corr_feats} features)")

    # 4) Categorical violin plot or empty
    if categorical_col is not None:
        categorical_data = X[:, categorical_col].astype(int)
        unique_classes, counts = np.unique(categorical_data, return_counts=True)
        percentages = counts / len(categorical_data) * 100
        
        # Create DataFrame for violin plot
        df_violin = pd.DataFrame({
            'Class': categorical_data,
            'Target': y
        })
        
        # Violin plot: target distribution by minority class
        colors = plt.cm.Set2(np.linspace(0, 1, len(unique_classes)))
        palette = {c: colors[i] for i, c in enumerate(unique_classes)}
        sns.violinplot(data=df_violin, x='Class', y='Target', ax=axes[1, 1], 
                      hue='Class', palette=palette, legend=False)
        axes[1, 1].set_xlabel("Minority Group Class")
        axes[1, 1].set_ylabel("Target Value")
        axes[1, 1].set_title("Target Distribution by Minority Group")
        
        # Add sample size annotations below each violin
        for i, (cls, count, pct) in enumerate(zip(unique_classes, counts, percentages)):
            axes[1, 1].annotate(f'n={count}\n({pct:.1f}%)',
                xy=(i, axes[1, 1].get_ylim()[0]),
                xytext=(0, -25), textcoords="offset points",
                ha='center', va='top', fontsize=8)
    else:
        axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()
