# simulation_pipeline.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import silhouette_score
from sklearn.utils import check_random_state
from collections import Counter
from sklearn.metrics import silhouette_score
from sklearn.utils import check_random_state
from collections import Counter
import itertools
import time
import joblib  # for saving models / results
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer


# ---------- Data generation ----------
def generate_gaussian_mixture(N, D, K=2, signal_fraction=0.1, snr=1.0, seed=None, class_balance='balanced'):
    rng = check_random_state(seed)
    # number of informative dims
    d_inf = max(1, int(np.round(D * signal_fraction)))
    # create class means separated on informative dims

    mus = rng.normal(scale=1.0, size=(K, d_inf))
    mus *= snr
    # covariance
    cov_inf = np.eye(d_inf)

    # latent variable that adds inter-feature correlation
    z = rng.normal(scale=1.0, size=(N, 1))

    # Determine class probabilities
    if class_balance == 'balanced':
        class_probs = np.ones(K) / K
    elif class_balance == 'imbalanced':
        # Exponentially decreasing probabil
        # ities
        class_probs = np.array([2**(-(i)) for i in range(K)])
        class_probs /= class_probs.sum()
    elif class_balance == 'power_law':
        # Power law distribution (minority classes are very small)
        class_probs = np.array([(i+1)**(-1.5) for i in range(K)])
        class_probs /= class_probs.sum()
    else:
        # Custom probabilities
        class_probs = np.array(class_balance)
        if len(class_probs) != K:
            raise ValueError(f"class_balance array must have length K={K}")
        if not np.isclose(class_probs.sum(), 1.0):
            raise ValueError("class_balance probabilities must sum to 1")

    # initialize data arrays
    X = np.zeros((N, D))
    y = np.zeros(N, dtype=int)

    # Sample class labels according to probabilities
    y = rng.choice(K, size=N, p=class_probs)

    for i in range(N):
        # assign class to sample
        c = y[i]

        # generate data with signal and noise
        if d_inf > 0:
            x_inf = rng.multivariate_normal(mean=mus[c], cov=cov_inf)
        else:
            x_inf = np.empty(0)
        if D - d_inf > 0:
            x_noise = rng.normal(scale=1.0, size=(D - d_inf,))
        else:
            x_noise = np.empty(0)

        # combine informative and noise features
        X[i, :] = np.concatenate([x_inf, x_noise])
    # shuffle feature columns so informative ones are not grouped
    perm = rng.permutation(D)
    X = X[:, perm]

    # finally we add some global correlation via latent variable z
    a = 0.3  # scaling factor for latent variable
    X = X + a * z  # broadcasting

    return X, y

def discretise(X, n_bins=5, strategy='uniform'):
    """Discretise continuous data into bins.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Continuous input data.
    n_bins : int, optional (default=5)
        Number of bins to discretise each feature into.
    strategy : {'uniform', 'quantile', 'kmeans'}, optional (default='uniform')
        Strategy used to define the widths of the bins.

    Returns
    -------
    Xd : array-like, shape (n_samples, n_features)
        Discretised data.
    """
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    Xd = discretizer.fit_transform(X)
    return Xd

def plot_regression_stats(X, y, n_features, random_state=None):
            # ---- basic plots + compressibility ----
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1) Feature 0 vs target
        axes[0, 0].scatter(X[:, 0], y, alpha=0.5)
        axes[0, 0].set_xlabel("Feature 0")
        axes[0, 0].set_ylabel("y")
        axes[0, 0].set_title("Feature 0 vs target")

        # 2) Distribution of first few features
        max_feats = min(4, n_features)
        for j in range(max_feats):
            axes[0, 1].hist(X[:, j], bins=30, alpha=0.5, label=f"Feature {j}")
        axes[0, 1].set_title("Distribution of first features")
        axes[0, 1].set_xlabel("Value")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].legend()

        # 3) Compressibility via PCA explained variance
        n_pca = min(20, n_features)
        pca = PCA(n_components=n_pca, random_state=random_state)
        pca.fit(X)
        evr = pca.explained_variance_ratio_
        cum_evr = evr.cumsum()

        axes[1, 0].bar(range(1, n_pca + 1), evr, alpha=0.7, label="per-component")
        axes[1, 0].plot(range(1, n_pca + 1), cum_evr, marker="o", color="C1", label="cumulative")
        axes[1, 0].set_xlabel("Principal component")
        axes[1, 0].set_ylabel("Explained variance ratio")
        axes[1, 0].set_title("PCA spectrum (compressibility)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4) Correlation heatmap of features (subset)
        n_corr_feats = min(20, n_features)
        corr = np.corrcoef(X[:, :n_corr_feats].T)
        sns.heatmap(
            corr,
            ax=axes[1, 1],
            cmap="coolwarm",
            center=0,
            cbar_kws={"label": "Correlation"},
        )
        axes[1, 1].set_title(f"Feature correlation (first {n_corr_feats} features)")

        plt.tight_layout()
        plt.show()



def generate_regression_data(
        n_samples=100, 
        n_features=100,
        n_informative=10,
        n_non_linear=0,
        n_targets=1,
        bias=0.0,
        effective_rank=10,
        tail_strength=0.01,
        noise=0.0,
        shuffle=True,
        coef=False,
        random_state=None
        ):
    """Generate a synthetic regression dataset and optionally plot basic statistics.
    args:
        n_samples: int, number of samples
        n_features: int, number of features
        n_informative: int, number of informative features
        n_targets: int, number of targets
        bias: float, bias term
        effective_rank: int, approximate number of singular vectors required to explain most of the data by linear combinations
        tail_strength: float, relative importance of the fat noisy tail of the singular values
        noise: float, standard deviation of the gaussian noise applied to the output
        shuffle: bool, whether to shuffle the samples and features
        coef: bool, whether to return the coefficients of the underlying linear model
        random_state: int or None, random seed
        plot: bool, whether to plot basic statistics of the generated data
    """
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_targets=n_targets,
        bias=bias,
        effective_rank=effective_rank,
        tail_strength=tail_strength,
        noise=noise,
        shuffle=shuffle,
        coef=coef,
        random_state=random_state)

    available_transformers = {
        'log': FunctionTransformer(np.log1p, validate=True),
        'sqrt': FunctionTransformer(np.sqrt, validate=True),
        'square': FunctionTransformer(np.square, validate=True),
        'sin': FunctionTransformer(np.sin, validate=True),
        'exp': FunctionTransformer(lambda x: np.exp(0.1*x), validate=True)
    }

    transformers = []
    for n in range(n_non_linear):
        # randomly select a transformer
        t_name = np.random.choice(list(available_transformers.keys()))
        t = available_transformers[t_name]
        transformers.append((f"{t_name}_feat_{n}", t, [n]))
    print(transformers)

    if len(transformers) > 0:
        ct = ColumnTransformer(transformers, remainder='passthrough')
        X = ct.fit_transform(X)

    # shuffle columns to mix transformed features
    rng = check_random_state(random_state)
    perm = rng.permutation(X.shape[1])
    X = X[:, perm]

    # fill numpy nans with zeros
    X = np.nan_to_num(X, nan=0.0)

    # standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


if __name__ == "__main__":
    import yaml
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    n_sample = config['n_samples']
    n_features = config['n_features']
    n_informative = config['n_informative']
    effective_rank = config['effective_rank']
    random_state = config['random_state']
    noise = config['noise']
    n_non_linear = config['n_non_linear']
    X, y = generate_regression_data(n_samples=n_sample, n_features=n_features, n_informative=n_informative, effective_rank=effective_rank, random_state=random_state, noise=noise, n_non_linear=n_non_linear)
    plot_regression_stats(X, y, n_features, random_state=random_state)
    Xd = discretise(X, n_bins=5, strategy=config['discretisation_strategy'])
    plot_regression_stats(Xd, y, n_features, random_state=random_state)