"""
Data generation functions for synthetic regression datasets.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.datasets import make_regression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer


def generate_gaussian_mixture(N, D, K=2, signal_fraction=0.1, snr=1.0, seed=None, class_balance='balanced'):
    """Generate a Gaussian mixture dataset for classification.
    
    Parameters
    ----------
    N : int
        Number of samples
    D : int
        Number of features (dimensions)
    K : int, default=2
        Number of classes
    signal_fraction : float, default=0.1
        Fraction of features that are informative
    snr : float, default=1.0
        Signal-to-noise ratio
    seed : int or None
        Random seed
    class_balance : str or array-like
        Class balance strategy: 'balanced', 'imbalanced', 'power_law', or custom probabilities
        
    Returns
    -------
    X : np.ndarray of shape (N, D)
        Feature matrix
    y : np.ndarray of shape (N,)
        Class labels
    """
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
        # Exponentially decreasing probabilities
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
        random_state=None,
        include_categorical=False,
        n_categorical_classes=None,
        categorical_weights=None
        ):
    """Generate a synthetic regression dataset.
    
    Parameters
    ----------
    n_samples : int, default=100
        Number of samples
    n_features : int, default=100
        Number of features
    n_informative : int, default=10
        Number of informative features
    n_non_linear : int, default=0
        Number of features to apply non-linear transformations to
    n_targets : int, default=1
        Number of targets
    bias : float, default=0.0
        Bias term
    effective_rank : int, default=10
        Approximate number of singular vectors required to explain most of the data
    tail_strength : float, default=0.01
        Relative importance of the fat noisy tail of the singular values
    noise : float, default=0.0
        Standard deviation of the gaussian noise applied to the output
    shuffle : bool, default=True
        Whether to shuffle the samples and features
    coef : bool, default=False
        Whether to return the coefficients of the underlying linear model
    random_state : int or None
        Random seed
    include_categorical : bool, default=False
        Whether to include a categorical feature representing a minority group (e.g., ethnicity)
    n_categorical_classes : int or None
        Number of classes for the categorical feature. Required if include_categorical is True.
    categorical_weights : list/array or str
        Weights for each class controlling minority representation.
        Can be:
        - 'uniform': equal probability for all classes
        - 'minority': first class is majority (50%), rest share remaining 50%
        - list/array of floats that sum to 1.0
        If None and include_categorical is True, defaults to 'uniform'.
        
    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features) or (n_samples, n_features + 1) if include_categorical
        Feature matrix
    y : np.ndarray of shape (n_samples,)
        Target values
    """
    rng = check_random_state(random_state)
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

    # Generate categorical feature for minority group representation (e.g., ethnicity)
    if include_categorical:
        if n_categorical_classes is None:
            raise ValueError("n_categorical_classes must be specified when include_categorical is True")
        
        # Determine class probabilities based on categorical_weights
        if categorical_weights is None or categorical_weights == 'uniform':
            # Equal probability for all classes
            class_probs = np.ones(n_categorical_classes) / n_categorical_classes
        elif categorical_weights == 'minority':
            # First class is majority (50%), rest share remaining 50%
            class_probs = np.zeros(n_categorical_classes)
            class_probs[0] = 0.5
            if n_categorical_classes > 1:
                remaining_prob = 0.5 / (n_categorical_classes - 1)
                class_probs[1:] = remaining_prob
        else:
            # Custom weights provided as list/array
            class_probs = np.array(categorical_weights, dtype=float)
            if len(class_probs) != n_categorical_classes:
                raise ValueError(f"categorical_weights must have length {n_categorical_classes}, got {len(class_probs)}")
            if not np.isclose(class_probs.sum(), 1.0):
                raise ValueError(f"categorical_weights must sum to 1.0, got {class_probs.sum()}")
        
        # Generate categorical labels based on probabilities
        categorical_feature = rng.choice(n_categorical_classes, size=n_samples, p=class_probs)
        
        # Add categorical feature as the last column of X
        X = np.column_stack([X, categorical_feature])

    return X, y
