"""
Discretisation functions for continuous data.
"""

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer


def discretise(X, n_bins=5, strategy='uniform', categorical_col=None):
    """Discretise continuous data into bins.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Continuous input data.
    n_bins : int, default=5
        Number of bins to discretise each feature into.
    strategy : {'uniform', 'quantile', 'kmeans'}, default='uniform'
        Strategy used to define the widths of the bins.
    categorical_col : int or None, default=None
        If specified, this column index will be excluded from discretisation
        and reattached after (useful for preserving categorical features like minority groups).

    Returns
    -------
    Xd : array-like, shape (n_samples, n_features)
        Discretised data.
    """
    if categorical_col is not None:
        # Separate categorical column
        if categorical_col == -1:
            X_features = X[:, :-1]
            X_categorical = X[:, -1:]
        else:
            X_features = np.delete(X, categorical_col, axis=1)
            X_categorical = X[:, categorical_col:categorical_col+1]
        
        # Discretise only the feature columns
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
        Xd_features = discretizer.fit_transform(X_features)
        
        # Reattach categorical column
        if categorical_col == -1:
            Xd = np.column_stack([Xd_features, X_categorical])
        else:
            Xd = np.insert(Xd_features, categorical_col, X_categorical.flatten(), axis=1)
    else:
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
        Xd = discretizer.fit_transform(X)
    
    return Xd
