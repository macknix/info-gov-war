"""
Main entry point for the generate module.

Run with: python -m src.generate
"""

import yaml
import numpy as np

from .data import generate_regression_data
from .discretise import discretise
from .mondrian import anonymise_mondrian
from .plotting import plot_regression_stats


def main():
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    n_samples = config['n_samples']
    n_features = config['n_features']
    n_informative = config['n_informative']
    effective_rank = config['effective_rank']
    random_state = config['random_state']
    noise = config['noise']
    n_non_linear = config['n_non_linear']
    
    # Categorical feature settings
    include_categorical = config.get('include_categorical', False)
    n_categorical_classes = config.get('n_categorical_classes', None)
    categorical_weights = config.get('categorical_weights', None)
    
    # Generate data
    print("Generating regression data...")
    X, y = generate_regression_data(
        n_samples=n_samples, 
        n_features=n_features, 
        n_informative=n_informative, 
        effective_rank=effective_rank, 
        random_state=random_state, 
        noise=noise, 
        n_non_linear=n_non_linear,
        include_categorical=include_categorical,
        n_categorical_classes=n_categorical_classes,
        categorical_weights=categorical_weights
    )
    
    # Set categorical column index if categorical feature is included
    categorical_col = -1 if include_categorical else None
    
    print(f"Generated data shape: {X.shape}")
    if include_categorical:
        print(f"Categorical feature in column {categorical_col} with {n_categorical_classes} classes")
    
    # Plot original data
    print("\nPlotting original data...")
    plot_regression_stats(
        X, y, n_features, 
        random_state=random_state, 
        categorical_col=categorical_col, 
        title="Original Data"
    )
    
    # Plot discretised data
    print("\nDiscretising and plotting...")
    Xd = discretise(
        X, 
        n_bins=5, 
        strategy=config['discretisation_strategy'],
        categorical_col=categorical_col
    )
    plot_regression_stats(
        Xd, y, n_features, 
        random_state=random_state, 
        categorical_col=categorical_col, 
        title="Discretised Data"
    )
    
    # Plot Mondrian anonymised data
    print("\nApplying Mondrian anonymisation and plotting...")
    k = 5 # Use first k value from config
    X_anon, anonymiser = anonymise_mondrian(
        X, 
        k=k, 
        categorical_col=categorical_col
    )
    
    stats = anonymiser.get_stats()
    print(f"Mondrian k={k}: achieved k={stats['k_achieved']}, NCP={stats['ncp']:.2f}%")
    
    plot_regression_stats(
        X_anon, y, n_features, 
        random_state=random_state, 
        categorical_col=categorical_col, 
        title=f"Mondrian Anonymised (k={k})"
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
