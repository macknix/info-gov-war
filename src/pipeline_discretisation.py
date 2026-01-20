"""
Discretisation Pipeline

This pipeline evaluates the privacy-utility trade-off when using 
discretisation (binning) as an anonymisation method.

Steps:
1. Generate synthetic regression data
2. Plot raw data statistics
3. Compute k-anonymity metrics across discretisation levels
4. Run modelling experiments (utility + equity)
"""

import yaml
import numpy as np

from .generate import (
    generate_regression_data,
    discretise,
    plot_regression_stats,
)
from .k_anonymity import (
    eq_class_stats,
    plot_k_anonymity_vs_discretisation,
)
from .modelling import (
    run_regression_experiment,
    plot_regression_discretisation_results,
    plot_regression_equity_deciles,
)


def main(config_path: str = "config.yaml") -> None:
    """Run the discretisation pipeline."""
    
    # 1) Load config
    print("=" * 60)
    print("DISCRETISATION PIPELINE")
    print("=" * 60)
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Unpack parameters
    n_samples = config["n_samples"]
    n_features = config["n_features"]
    n_informative = config["n_informative"]
    effective_rank = config["effective_rank"]
    random_state = config["random_state"]
    noise = config["noise"]
    n_non_linear = config.get("n_non_linear", 0)
    discretisation_levels = config["discretisation_levels"]
    disc_strategy = config.get("discretisation_strategy", "uniform")

    # 2) Generate base regression data
    print(f"\n[1/4] Generating data: {n_samples} samples, {n_features} features...")
    X, y = generate_regression_data(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_non_linear=n_non_linear,
        effective_rank=effective_rank,
        noise=noise,
        random_state=random_state,
    )

    # 3) Basic data plots
    print("\n[2/4] Plotting raw data statistics...")
    plot_regression_stats(X, y, n_features, random_state=random_state)

    # 4) k-anonymity vs discretisation
    print("\n[3/4] Computing k-anonymity statistics across discretisation levels...")
    stats_by_bins = []
    for n_bins in discretisation_levels:
        Xd = discretise(X, n_bins=n_bins, strategy=disc_strategy)
        stats, sizes = eq_class_stats(Xd)
        stats["n_bins"] = n_bins
        stats_by_bins.append(stats)
        print(f"  {n_bins} bins: k_min={stats['k_min']}, "
              f"discernibility={stats['discernibility']:.2e}, "
              f"norm_EC={stats['normalised_eq_class_metric']:.2f}")

    print("\nPlotting k-anonymity vs discretisation...")
    plot_k_anonymity_vs_discretisation(stats_by_bins)

    # 5) Modelling experiment
    print("\n[4/4] Running regression experiments across discretisation levels...")
    results = run_regression_experiment(
        X,
        y,
        dbins=discretisation_levels,
        random_state=random_state,
    )

    print("\nPlotting utility/equity results...")
    plot_regression_discretisation_results(results)
    plot_regression_equity_deciles(results)

    print("\n" + "=" * 60)
    print("DISCRETISATION PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
