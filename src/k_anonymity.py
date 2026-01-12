from collections import Counter
import numpy as np
from .generate import generate_gaussian_mixture, discretise,  generate_regression_data
import matplotlib.pyplot as plt

def eq_class_stats(X_disc):
    # treat each row as tuple key
    keys = [tuple(row) for row in X_disc]
    counts = Counter(keys)
    sizes = np.array([counts[k] for k in keys])
    return {
        'k_min': int(sizes.min()),
        'k_median': float(np.median(sizes)),
        'mean_size': float(np.mean(sizes)),
        'fraction_k_eq_1': float((sizes == 1).mean()),   # NEW: singletons
        'fraction_k_ge_2': float((sizes >= 2).mean()),
        'fraction_k_ge_5': float((sizes >= 5).mean()),
        'fraction_k_ge_10': float((sizes >= 10).mean())
    }, sizes

def plot_k_anonymity_vs_discretisation(stats_by_bins):
    """
    Given a list of dicts (one per discretisation level) with k-anonymity stats,
    produce summary plots.
    """
    # Prepare data for plotting
    bins = [s['n_bins'] for s in stats_by_bins]
    k_min = [s['k_min'] for s in stats_by_bins]
    k_median = [s['k_median'] for s in stats_by_bins]
    mean_size = [s['mean_size'] for s in stats_by_bins]
    frac_eq1 = [s['fraction_k_eq_1'] for s in stats_by_bins] 
    frac_ge2 = [s['fraction_k_ge_2'] for s in stats_by_bins]
    frac_ge5 = [s['fraction_k_ge_5'] for s in stats_by_bins]
    frac_ge10 = [s['fraction_k_ge_10'] for s in stats_by_bins]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1) Min / median / mean equivalence class size
    axes[0, 0].plot(bins, k_median, marker='s', label='k_median')
    #axes[0, 0].plot(bins, mean_size, marker='^', label='mean_size')
    axes[0, 0].set_xlabel('Number of bins')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_ylabel('Equivalence class size')
    axes[0, 0].set_title('Equivalence class sizes vs discretisation')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2) Fraction with k ≥ 2, 5, 10
    axes[0, 1].plot(bins, frac_eq1, marker='x', label='k = 1 (singletons)')
    axes[0, 1].plot(bins, frac_ge2, marker='o', label='k ≥ 2')
    axes[0, 1].plot(bins, frac_ge5, marker='s', label='k ≥ 5')
    axes[0, 1].plot(bins, frac_ge10, marker='^', label='k ≥ 10')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_xlabel('Number of bins')
    axes[0, 1].set_ylabel('Fraction of records')
    axes[0, 1].set_title('Fraction of records in k‑anonymous equivalence classes')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3) (Optional) direct k_min vs n_bins (emphasised)
    axes[1, 0].plot(bins, k_min, marker='o', color='C3')
    axes[1, 0].set_xlabel('Number of bins')
    axes[1, 0].set_ylabel('k_min')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title('Minimum k vs discretisation level')
    axes[1, 0].grid(True, alpha=0.3)

    # 4) Leave last subplot empty or add any extra stat you want
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('k_anonymity_vs_discretisation.png', dpi=300, bbox_inches='tight')
    plt.show()

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

    # Vary discretisation levels
    discretisation_levels = config['discretisation_levels']

    stats_by_bins = []
    for n_bins in discretisation_levels:
        Xd = discretise(X, n_bins=n_bins, strategy=config['discretisation_strategy'])
        stats, sizes = eq_class_stats(Xd)
        stats['n_bins'] = n_bins
        stats_by_bins.append(stats)
        print(f"{n_bins} bins: {stats}")

    plot_k_anonymity_vs_discretisation(stats_by_bins)