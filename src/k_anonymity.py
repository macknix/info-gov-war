from collections import Counter
import numpy as np
from .generate import generate_gaussian_mixture, discretise,  generate_regression_data
from .mondrian_anonymiser import MondrianAnonymiser, mondrian_anonymise
import matplotlib.pyplot as plt

def eq_class_stats(X_disc):
    # treat each row as tuple key
    keys = [tuple(row) for row in X_disc]
    counts = Counter(keys)
    sizes = np.array([counts[k] for k in keys])
    
    # Get unique equivalence class sizes
    eq_class_sizes = np.array(list(counts.values()))
    n_records = len(X_disc)
    n_eq_classes = len(counts)
    k_min = int(eq_class_sizes.min())
    
    # Discernibility metric: sum of squares of each equivalence class size
    discernibility = int(np.sum(eq_class_sizes ** 2))
    
    # Normalised equivalence class metric: (total records / n equivalence classes) / k
    avg_eq_class_size = n_records / n_eq_classes
    normalised_eq_class_metric = avg_eq_class_size / k_min if k_min > 0 else float('inf')
    
    return {
        'k_min': k_min,
        'k_median': float(np.median(sizes)),
        'mean_size': float(np.mean(sizes)),
        'fraction_k_eq_1': float((sizes == 1).mean()),   # NEW: singletons
        'fraction_k_ge_2': float((sizes >= 2).mean()),
        'fraction_k_ge_5': float((sizes >= 5).mean()),
        'fraction_k_ge_10': float((sizes >= 10).mean()),
        'discernibility': discernibility,
        'normalised_eq_class_metric': float(normalised_eq_class_metric),
        'n_eq_classes': n_eq_classes
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
    discernibility = [s['discernibility'] for s in stats_by_bins]
    normalised_eq = [s['normalised_eq_class_metric'] for s in stats_by_bins]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

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

    # 4) Discernibility metric: sum of squares of equivalence class sizes
    axes[0, 2].plot(bins, discernibility, marker='D', color='C4')
    axes[0, 2].set_xlabel('Number of bins')
    axes[0, 2].set_ylabel('Discernibility metric')
    axes[0, 2].set_yscale('log')
    axes[0, 2].set_title('Discernibility metric (Σ|EC|²) vs discretisation')
    axes[0, 2].grid(True, alpha=0.3)

    # 5) Normalised equivalence class metric: (n_records / n_eq_classes) / k
    axes[1, 1].plot(bins, normalised_eq, marker='p', color='C5')
    axes[1, 1].set_xlabel('Number of bins')
    axes[1, 1].set_ylabel('Normalised EC metric')
    axes[1, 1].set_title('Normalised EC metric (avg_size / k) vs discretisation')
    axes[1, 1].grid(True, alpha=0.3)

    # 6) Leave last subplot empty or add any extra stat you want
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('k_anonymity_vs_discretisation.png', dpi=300, bbox_inches='tight')
    plt.show()


def compare_anonymisation_methods(X, k_values=[2, 5, 10, 20, 50], 
                                   discretisation_levels=[2, 3, 4, 5, 7, 10, 15, 20],
                                   disc_strategy='uniform'):
    """
    Compare discretisation-based anonymisation vs Mondrian k-anonymity.
    
    Parameters
    ----------
    X : np.ndarray
        Input data
    k_values : list of int
        k values to test for Mondrian
    discretisation_levels : list of int
        Number of bins to test for discretisation
    disc_strategy : str
        Discretisation strategy ('uniform', 'quantile', 'kmeans')
    
    Returns
    -------
    results : dict
        Dictionary containing stats for both methods
    """
    results = {
        'discretisation': [],
        'mondrian': []
    }
    
    # Run discretisation experiments
    print("Running discretisation experiments...")
    for n_bins in discretisation_levels:
        Xd = discretise(X, n_bins=n_bins, strategy=disc_strategy)
        stats, _ = eq_class_stats(Xd)
        stats['n_bins'] = n_bins
        stats['method'] = 'discretisation'
        results['discretisation'].append(stats)
        print(f"  {n_bins} bins: k={stats['k_min']}, discernibility={stats['discernibility']:.2e}")
    
    # Run Mondrian experiments
    print("\nRunning Mondrian experiments...")
    for k in k_values:
        try:
            anonymiser = MondrianAnonymiser(k=k, generalisation='midpoint')
            X_anon = anonymiser.fit_transform(X)
            stats = anonymiser.get_stats()
            
            # Also compute eq_class_stats on the anonymised data for consistency
            eq_stats, _ = eq_class_stats(X_anon)
            
            stats['k_target'] = k
            stats['method'] = 'mondrian'
            # Add fraction metrics from eq_class_stats
            stats['fraction_k_eq_1'] = eq_stats['fraction_k_eq_1']
            stats['fraction_k_ge_2'] = eq_stats['fraction_k_ge_2']
            stats['fraction_k_ge_5'] = eq_stats['fraction_k_ge_5']
            stats['fraction_k_ge_10'] = eq_stats['fraction_k_ge_10']
            
            results['mondrian'].append(stats)
            print(f"  k={k}: achieved k={stats['k_achieved']}, NCP={stats['ncp']:.2f}%, "
                  f"discernibility={stats['discernibility']:.2e}")
        except Exception as e:
            print(f"  k={k}: Failed - {e}")
    
    return results


def plot_anonymisation_comparison(results):
    """
    Plot comparison of discretisation vs Mondrian anonymisation methods.
    
    Parameters
    ----------
    results : dict
        Output from compare_anonymisation_methods()
    """
    disc_stats = results['discretisation']
    mond_stats = results['mondrian']
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Prepare data
    disc_k = [s['k_min'] for s in disc_stats]
    disc_discern = [s['discernibility'] for s in disc_stats]
    disc_norm_eq = [s['normalised_eq_class_metric'] for s in disc_stats]
    disc_n_eq = [s['n_eq_classes'] for s in disc_stats]
    disc_bins = [s['n_bins'] for s in disc_stats]
    
    mond_k_target = [s['k_target'] for s in mond_stats]
    mond_k_achieved = [s['k_achieved'] for s in mond_stats]
    mond_discern = [s['discernibility'] for s in mond_stats]
    mond_norm_eq = [s['normalised_eq_class_metric'] for s in mond_stats]
    mond_n_eq = [s['n_eq_classes'] for s in mond_stats]
    mond_ncp = [s['ncp'] for s in mond_stats]
    
    # 1) k achieved vs parameter (bins for disc, k_target for mondrian)
    ax = axes[0, 0]
    ax.plot(disc_bins, disc_k, marker='o', label='Discretisation (k achieved)')
    ax.plot(mond_k_target, mond_k_achieved, marker='s', label='Mondrian (k achieved)')
    ax.plot(mond_k_target, mond_k_target, 'k--', alpha=0.5, label='k target')
    ax.set_xlabel('Bins / k target')
    ax.set_ylabel('k achieved')
    ax.set_yscale('log')
    ax.set_title('k-Anonymity Achieved')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2) Discernibility vs k achieved (both methods on same plot)
    ax = axes[0, 1]
    ax.scatter(disc_k, disc_discern, marker='o', s=100, label='Discretisation', alpha=0.7)
    ax.scatter(mond_k_achieved, mond_discern, marker='s', s=100, label='Mondrian', alpha=0.7)
    ax.set_xlabel('k achieved')
    ax.set_ylabel('Discernibility (Σ|EC|²)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Discernibility vs k-Anonymity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3) Normalised EC metric vs k achieved
    ax = axes[0, 2]
    ax.scatter(disc_k, disc_norm_eq, marker='o', s=100, label='Discretisation', alpha=0.7)
    ax.scatter(mond_k_achieved, mond_norm_eq, marker='s', s=100, label='Mondrian', alpha=0.7)
    ax.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Optimal (=1)')
    ax.set_xlabel('k achieved')
    ax.set_ylabel('Normalised EC metric (avg_size / k)')
    ax.set_xscale('log')
    ax.set_title('Normalised EC Metric vs k-Anonymity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4) Number of equivalence classes vs k
    ax = axes[1, 0]
    ax.scatter(disc_k, disc_n_eq, marker='o', s=100, label='Discretisation', alpha=0.7)
    ax.scatter(mond_k_achieved, mond_n_eq, marker='s', s=100, label='Mondrian', alpha=0.7)
    ax.set_xlabel('k achieved')
    ax.set_ylabel('Number of equivalence classes')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Number of ECs vs k-Anonymity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5) Mondrian NCP vs k
    ax = axes[1, 1]
    ax.plot(mond_k_target, mond_ncp, marker='s', color='C1')
    ax.set_xlabel('k target')
    ax.set_ylabel('NCP (%)')
    ax.set_title('Mondrian: Information Loss (NCP) vs k')
    ax.grid(True, alpha=0.3)
    
    # 6) Summary comparison at similar k levels
    ax = axes[1, 2]
    # Find comparable k values
    text_lines = ["Method Comparison Summary\n"]
    text_lines.append("-" * 30 + "\n")
    text_lines.append(f"Discretisation:\n")
    text_lines.append(f"  k range: {min(disc_k)} - {max(disc_k)}\n")
    text_lines.append(f"  # ECs: {min(disc_n_eq)} - {max(disc_n_eq)}\n")
    text_lines.append(f"\nMondrian:\n")
    text_lines.append(f"  k range: {min(mond_k_achieved)} - {max(mond_k_achieved)}\n")
    text_lines.append(f"  # ECs: {min(mond_n_eq)} - {max(mond_n_eq)}\n")
    text_lines.append(f"  NCP range: {min(mond_ncp):.1f}% - {max(mond_ncp):.1f}%\n")
    text_lines.append(f"\nMondrian guarantees k-anonymity")
    text_lines.append(f"\nwhile discretisation does not.")
    ax.text(0.1, 0.9, ''.join(text_lines), transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('anonymisation_comparison.png', dpi=300, bbox_inches='tight')
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