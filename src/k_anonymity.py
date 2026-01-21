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

def plot_k_anonymity_vs_discretisation(stats_by_bins, save_path='k_anonymity_vs_discretisation.png'):
    """
    Given a list of dicts (one per discretisation level) with k-anonymity stats,
    produce summary plots.
    
    Parameters
    ----------
    stats_by_bins : list of dict
        Each dict contains k-anonymity stats for a discretisation level
    save_path : str
        Path to save the figure
    """
    # Prepare data for plotting
    bins = [s['n_bins'] for s in stats_by_bins]
    k_min = [s['k_min'] for s in stats_by_bins]
    k_median = [s['k_median'] for s in stats_by_bins]
    frac_eq1 = [s['fraction_k_eq_1'] for s in stats_by_bins] 
    frac_ge5 = [s['fraction_k_ge_5'] for s in stats_by_bins]
    discernibility = [s['discernibility'] for s in stats_by_bins]
    n_eq_classes = [s['n_eq_classes'] for s in stats_by_bins]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Color scheme
    main_color = '#2ecc71'
    secondary_color = '#e74c3c'

    # 1) k_min and k_median vs bins
    ax = axes[0, 0]
    ax.plot(bins, k_min, marker='o', markersize=8, linewidth=2, 
            color=secondary_color, label='k_min (privacy level)')
    ax.plot(bins, k_median, marker='s', markersize=8, linewidth=2, 
            color=main_color, label='k_median')
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='k=1 (unique)')
    ax.set_xlabel('Number of bins', fontsize=11)
    ax.set_ylabel('Equivalence class size', fontsize=11)
    ax.set_yscale('log')
    ax.set_title('Equivalence Class Sizes', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2) Fraction singletons vs k >= 5
    ax = axes[0, 1]
    ax.plot(bins, frac_eq1, marker='x', markersize=10, linewidth=2, 
            color=secondary_color, label='k=1 (singletons - risky)')
    ax.plot(bins, frac_ge5, marker='s', markersize=8, linewidth=2, 
            color=main_color, label='k≥5 (well protected)')
    ax.fill_between(bins, 0, frac_eq1, alpha=0.2, color=secondary_color)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Number of bins', fontsize=11)
    ax.set_ylabel('Fraction of records', fontsize=11)
    ax.set_title('Record Protection Levels', fontsize=12, fontweight='bold')
    ax.legend(loc='center right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # 3) Number of equivalence classes
    ax = axes[1, 0]
    ax.plot(bins, n_eq_classes, marker='D', markersize=8, linewidth=2, color='#9b59b6')
    ax.set_xlabel('Number of bins', fontsize=11)
    ax.set_ylabel('Number of equivalence classes', fontsize=11)
    ax.set_yscale('log')
    ax.set_title('Data Granularity (# of Unique Groups)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 4) Discernibility metric
    ax = axes[1, 1]
    ax.plot(bins, discernibility, marker='D', markersize=8, linewidth=2, color='#3498db')
    ax.set_xlabel('Number of bins', fontsize=11)
    ax.set_ylabel('Discernibility (Σ|EC|²)', fontsize=11)
    ax.set_yscale('log')
    ax.set_title('Discernibility Metric (Lower = Less Utility)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
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


def plot_anonymisation_comparison(results, save_path='anonymisation_comparison.png'):
    """
    Plot comparison of discretisation vs Mondrian anonymisation methods.
    
    Creates a simple 2x2 grid comparing key metrics between methods.
    
    Parameters
    ----------
    results : dict
        Output from compare_anonymisation_methods()
    save_path : str
        Path to save the figure
    """
    disc_stats = results['discretisation']
    mond_stats = results['mondrian']
    
    # Check we have data
    if not disc_stats or not mond_stats:
        print("Warning: Missing data for one or both methods")
        return
    
    # Prepare data
    disc_k = [s['k_min'] for s in disc_stats]
    disc_discern = [s['discernibility'] for s in disc_stats]
    disc_n_eq = [s['n_eq_classes'] for s in disc_stats]
    disc_bins = [s['n_bins'] for s in disc_stats]
    
    mond_k_target = [s['k_target'] for s in mond_stats]
    mond_k_achieved = [s['k_achieved'] for s in mond_stats]
    mond_discern = [s['discernibility'] for s in mond_stats]
    mond_n_eq = [s['n_eq_classes'] for s in mond_stats]
    mond_ncp = [s['ncp'] for s in mond_stats]
    
    # Colors
    disc_color = '#2ecc71'  # green
    mond_color = '#3498db'  # blue
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1) Privacy-Utility Trade-off: Discernibility vs k achieved
    ax = axes[0, 0]
    ax.scatter(disc_k, disc_discern, marker='o', s=120, c=disc_color, 
               label='Discretisation', alpha=0.8, edgecolors='black', linewidth=0.5)
    ax.scatter(mond_k_achieved, mond_discern, marker='s', s=120, c=mond_color, 
               label='Mondrian', alpha=0.8, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('k achieved (privacy level)', fontsize=11)
    ax.set_ylabel('Discernibility (Σ|EC|²)', fontsize=11)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Privacy vs Utility Trade-off', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2) k achieved comparison (side by side bars)
    ax = axes[0, 1]
    x_pos = np.arange(len(mond_k_target))
    bar_width = 0.35
    
    # For discretisation, show best k achieved at different bin levels
    ax.bar(x_pos - bar_width/2, mond_k_target, bar_width, 
           label='Mondrian target k', color=mond_color, alpha=0.6)
    ax.bar(x_pos + bar_width/2, mond_k_achieved, bar_width, 
           label='Mondrian achieved k', color=mond_color, alpha=1.0, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Configuration', fontsize=11)
    ax.set_ylabel('k value', fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'k={k}' for k in mond_k_target], fontsize=9)
    ax.set_title('Mondrian: Target vs Achieved k', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3) Number of equivalence classes vs k
    ax = axes[1, 0]
    ax.scatter(disc_k, disc_n_eq, marker='o', s=120, c=disc_color, 
               label='Discretisation', alpha=0.8, edgecolors='black', linewidth=0.5)
    ax.scatter(mond_k_achieved, mond_n_eq, marker='s', s=120, c=mond_color, 
               label='Mondrian', alpha=0.8, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('k achieved', fontsize=11)
    ax.set_ylabel('Number of equivalence classes', fontsize=11)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Granularity: ECs vs Privacy Level', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 4) Discretisation: k achieved vs bins
    ax = axes[1, 1]
    ax.plot(disc_bins, disc_k, marker='o', markersize=10, linewidth=2, 
            color=disc_color, label='k achieved')
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='k=1 (no privacy)')
    ax.fill_between(disc_bins, 1, disc_k, alpha=0.2, color=disc_color)
    ax.set_xlabel('Number of bins', fontsize=11)
    ax.set_ylabel('k achieved', fontsize=11)
    ax.set_yscale('log')
    ax.set_title('Discretisation: k vs Number of Bins', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("ANONYMISATION METHOD COMPARISON SUMMARY")
    print("="*50)
    print(f"\nDiscretisation:")
    print(f"  Bins tested: {disc_bins}")
    print(f"  k range achieved: {min(disc_k)} - {max(disc_k)}")
    print(f"  ⚠️  Does NOT guarantee k-anonymity")
    print(f"\nMondrian:")
    print(f"  k targets: {mond_k_target}")
    print(f"  k achieved: {mond_k_achieved}")
    print(f"  NCP (info loss): {min(mond_ncp):.1f}% - {max(mond_ncp):.1f}%")
    print(f"  ✓ Guarantees k-anonymity")
    print("="*50)

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

    # Get config values
    discretisation_levels = config['discretisation_levels']
    mondrian_k_values = config.get('mondrian_k_values', [2, 5, 10, 20, 50])
    disc_strategy = config['discretisation_strategy']

    print("=" * 60)
    print("ANONYMISATION METHOD COMPARISON")
    print("=" * 60)
    
    # Run comparison
    results = compare_anonymisation_methods(
        X, 
        k_values=mondrian_k_values,
        discretisation_levels=discretisation_levels,
        disc_strategy=disc_strategy
    )
    
    # Plot discretisation-only results
    print("\n--- Discretisation Analysis ---")
    plot_k_anonymity_vs_discretisation(results['discretisation'])
    
    # Plot comparison between methods
    print("\n--- Method Comparison ---")
    plot_anonymisation_comparison(results)