"""
Mondrian k-Anonymity Pipeline

This pipeline evaluates the privacy-utility trade-off when using 
Mondrian k-anonymity as an anonymisation method.

Steps:
1. Generate synthetic regression data
2. Plot raw data statistics  
3. For each target k value:
   - Apply Mondrian anonymisation
   - Compute k-anonymity metrics (discernibility, normalised EC, etc.)
4. Plot k-anonymity metrics vs k
5. Run modelling experiments on Mondrian-anonymised data
6. Plot utility and equity results
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt

from .generate import (
    generate_regression_data,
    plot_regression_stats,
)
from .k_anonymity import eq_class_stats
from .mondrian_anonymiser import MondrianAnonymiser


def run_mondrian_anonymisation(X, k_values):
    """
    Apply Mondrian anonymisation for each k value.
    
    Parameters
    ----------
    X : np.ndarray
        Input data
    k_values : list of int
        k-anonymity values to test
    
    Returns
    -------
    results : list of dict
        Each dict contains:
        - 'k_target': target k value
        - 'k_achieved': actual minimum equivalence class size
        - 'X_anon': anonymised data
        - 'sizes': equivalence class sizes for each record
        - 'discernibility': sum of squared EC sizes
        - 'normalised_eq_class_metric': avg_size / k
        - 'n_eq_classes': number of equivalence classes
        - 'ncp': normalised certainty penalty (%)
        - 'runtime': time taken (seconds)
    """
    results = []
    
    for k in k_values:
        print(f"  Anonymising with k={k}...")
        try:
            anonymiser = MondrianAnonymiser(k=k, generalisation='midpoint')
            X_anon = anonymiser.fit_transform(X)
            stats = anonymiser.get_stats()
            
            # Get per-record equivalence class sizes
            eq_stats, sizes = eq_class_stats(X_anon)
            
            result = {
                'k_target': k,
                'k_achieved': stats['k_achieved'],
                'X_anon': X_anon,
                'sizes': sizes,
                'discernibility': stats['discernibility'],
                'normalised_eq_class_metric': stats['normalised_eq_class_metric'],
                'n_eq_classes': stats['n_eq_classes'],
                'ncp': stats['ncp'],
                'runtime': stats['runtime'],
                'mean_class_size': stats['mean_class_size'],
                'median_class_size': stats['median_class_size'],
                # Also include fraction metrics
                'fraction_k_eq_1': eq_stats['fraction_k_eq_1'],
                'fraction_k_ge_2': eq_stats['fraction_k_ge_2'],
                'fraction_k_ge_5': eq_stats['fraction_k_ge_5'],
                'fraction_k_ge_10': eq_stats['fraction_k_ge_10'],
            }
            results.append(result)
            
            print(f"    k_achieved={stats['k_achieved']}, "
                  f"NCP={stats['ncp']:.2f}%, "
                  f"discernibility={stats['discernibility']:.2e}, "
                  f"#ECs={stats['n_eq_classes']}")
                  
        except Exception as e:
            print(f"    Failed: {e}")
    
    return results


def plot_mondrian_k_anonymity_stats(mondrian_results):
    """
    Plot k-anonymity statistics for Mondrian anonymisation.
    
    Similar to plot_k_anonymity_vs_discretisation but for Mondrian.
    """
    k_targets = [r['k_target'] for r in mondrian_results]
    k_achieved = [r['k_achieved'] for r in mondrian_results]
    discernibility = [r['discernibility'] for r in mondrian_results]
    normalised_eq = [r['normalised_eq_class_metric'] for r in mondrian_results]
    n_eq_classes = [r['n_eq_classes'] for r in mondrian_results]
    ncp = [r['ncp'] for r in mondrian_results]
    median_size = [r['median_class_size'] for r in mondrian_results]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Mondrian k-Anonymity Statistics', fontsize=14, fontweight='bold')
    
    # 1) k achieved vs k target
    ax = axes[0, 0]
    ax.plot(k_targets, k_achieved, marker='o', color='C0', label='k achieved')
    ax.plot(k_targets, k_targets, 'k--', alpha=0.5, label='k = target')
    ax.set_xlabel('k target')
    ax.set_ylabel('k achieved')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('k Achieved vs k Target')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2) Median equivalence class size
    ax = axes[0, 1]
    ax.plot(k_targets, median_size, marker='s', color='C1')
    ax.set_xlabel('k target')
    ax.set_ylabel('Median EC size')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Median Equivalence Class Size vs k')
    ax.grid(True, alpha=0.3)
    
    # 3) Discernibility metric
    ax = axes[0, 2]
    ax.plot(k_targets, discernibility, marker='D', color='C4')
    ax.set_xlabel('k target')
    ax.set_ylabel('Discernibility (Σ|EC|²)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Discernibility Metric vs k')
    ax.grid(True, alpha=0.3)
    
    # 4) Normalised EC metric
    ax = axes[1, 0]
    ax.plot(k_targets, normalised_eq, marker='p', color='C5')
    ax.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Optimal (=1)')
    ax.set_xlabel('k target')
    ax.set_ylabel('Normalised EC metric (avg_size / k)')
    ax.set_xscale('log')
    ax.set_title('Normalised EC Metric vs k')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5) Number of equivalence classes
    ax = axes[1, 1]
    ax.plot(k_targets, n_eq_classes, marker='o', color='C2')
    ax.set_xlabel('k target')
    ax.set_ylabel('Number of equivalence classes')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Number of Equivalence Classes vs k')
    ax.grid(True, alpha=0.3)
    
    # 6) NCP (Information Loss)
    ax = axes[1, 2]
    ax.plot(k_targets, ncp, marker='^', color='C3')
    ax.set_xlabel('k target')
    ax.set_ylabel('NCP (%)')
    ax.set_xscale('log')
    ax.set_title('Information Loss (NCP) vs k')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mondrian_k_anonymity_stats.png', dpi=300, bbox_inches='tight')
    plt.show()


def run_mondrian_regression_experiment(mondrian_results, y, random_state=None):
    """
    Run regression experiments on Mondrian-anonymised datasets.
    
    Parameters
    ----------
    mondrian_results : list of dict
        Output from run_mondrian_anonymisation()
    y : np.ndarray
        Target variable
    random_state : int
        Random seed
    
    Returns
    -------
    results : dict
        results[k_target][model_name] = metrics dict
    """
    from .modelling import train_and_eval_regression
    
    results = {}
    
    for mr in mondrian_results:
        k = mr['k_target']
        X_anon = mr['X_anon']
        sizes = mr['sizes']
        
        print(f"\n  Running models for k={k}...")
        results[k] = {}
        
        for model_name in ['lr', 'rf', 'mlp', 'tabpfn']:
            res = train_and_eval_regression(
                X_anon, y,
                model_name=model_name,
                random_state=random_state,
                sizes_full=sizes,
            )
            results[k][model_name] = res
            print(f"    {model_name}: RMSE={res['rmse']:.4f}, R²={res['r2']:.4f}")
    
    return results


def plot_mondrian_regression_results(results_by_k):
    """
    Plot utility and equity results for Mondrian pipeline.
    
    Parameters
    ----------
    results_by_k : dict
        results_by_k[k][model_name] = metrics dict
    """
    k_levels = sorted(results_by_k.keys())
    model_names = ['lr', 'rf', 'mlp', 'tabpfn']
    model_labels = {'lr': 'Linear', 'rf': 'RandomForest', 'mlp': 'MLP', 'tabpfn': 'TabPFN'}
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Mondrian: Model Utility & Equity vs k-Anonymity', fontsize=14, fontweight='bold')
    
    # 1) RMSE vs k
    ax = axes[0, 0]
    for m in model_names:
        rmses = [results_by_k[k][m]['rmse'] for k in k_levels]
        ax.plot(k_levels, rmses, marker='o', label=model_labels.get(m, m))
    ax.set_xscale('log')
    ax.set_xlabel('k (anonymity level)')
    ax.set_ylabel('RMSE')
    ax.set_title('Utility: RMSE vs k')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2) R² vs k
    ax = axes[0, 1]
    for m in model_names:
        r2s = [results_by_k[k][m]['r2'] for k in k_levels]
        ax.plot(k_levels, r2s, marker='o', label=model_labels.get(m, m))
    ax.set_xscale('log')
    ax.set_xlabel('k (anonymity level)')
    ax.set_ylabel('R²')
    ax.set_title('Utility: R² vs k')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3) Equity: RMSE bottom vs top k-quartile
    ax = axes[1, 0]
    for m in model_names:
        rmse_bottom = [results_by_k[k][m]['rmse_bottom_k_q'] for k in k_levels]
        rmse_top = [results_by_k[k][m]['rmse_top_k_q'] for k in k_levels]
        ax.plot(k_levels, rmse_bottom, marker='o', linestyle='-',
                label=f'{model_labels.get(m, m)} bottom 25%')
        ax.plot(k_levels, rmse_top, marker='s', linestyle='--',
                label=f'{model_labels.get(m, m)} top 25%')
    ax.set_xscale('log')
    ax.set_xlabel('k (anonymity level)')
    ax.set_ylabel('RMSE')
    ax.set_title('Equity: RMSE for low vs high EC size groups')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # 4) Equity: R² bottom vs top k-quartile
    ax = axes[1, 1]
    for m in model_names:
        r2_bottom = [results_by_k[k][m]['r2_bottom_k_q'] for k in k_levels]
        r2_top = [results_by_k[k][m]['r2_top_k_q'] for k in k_levels]
        ax.plot(k_levels, r2_bottom, marker='o', linestyle='-',
                label=f'{model_labels.get(m, m)} bottom 25%')
        ax.plot(k_levels, r2_top, marker='s', linestyle='--',
                label=f'{model_labels.get(m, m)} top 25%')
    ax.set_xscale('log')
    ax.set_xlabel('k (anonymity level)')
    ax.set_ylabel('R²')
    ax.set_title('Equity: R² for low vs high EC size groups')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mondrian_regression_utility_equity.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_mondrian_equity_deciles(results_by_k):
    """
    Plot equity variance across k-groups and target buckets for Mondrian.
    """
    k_levels = sorted(results_by_k.keys())
    model_names = ['lr', 'rf', 'mlp', 'tabpfn']
    model_labels = {'lr': 'Linear', 'rf': 'RandomForest', 'mlp': 'MLP', 'tabpfn': 'TabPFN'}
    
    # Get n_tiles from first result
    first_k = k_levels[0]
    first_m = model_names[0]
    dec_example = results_by_k[first_k][first_m].get('decile_rmse')
    n_tiles = len(dec_example) if dec_example is not None else 4
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Mondrian: Performance Consistency Across Groups', fontsize=14, fontweight='bold')
    
    # Top row: k-group variance
    ax_rmse = axes[0, 0]
    ax_r2 = axes[0, 1]
    
    for m in model_names:
        rmse_var = [results_by_k[k][m].get('decile_rmse_var', np.nan) for k in k_levels]
        r2_var = [results_by_k[k][m].get('decile_r2_var', np.nan) for k in k_levels]
        ax_rmse.plot(k_levels, rmse_var, marker='o', label=model_labels.get(m, m))
        ax_r2.plot(k_levels, r2_var, marker='o', label=model_labels.get(m, m))
    
    ax_rmse.set_xscale('log')
    ax_rmse.set_xlabel('k (anonymity level)')
    ax_rmse.set_ylabel(f'Var(RMSE across {n_tiles} k-groups)')
    ax_rmse.set_title('Consistency across k-groups (RMSE variance)')
    ax_rmse.legend(fontsize=8)
    ax_rmse.grid(True, alpha=0.3)
    
    ax_r2.set_xscale('log')
    ax_r2.set_xlabel('k (anonymity level)')
    ax_r2.set_ylabel(f'Var(R² across {n_tiles} k-groups)')
    ax_r2.set_title('Consistency across k-groups (R² variance)')
    ax_r2.legend(fontsize=8)
    ax_r2.grid(True, alpha=0.3)
    
    # Bottom row: target bucket variance
    ax_trmse = axes[1, 0]
    ax_tr2 = axes[1, 1]
    
    for m in model_names:
        t_rmse_var = [results_by_k[k][m].get('target_bucket_rmse_var', np.nan) for k in k_levels]
        t_r2_var = [results_by_k[k][m].get('target_bucket_r2_var', np.nan) for k in k_levels]
        ax_trmse.plot(k_levels, t_rmse_var, marker='o', label=model_labels.get(m, m))
        ax_tr2.plot(k_levels, t_r2_var, marker='o', label=model_labels.get(m, m))
    
    ax_trmse.set_xscale('log')
    ax_trmse.set_xlabel('k (anonymity level)')
    ax_trmse.set_ylabel(f'Var(RMSE across {n_tiles} target buckets)')
    ax_trmse.set_title('Consistency across target buckets (RMSE variance)')
    ax_trmse.legend(fontsize=8)
    ax_trmse.grid(True, alpha=0.3)
    
    ax_tr2.set_xscale('log')
    ax_tr2.set_xlabel('k (anonymity level)')
    ax_tr2.set_ylabel(f'Var(R² across {n_tiles} target buckets)')
    ax_tr2.set_title('Consistency across target buckets (R² variance)')
    ax_tr2.legend(fontsize=8)
    ax_tr2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mondrian_equity_variance.png', dpi=300, bbox_inches='tight')
    plt.show()


def main(config_path: str = "config.yaml") -> None:
    """Run the Mondrian k-anonymity pipeline."""
    
    # 1) Load config
    print("=" * 60)
    print("MONDRIAN k-ANONYMITY PIPELINE")
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
    k_values = config.get("mondrian_k_values", [2, 5, 10, 20, 50, 100])

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

    # 4) Apply Mondrian anonymisation for each k
    print(f"\n[3/4] Applying Mondrian anonymisation for k values: {k_values}")
    mondrian_results = run_mondrian_anonymisation(X, k_values)
    
    print("\nPlotting Mondrian k-anonymity statistics...")
    plot_mondrian_k_anonymity_stats(mondrian_results)

    # 5) Run modelling experiments
    print("\n[4/4] Running regression experiments on Mondrian-anonymised data...")
    model_results = run_mondrian_regression_experiment(
        mondrian_results, y, random_state=random_state
    )
    
    print("\nPlotting utility/equity results...")
    plot_mondrian_regression_results(model_results)
    plot_mondrian_equity_deciles(model_results)

    print("\n" + "=" * 60)
    print("MONDRIAN PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
