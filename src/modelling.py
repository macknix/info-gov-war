
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from .generate import generate_gaussian_mixture, discretise, generate_regression_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import mean_squared_error, r2_score
from .k_anonymity import eq_class_stats
from tabpfn import TabPFNRegressor

n_tiles = 4

# ---------- Utility & equity ----------
def train_and_eval_classification(X, y, model_name='logreg', random_state=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y)
    if model_name == 'logreg':
        clf = LogisticRegression(max_iter=2000, solver='saga', multi_class='multinomial')
    elif model_name == 'rf':
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=random_state)
    elif model_name == 'mlp':
        clf = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=300, random_state=random_state)
    else:
        raise ValueError(model_name)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    # per-class metrics
    per_class_acc = {}
    classes = np.unique(y)
    for c in classes:
        idx = (y_test == c)
        per_class_acc[int(c)] = float((y_pred[idx] == y_test[idx]).mean()) if idx.sum() > 0 else np.nan
    return {'acc': acc, 'bacc': bacc, 'macro_f1': macro_f1, 'per_class_acc': per_class_acc, 'model': clf}

def train_and_eval_regression(X, y, model_name='lr', random_state=None, sizes_full=None):
    """
    X is assumed discretised if sizes_full is provided.

    sizes_full : array of length N (same as len(X)), giving equivalence
                 class sizes computed on the full population Xd.
                 We will restrict this to the test indices for equity metrics.
    """
    # We need indices to map sizes_full -> test split
    N = X.shape[0]
    all_idx = np.arange(N)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, all_idx, test_size=0.3, random_state=random_state
    )

    if model_name == 'lr':
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    elif model_name == 'rf':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=random_state)
    elif model_name == 'mlp':
        from sklearn.neural_network import MLPRegressor
        model = MLPRegressor(hidden_layer_sizes=(128,64), max_iter=300, random_state=random_state)
    elif model_name == 'tabpfn':
        model = TabPFNRegressor(device='auto', random_state=random_state)
    else:
        raise ValueError(model_name)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred)**0.5
    r2 = r2_score(y_test, y_pred)

    # defaults
    rmse_bottom = rmse_top = np.nan
    r2_bottom = r2_top = np.nan
    decile_rmse_var = np.nan
    decile_r2_var = np.nan

    if sizes_full is not None:
        # sizes_full is population-level k, restrict to test rows
        sizes_test = sizes_full[idx_test]

        q25 = np.percentile(sizes_test, 25)
        q75 = np.percentile(sizes_test, 75)

        # bottom/top groups in TEST SET, but k is relative to population
        mask_bottom = sizes_test <= q25
        mask_top    = sizes_test >= q75

        def subgroup_rmse_r2(y_true_sub, y_pred_sub):
            if len(y_true_sub) == 0:
                return np.nan, np.nan
            rmse_s = float(np.sqrt(((y_true_sub - y_pred_sub) ** 2).mean()))
            if np.allclose(y_true_sub, y_true_sub.mean()):
                r2_s = np.nan
            else:
                ss_res = ((y_true_sub - y_pred_sub) ** 2).sum()
                ss_tot = ((y_true_sub - y_true_sub.mean()) ** 2).sum()
                r2_s = float(1.0 - ss_res / ss_tot)
            return rmse_s, r2_s

        rmse_bottom, r2_bottom = subgroup_rmse_r2(y_test[mask_bottom], y_pred[mask_bottom])
        rmse_top,    r2_top    = subgroup_rmse_r2(y_test[mask_top],    y_pred[mask_top])

       # ---------- NEW: n equity groups (n-tiles, default 10) ----------

        # Define group boundaries on sizes_test with quantiles
        quantiles = np.linspace(0, 100, n_tiles + 1)
        tile_edges = np.percentile(sizes_test, quantiles)

        tile_rmse = []
        tile_r2 = []

        for i in range(n_tiles):
            # Left-inclusive, right-exclusive, except last bucket inclusive on both ends
            if i < n_tiles - 1:
                mask_tile = (sizes_test >= tile_edges[i]) & (sizes_test < tile_edges[i+1])
            else:
                mask_tile = (sizes_test >= tile_edges[i]) & (sizes_test <= tile_edges[i+1])

            rmse_g, r2_g = subgroup_rmse_r2(y_test[mask_tile], y_pred[mask_tile])
            tile_rmse.append(rmse_g)
            tile_r2.append(r2_g)

        tile_rmse = np.array(tile_rmse, dtype=float)
        tile_r2 = np.array(tile_r2, dtype=float)

        # Aggregate consistency metrics across groups (ignore NaNs)
        if np.isfinite(tile_rmse).any():
            decile_rmse_var = float(np.nanvar(tile_rmse))
        if np.isfinite(tile_r2).any():
            decile_r2_var = float(np.nanvar(tile_r2))

        # Keep old names in result dict for backwards compatibility
        decile_rmse = tile_rmse
        decile_r2 = tile_r2

        # ---------- NEW: performance buckets over the TARGET (y_test) ----------
        # We use the same n_tiles to partition the target distribution.
        y_quantiles = np.linspace(0, 100, n_tiles + 1)
        y_edges = np.percentile(y_test, y_quantiles)

        ytile_rmse = []
        ytile_r2 = []

        for i in range(n_tiles):
            if i < n_tiles - 1:
                mask_y = (y_test >= y_edges[i]) & (y_test < y_edges[i+1])
            else:
                mask_y = (y_test >= y_edges[i]) & (y_test <= y_edges[i+1])

            rmse_g, r2_g = subgroup_rmse_r2(y_test[mask_y], y_pred[mask_y])
            ytile_rmse.append(rmse_g)
            ytile_r2.append(r2_g)

        ytile_rmse = np.array(ytile_rmse, dtype=float)
        ytile_r2 = np.array(ytile_r2, dtype=float)

        # Aggregate consistency metrics across target buckets (ignore NaNs)
        if np.isfinite(ytile_rmse).any():
            target_bucket_rmse_var = float(np.nanvar(ytile_rmse))
        if np.isfinite(ytile_r2).any():
            target_bucket_r2_var = float(np.nanvar(ytile_r2))

    return {
        'rmse': rmse,
        'r2': r2,
        'rmse_bottom_k_q': rmse_bottom,
        'r2_bottom_k_q': r2_bottom,
        'rmse_top_k_q': rmse_top,
        'r2_top_k_q': r2_top,
        'decile_rmse': decile_rmse,          # length-10 array or None
        'decile_r2': decile_r2,              # length-10 array or None
        'decile_rmse_var': decile_rmse_var,  # scalar variance of decile RMSEs
        'decile_r2_var': decile_r2_var,
        'target_bucket_rmse_var': target_bucket_rmse_var,
        'target_bucket_r2_var': target_bucket_r2_var,
        'model': model,
    }

def run_regression_experiment(X, y, dbins=[2,4,8,10], random_state=None):
    results = {}
    for n_bins in dbins:
        # Discretise full population
        Xd = discretise(X, n_bins=n_bins, strategy='uniform')
        # k-anonymity sizes relative to entire population
        _, sizes_full = eq_class_stats(Xd)

        results[n_bins] = {}  # one dict per model at this n_bins
        for model_name in ['lr','rf','mlp','tabpfn']:
            res = train_and_eval_regression(
                Xd, y, model_name=model_name,
                random_state=random_state,
                sizes_full=sizes_full,  # NEW
            )
            results[n_bins][model_name] = res
            print(f"n_bins={n_bins}, model={model_name}, results={res}")
    return results



def plot_regression_discretisation_results(results_by_bins):
    """
    results_by_bins[n_bins][model_name] = {
        'rmse', 'r2',
        'rmse_bottom_k_q', 'r2_bottom_k_q',
        'rmse_top_k_q',    'r2_top_k_q',
        'model': ...
    }
    """
    bin_levels = sorted(results_by_bins.keys())
    model_names = ['lr', 'rf', 'mlp', 'tabpfn']
    model_labels = {'lr': 'Linear', 'rf': 'RandomForest', 'mlp': 'MLP', 'tabpfn': 'TabPFN'}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1) Utility – RMSE vs n_bins
    for m in model_names:
        rmses = [results_by_bins[b][m]["rmse"] for b in bin_levels]
        axes[0, 0].plot(bin_levels, rmses, marker='o', label=model_labels.get(m, m))
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_xlabel('Number of bins (n_bins)')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].set_title('Utility: overall RMSE vs discretisation level')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # 2) Utility – R² vs n_bins
    for m in model_names:
        r2s = [results_by_bins[b][m]["r2"] for b in bin_levels]
        axes[0, 1].plot(bin_levels, r2s, marker='o', label=model_labels.get(m, m))
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_xlabel('Number of bins (n_bins)')
    axes[0, 1].set_ylabel('R²')
    axes[0, 1].set_title('Utility: overall R² vs discretisation level')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # 3) Equity – RMSE bottom vs top k-quartile per model
    for m in model_names:
        rmse_bottom = [results_by_bins[b][m]["rmse_bottom_k_q"] for b in bin_levels]
        rmse_top    = [results_by_bins[b][m]["rmse_top_k_q"]    for b in bin_levels]
        axes[1, 0].plot(bin_levels, rmse_bottom, marker='o',
                        linestyle='-',  label=f'{model_labels.get(m,m)} bottom 25%')
        axes[1, 0].plot(bin_levels, rmse_top,    marker='s',
                        linestyle='--', label=f'{model_labels.get(m,m)} top 25%')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_xlabel('Number of bins (n_bins)')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].set_title('Equity: RMSE for low vs high represented groups')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=8)

    # 4) Equity – R² bottom vs top k-quartile per model
    for m in model_names:
        r2_bottom = [results_by_bins[b][m]["r2_bottom_k_q"] for b in bin_levels]
        r2_top    = [results_by_bins[b][m]["r2_top_k_q"]    for b in bin_levels]
        axes[1, 1].plot(bin_levels, r2_bottom, marker='o',
                        linestyle='-',  label=f'{model_labels.get(m,m)} bottom 25%')
        axes[1, 1].plot(bin_levels, r2_top,    marker='s',
                        linestyle='--', label=f'{model_labels.get(m,m)} top 25%')
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_xlabel('Number of bins (n_bins)')
    axes[1, 1].set_ylabel('R²')
    axes[1, 1].set_title('Equity: R² for low vs high represented groups')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=8)


    plt.tight_layout()
    plt.savefig('regression_utility_equity_vs_discretisation.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_regression_equity_deciles(results_by_bins, example_bins=None):
    """
    Visualise how consistent performance is across:
      - n k-anonymity groups (n-tiles of k size)
      - n target buckets (n-tiles of the target distribution)

    results_by_bins[n_bins][model_name] must contain:
        'decile_rmse', 'decile_r2',
        'decile_rmse_var', 'decile_r2_var',
        'target_bucket_rmse_var', 'target_bucket_r2_var'
    """
    bin_levels = sorted(results_by_bins.keys())
    model_names = ['lr', 'rf', 'mlp', 'tabpfn']
    model_labels = {'lr': 'Linear', 'rf': 'RandomForest', 'mlp': 'MLP', 'tabpfn': 'TabPFN'}

    first_b = bin_levels[0]
    first_m = model_names[0]
    dec_example = results_by_bins[first_b][first_m]['decile_rmse']
    n_tiles = len(dec_example) if dec_example is not None else 10

    # 1) Variance across k-tiles vs discretisation level
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # --- Top row: k-group variance, with count bars as before ---
    ax_rmse = axes[0, 0]
    ax_r2   = axes[0, 1]
    ax_rmse_count = ax_rmse.twinx()
    ax_r2_count   = ax_r2.twinx()

    width = 0.3

    rmse_valid_count = []
    r2_valid_count   = []
    for b in bin_levels:
        rmse_counts = []
        r2_counts = []
        for m in model_names:
            dec_rmse = results_by_bins[b][m]["decile_rmse"]
            dec_r2   = results_by_bins[b][m]["decile_r2"]
            if dec_rmse is not None:
                rmse_counts.append(np.isfinite(dec_rmse).sum())
            if dec_r2 is not None:
                r2_counts.append(np.isfinite(dec_r2).sum())
        rmse_mean_count = np.mean(rmse_counts) if rmse_counts else 0.0
        r2_mean_count   = np.mean(r2_counts)   if r2_counts   else 0.0
        rmse_valid_count.append(rmse_mean_count)
        r2_valid_count.append(r2_mean_count)

    ax_rmse_count.bar(bin_levels, rmse_valid_count, width=width,
                      color='lightgrey', alpha=0.4, align='center')
    ax_r2_count.bar(bin_levels,   r2_valid_count,   width=width,
                    color='lightgrey', alpha=0.4, align='center')

    ax_rmse_count.set_ylim(0, n_tiles + 0.5)
    ax_r2_count.set_ylim(0, n_tiles + 0.5)
    ax_rmse_count.set_ylabel(f'Non-NaN k-tiles (max={n_tiles})', color='grey')
    ax_r2_count.set_ylabel(f'Non-NaN k-tiles (max={n_tiles})', color='grey')
    ax_rmse_count.tick_params(axis='y', labelcolor='grey')
    ax_r2_count.tick_params(axis='y', labelcolor='grey')

    for m in model_names:
        rmse_var = [results_by_bins[b][m]["decile_rmse_var"] for b in bin_levels]
        r2_var   = [results_by_bins[b][m]["decile_r2_var"]   for b in bin_levels]
        ax_rmse.plot(bin_levels, rmse_var, marker='o', label=model_labels.get(m, m))
        ax_r2.plot(bin_levels,   r2_var,   marker='o', label=model_labels.get(m, m))

    ax_rmse.set_xscale('log')
    ax_rmse.set_xlabel('Number of bins (n_bins)')
    ax_rmse.set_ylabel(f'Var(RMSE across {n_tiles} k-groups)')
    ax_rmse.set_title('Consistency across k-groups (RMSE variance)')
    ax_rmse.grid(True, alpha=0.3)
    ax_rmse.legend(fontsize=8)

    ax_r2.set_xscale('log')
    ax_r2.set_xlabel('Number of bins (n_bins)')
    ax_r2.set_ylabel(f'Var(R² across {n_tiles} k-groups)')
    ax_r2.set_title('Consistency across k-groups (R² variance)')
    ax_r2.grid(True, alpha=0.3)
    ax_r2.legend(fontsize=8)

    # --- Bottom row: target-bucket variance vs discretisation level ---
    ax_trmse = axes[1, 0]
    ax_tr2   = axes[1, 1]

    for m in model_names:
        t_rmse_var = [results_by_bins[b][m]["target_bucket_rmse_var"] for b in bin_levels]
        t_r2_var   = [results_by_bins[b][m]["target_bucket_r2_var"]   for b in bin_levels]
        ax_trmse.plot(bin_levels, t_rmse_var, marker='o', label=model_labels.get(m, m))
        ax_tr2.plot(bin_levels,   t_r2_var,   marker='o', label=model_labels.get(m, m))

    ax_trmse.set_xscale('log')
    ax_trmse.set_xlabel('Number of bins (n_bins)')
    ax_trmse.set_ylabel(f'Var(RMSE across {n_tiles} target buckets)')
    ax_trmse.set_title('Consistency across target buckets (RMSE variance)')
    ax_trmse.grid(True, alpha=0.3)
    ax_trmse.legend(fontsize=8)

    ax_tr2.set_xscale('log')
    ax_tr2.set_xlabel('Number of bins (n_bins)')
    ax_tr2.set_ylabel(f'Var(R² across {n_tiles} target buckets)')
    ax_tr2.set_title('Consistency across target buckets (R² variance)')
    ax_tr2.grid(True, alpha=0.3)
    ax_tr2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('regression_equity_k_and_target_variance_vs_discretisation.png',
                dpi=300, bbox_inches='tight')
    plt.show()



if __name__ == "__main__":
    import yaml
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    X, y = generate_regression_data(n_samples=config['n_samples'], n_features=config['n_features'],n_non_linear=config['n_non_linear'] ,n_informative=config['n_informative'], effective_rank=config['effective_rank'], random_state=config['random_state'], noise=config['noise'])
    results = run_regression_experiment(X, y, dbins=[2, 4, 8, 16, 32])
    plot_regression_discretisation_results(results)
    plot_regression_equity_deciles(results)