# Information Governance: Privacy-Utility Trade-offs

This project evaluates the trade-off between **privacy** (measured via k-anonymity metrics) and **utility** (measured via model performance) when anonymising data using different techniques.

## Overview

When data is anonymised for privacy protection, there is typically a loss of information that impacts downstream modelling tasks. This project provides tools to:

1. **Generate synthetic regression data** with configurable properties
2. **Anonymise data** using two different methods:
   - **Discretisation** (binning): Simple approach that groups continuous values into bins
   - **Mondrian k-anonymity**: Algorithm that guarantees each equivalence class has at least k records
3. **Measure privacy** using k-anonymity metrics:
   - Minimum k (smallest equivalence class size)
   - Discernibility metric: $\sum_{i} |EC_i|^2$
   - Normalised EC metric: (average EC size) / k
4. **Measure utility** by training regression models and evaluating:
   - Overall performance (RMSE, R²)
   - Equity across groups (performance for high vs low represented groups)

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd info-gov-war

# Install dependencies using uv
uv sync
```

## Configuration

Edit `config.yaml` to configure the experiments:

```yaml
# Data generation
n_samples: 10000
n_features: 6
n_informative: 4
effective_rank: 3
noise: 0.1
random_state: 45

# Discretisation pipeline settings
discretisation_levels: [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
discretisation_strategy: uniform  # or 'quantile', 'kmeans'

# Mondrian pipeline settings
mondrian_k_values: [2, 5, 10, 20, 50, 100]
```

## Usage

### Run Both Pipelines

```bash
uv run python3 -m src.pipeline
```

### Run Individual Pipelines

```bash
# Discretisation pipeline only
uv run python3 -m src.pipeline --discretisation

# Mondrian pipeline only
uv run python3 -m src.pipeline --mondrian
```

### Custom Config File

```bash
uv run python3 -m src.pipeline --config my_config.yaml
```

## Pipelines

### 1. Discretisation Pipeline (`--discretisation`)

Evaluates privacy-utility trade-off when using binning/discretisation.

**Process:**
1. Generate synthetic regression data
2. For each number of bins (2, 4, 5, ...):
   - Discretise features into bins
   - Compute k-anonymity metrics
   - Train models (Linear, RandomForest, MLP, TabPFN)
   - Evaluate utility and equity

**Outputs:**
- `k_anonymity_vs_discretisation.png` - Privacy metrics vs number of bins
- `regression_utility_equity_vs_discretisation.png` - Model performance vs bins
- `regression_equity_k_and_target_variance_vs_discretisation.png` - Equity analysis

### 2. Mondrian Pipeline (`--mondrian`)

Evaluates privacy-utility trade-off using Mondrian k-anonymity algorithm.

**Process:**
1. Generate synthetic regression data
2. For each target k value (2, 5, 10, ...):
   - Apply Mondrian anonymisation (guarantees k-anonymity)
   - Compute privacy metrics (discernibility, NCP)
   - Train models on anonymised data
   - Evaluate utility and equity

**Outputs:**
- `mondrian_k_anonymity_stats.png` - Privacy metrics vs k
- `mondrian_regression_utility_equity.png` - Model performance vs k
- `mondrian_equity_variance.png` - Equity consistency analysis

## Key Metrics

### Privacy Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **k-anonymity** | Minimum equivalence class size | Higher = more privacy |
| **Discernibility** | $\sum \|EC_i\|^2$ | Lower = better (more even distribution) |
| **Normalised EC** | (avg EC size) / k | 1 = optimal, higher = more variance |
| **NCP** | Normalised Certainty Penalty | Lower = less information loss |

### Utility Metrics

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Square Error (lower = better) |
| **R²** | Coefficient of determination (higher = better) |

### Equity Metrics

| Metric | Description |
|--------|-------------|
| **Bottom/Top 25%** | Performance for records in smallest/largest ECs |
| **Decile Variance** | Variance of performance across k-groups |

## Project Structure

```
info-gov-war/
├── config.yaml                 # Configuration file
├── src/
│   ├── pipeline.py             # Main entry point
│   ├── pipeline_discretisation.py  # Discretisation pipeline
│   ├── pipeline_mondrian.py    # Mondrian pipeline
│   ├── generate.py             # Data generation utilities
│   ├── k_anonymity.py          # k-anonymity metrics and plots
│   ├── mondrian_anonymiser.py  # Mondrian algorithm implementation
│   └── modelling.py            # Model training and evaluation
├── notebooks/
│   └── exploration.ipynb       # Exploratory analysis
└── tests/
    └── ...                     # Unit tests
```

## Mondrian Algorithm

The Mondrian algorithm is a multidimensional k-anonymity method that:

1. Starts with all records in one partition
2. Recursively splits partitions along the dimension with the largest normalised range
3. Stops when a split would create a partition with fewer than k records
4. Generalises all records in a partition to the same value (e.g., midpoint)

**Advantages over discretisation:**
- **Guarantees** k-anonymity (discretisation does not)
- Adaptive bin boundaries based on data distribution
- Provides NCP (information loss) metric

## References

- LeFevre, K., DeWitt, D. J., & Ramakrishnan, R. (2006). Mondrian Multidimensional K-Anonymity. *ICDE '06*.
- Sweeney, L. (2002). k-anonymity: A model for protecting privacy. *International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems*.

## License

MIT
