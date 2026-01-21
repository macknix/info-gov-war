"""
Generate module for synthetic data generation, anonymisation, and plotting.

This module provides tools for:
- Generating synthetic regression data with optional minority group features
- Discretising continuous data
- Applying Mondrian k-anonymity
- Plotting data statistics
"""

from .data import generate_regression_data, generate_gaussian_mixture
from .discretise import discretise
from .mondrian import anonymise_mondrian
from .plotting import plot_regression_stats

__all__ = [
    'generate_regression_data',
    'generate_gaussian_mixture', 
    'discretise',
    'anonymise_mondrian',
    'plot_regression_stats',
]
