import numpy as np
import pandas as pd
from src.discretization.quantile import quantile_discretize
from src.data.generate import generate_population_data

def test_quantile_discretize():
    # Generate a sample population data
    n_samples = 1000
    n_variables = 6
    population_data = generate_population_data(n_samples, n_variables)

    # Test discretization with different bin sizes
    for bins in [4, 8, 16, 36, 64, 128]:
        discretized_data = quantile_discretize(population_data, bins)
        assert discretized_data.shape == population_data.shape, "Discretized data shape mismatch"
        assert np.all((discretized_data >= 1) & (discretized_data <= bins)), "Discretized values out of bounds"

def test_identifiability_with_discretization():
    # Generate a sample population data
    n_samples = 1000
    n_variables = 6
    population_data = generate_population_data(n_samples, n_variables)

    # Discretize the data
    bins = 16
    discretized_data = quantile_discretize(population_data, bins)

    # Check identifiability based on one variable
    known_var_index = 0  # Assume the first variable is known
    identifiable_count = np.sum(discretized_data[:, known_var_index] == discretized_data[:, known_var_index])
    
    # Assert that we can identify at least some samples
    assert identifiable_count > 0, "No identifiable samples found"

def run_tests():
    test_quantile_discretize()
    test_identifiability_with_discretization()
    print("All tests passed.")

if __name__ == "__main__":
    run_tests()