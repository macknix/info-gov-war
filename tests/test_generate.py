import numpy as np
import pytest
from src.data.generate import generate_population_data

def test_generate_population_data():
    # Test parameters
    n_samples = 1000
    n_variables = 6
    covariance_structure = np.array([[1, 0.5, 0.5, 0.5, 0.5, 0.5],
                                      [0.5, 1, 0.5, 0.5, 0.5, 0.5],
                                      [0.5, 0.5, 1, 0.5, 0.5, 0.5],
                                      [0.5, 0.5, 0.5, 1, 0.5, 0.5],
                                      [0.5, 0.5, 0.5, 0.5, 1, 0.5],
                                      [0.5, 0.5, 0.5, 0.5, 0.5, 1]])

    # Generate population data
    data = generate_population_data(n_samples, covariance_structure)

    # Check the shape of the generated data
    assert data.shape == (n_samples, n_variables), "Shape of generated data is incorrect."

    # Check if the covariance structure is approximately preserved
    sample_cov = np.cov(data, rowvar=False)
    assert np.allclose(sample_cov, covariance_structure, atol=0.1), "Covariance structure is not preserved."

    # Check for NaN values in the generated data
    assert not np.any(np.isnan(data)), "Generated data contains NaN values."

    # Check for the mean of the generated data
    assert np.allclose(data.mean(axis=0), np.zeros(n_variables), atol=0.1), "Mean of generated data is not close to zero."