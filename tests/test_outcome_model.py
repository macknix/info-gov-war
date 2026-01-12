import numpy as np
import pandas as pd
import pytest
from src.personalization.outcome_model import model_outcome, evaluate_outcome_identifiability

# Sample data for testing
def generate_sample_data(num_samples=1000):
    np.random.seed(42)
    # Generate a population with 6 variables
    mu = np.zeros(6)
    Sigma = 0.5 * np.ones((6, 6)) + 0.5 * np.eye(6)  # correlated structure
    data = np.random.multivariate_normal(mu, Sigma, size=num_samples)
    return data

def test_model_outcome():
    data = generate_sample_data()
    outcome = model_outcome(data)
    
    # Check if the outcome has the correct shape
    assert outcome.shape == (data.shape[0],), "Outcome shape mismatch"
    
    # Check if the outcome is a valid numerical array
    assert np.issubdtype(outcome.dtype, np.number), "Outcome should be numerical"

def test_evaluate_outcome_identifiability():
    data = generate_sample_data()
    identifiability_score = evaluate_outcome_identifiability(data)
    
    # Check if the identifiability score is between 0 and 1
    assert 0 <= identifiability_score <= 1, "Identifiability score out of bounds"