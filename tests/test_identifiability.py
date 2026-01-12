import numpy as np
import pandas as pd
from src.data.generate import generate_population_data
from src.discretization.quantile import quantile_discretize
from src.identifiability.evaluate import identify_accuracy

def test_identifiability():
    # Generate a population with a specified covariance structure
    N_total = 1000000
    D = 6
    population_data = generate_population_data(N_total, D)

    # Split the population into training and testing sets
    N_train = 800000
    latent_train = population_data[:N_train]
    latent_test = population_data[N_train:]

    # Test different levels of discretization
    bins_list = [4, 8, 16, 36, 64, 128]
    results = []

    for bins in bins_list:
        # Discretize the training and testing data
        train_disc = quantile_discretize(latent_train, bins)
        test_disc = quantile_discretize(latent_test, bins)

        # Evaluate identifiability
        accuracy = identify_accuracy(train_disc, test_disc, bins)
        results.append((bins, accuracy))

    # Convert results to DataFrame for easier analysis
    results_df = pd.DataFrame(results, columns=["Discretization_bins", "Identifiability"])
    
    # Assert that the results are as expected (you can adjust the expected values)
    assert results_df["Identifiability"].max() > 0.5, "Identifiability should be greater than 50% for some bins"
    assert results_df["Identifiability"].min() >= 0, "Identifiability should not be negative"
    print("All tests passed!")

# Run the test function
test_identifiability()