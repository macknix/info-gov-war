import numpy as np

# Configuration settings for the simulations

# Population parameters
N_total = 1000000       # total population size
N_train = 800000        # training cohort size
N_test = N_total - N_train
D = 6                    # number of variables

# Covariance structure
covariance_scale = 0.5  # scale for covariance
Sigma = covariance_scale * np.ones((D, D)) + (1 - covariance_scale) * np.eye(D)

# Discretization parameters
bins_list = [4, 8, 16, 36, 64, 128]  # levels of discretization

# Clinical outcome variable index
clinical_outcome_index = 0  # assuming the first variable is the clinical outcome

# Set reproducibility
np.random.seed(42)