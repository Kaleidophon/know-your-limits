"""
Module to be the single place to bundle all the information about models: Available models and their name,
hyperparameters, etc.
"""

# EXT
import numpy as np
from scipy.stats import uniform

# ### Models and novelty scoring functions ###

SINGLE_PRED_NN_MODELS = {
    "NN",  # Single neural discriminator
    "PlattScalingNN",  # Single neural discriminator with platt scaling
}

ENSEMBLE_MODELS = {
    "NNEnsemble",  # Ensemble of neural discriminators
    "AnchoredNNEnsemble",  # Bayesian ensemble of neural discriminators with special regularization
}

SINGLE_INST_MULTIPLE_PRED_NN_MODELS = {
    "MCDropout",  # Single neural discriminator using MC Dropout for uncertainty estimation
    "BBB",  # Bayesian Neural Network
}

NO_ENSEMBLE_NN_MODELS = SINGLE_PRED_NN_MODELS | SINGLE_INST_MULTIPLE_PRED_NN_MODELS

AVAILABLE_MODELS = NO_ENSEMBLE_NN_MODELS | ENSEMBLE_MODELS

# Available novelty scoring functions for models
AVAILABLE_SCORING_FUNCS = {
    "NN": ("entropy", "max_prob"),
    "PlattScalingNN": ("entropy", "max_prob"),
    "MCDropout": ("entropy", "var", "mutual_information"),
    "BBB": ("entropy", "var", "mutual_information"),
    "NNEnsemble": ("entropy", "var", "mutual_information"),
    "AnchoredNNEnsemble": ("entropy", "var", "mutual_information"),
}

# ### Hyperparameters ###

MODEL_PARAMS = {
    "NN": {
        "dropout_rate": 0.365624,
        "hidden_sizes": [20, 20],
        "lr": 0.000538,
    },
    "PlattScalingNN": {
        "dropout_rate": 0.365624,
        "hidden_sizes": [20, 20],
        "lr": 0.000538,
    },
    "MCDropout": {
        "dropout_rate": 0.369903,
        "hidden_sizes": [15, 15],
        "lr": 0.000526,
        "class_weight": False,
    },
    "BBB": {
        "anneal": False,
        "beta": 2.299666,
        "dropout_rate": 0.170072,
        "hidden_sizes": [20],
        "posterior_mu_init": -0.048948,
        "posterior_rho_init": -0.733795,
        "prior_pi": 0.527592,
        "prior_sigma_1": 0.246597,
        "prior_sigma_2": 0.055023,
        "lr": 0.000731,
        "class_weight": False,
    },
    "NNEnsemble": {
        "n_models": 10,
        "bootstrap": False,
        "model_params": {
            "dropout_rate": 0.365624,
            "hidden_sizes": [20, 20],
            "lr": 0.000538,
        },
    },
    "AnchoredNNEnsemble": {
        "n_models": 10,
        "model_params": {
            "dropout_rate": 0.365624,
            "hidden_sizes": [20, 20],
            "lr": 0.000538,
        },
    },
}

TRAIN_PARAMS = {
    "NN": {
        "batch_size": 256,
        "early_stopping": True,
        "early_stopping_patience": 3,
        "n_epochs": 10,
    },
    "PlattScalingNN": {
        "batch_size": 256,
        "early_stopping": True,
        "early_stopping_patience": 3,
        "n_epochs": 10,
    },
    "MCDropout": {
        "batch_size": 256,
        "early_stopping": True,
        "early_stopping_patience": 3,
        "n_epochs": 10,
    },
    "BBB": {
        "batch_size": 256,
        "early_stopping": True,
        "early_stopping_patience": 6,
        "n_epochs": 10,
    },
    "NNEnsemble": {
        "batch_size": 256,
        "early_stopping": True,
        "early_stopping_patience": 3,
        "n_epochs": 8,
    },
    "AnchoredNNEnsemble": {
        "batch_size": 256,
        "early_stopping": True,
        "early_stopping_patience": 3,
        "n_epochs": 10,
    },
}

# Hyperparameter ranges / distributions that should be considered during the random search
PARAM_SEARCH = {
    "hidden_sizes": [
        [hidden_size] * num_layers
        for hidden_size in [5, 10, 15, 20]
        for num_layers in range(1, 3)
    ],
    "batch_size": [16, 32, 64],
    # Intervals become [loc, loc + scale] for uniform
    "dropout_rate": uniform(loc=0, scale=0.5),  # [0, 0.5]
    "posterior_rho_init": uniform(loc=-8, scale=8),  # [-8, -2]
    "posterior_mu_init": uniform(loc=-0.9, scale=1.8),  # [-0.6, 0.6]
    "prior_pi": uniform(loc=0.05, scale=0.9),  # [0.1, 0.9]
    "prior_sigma_1": [np.exp(d) for d in np.arange(-4, 0, 0.1)],
    "prior_sigma_2": [np.exp(d) for d in np.arange(-4, 0, 0.1)],
    "anneal": [True, False],
    "beta": uniform(loc=0.05, scale=2.4),  # [0.1, 2.5]
}
NUM_EVALS = {
    "NN": 100,
    "MCDropout": 100,
    "BBB": 500,
}


# Default training hyperparameters
DEFAULT_LEARNING_RATE: float = 1e-2
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_N_EPOCHS: int = 6
DEFAULT_EARLY_STOPPING_PAT: int = 2

DEFAULT_RECONSTR_ERROR_WEIGHT: float = 1e20
DEFAULT_N_VAE_SAMPLES: int = 100
