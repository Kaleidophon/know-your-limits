"""
Module defining a wrapper class for all define modules such that they can be used to assess the novelty of a data
sample. The way that novelty is scored depends on the model type and scoring function.
"""

# EXT
import numpy as np

# PROJECT
from src.utils import entropy, max_prob
from src.info import AVAILABLE_MODELS, ENSEMBLE_MODELS

# CONST
# Define all combination of possible models and scoring funcs
SCORING_FUNCS = {
    ("NN", "entropy"): lambda model, data: entropy(model.predict_proba(data), axis=1),
    ("NN", "max_prob"): lambda model, data: max_prob(model.predict_proba(data), axis=1),
    ("PlattScalingNN", "entropy"): lambda model, data: entropy(
        model.predict_proba(data), axis=1
    ),
    ("PlattScalingNN", "max_prob"): lambda model, data: max_prob(
        model.predict_proba(data), axis=1
    ),
    ("MCDropout", "entropy"): lambda model, data: entropy(
        model.predict_proba(data), axis=1
    ),
    ("MCDropout", "std"): lambda model, data: model.get_std(data),
    (
        "MCDropout",
        "mutual_information",
    ): lambda model, data: model.get_mutual_information(data),
    ("BBB", "entropy"): lambda model, data: entropy(model.predict_proba(data), axis=1),
    ("BBB", "std"): lambda model, data: model.get_std(data),
    ("BBB", "mutual_information"): lambda model, data: model.get_mutual_information(
        data
    ),
    ("NNEnsemble", "entropy"): lambda model, data: entropy(
        model.predict_proba(data), axis=1
    ),
    ("NNEnsemble", "std"): lambda model, data: model.get_std(data),
    (
        "NNEnsemble",
        "mutual_information",
    ): lambda model, data: model.get_mutual_information(data),
    ("AnchoredNNEnsemble", "entropy"): lambda model, data: entropy(
        model.predict_proba(data), axis=1
    ),
    ("AnchoredNNEnsemble", "std"): lambda model, data: model.get_std(data),
    (
        "AnchoredNNEnsemble",
        "mutual_information",
    ): lambda model, data: model.get_mutual_information(data),
}


class NoveltyEstimator:
    """
    Wrapper class for novelty estimation methods

    Parameters
    ----------
    model_type:
        the model to use, e.g. AE, PCA
    model_params: dict
        The parameters used when initializing the model.
    train_params: dict
        The parameters used when fitting the model.
    method_name: str
        Which type of method: 'AE', or 'sklearn' for a sklearn-style novelty detector.
    """

    def __init__(self, model_type, model_params, train_params, method_name):
        self.model_type = model_type
        self.name = method_name
        self.model_params = model_params
        self.train_params = train_params

    def train(self, X_train, y_train, X_val, y_val):
        """
        Fit the novelty estimator.

        Parameters
        ----------
        X_train: np.array
            Training samples.
        y_train: np.array
            Training labels.
        X_val: np.array
            Validation samples.
        y_val: np.array
            Validation labels.
        """
        if self.name in ENSEMBLE_MODELS:
            self.model = self.model_type(**self.model_params)
            self.model.train(
                X_train, y_train, X_val, y_val, training_params=self.train_params
            )

        else:
            self.model = self.model_type(**self.model_params)
            self.model.train(X_train, y_train, X_val, y_val, **self.train_params)

    def get_novelty_score(self, data, scoring_func: str):
        """Apply the novelty estimator to obtain a novelty score for the data.

        Parameters
        ----------
        data: np.ndarray
            The data for which we want to get a novelty score
        scoring_func: str
            Name of function that is used to assess novelty.

        Returns
        -------
        np.ndarray
            The novelty estimates.
        """

        assert self.name in AVAILABLE_MODELS, (
            f"Unknown model {self.name} found, has to be one of "
            f"{', '.join(AVAILABLE_MODELS)}."
        )

        assert (self.name, scoring_func) in SCORING_FUNCS.keys(), (
            f"Unknown combination of {self.name} and "
            f"{scoring_func} found, has it been added to "
            f"SCORING_FUNCS in src.models.novelty_estimator.py?"
        )

        return SCORING_FUNCS[(self.name, scoring_func)](self.model, data)
