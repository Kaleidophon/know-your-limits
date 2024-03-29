"""
Define how models for experiments are initialized.
"""

# STD
from typing import Optional, Iterable, Dict, Tuple, List, Type

# PROJECT
from src.nn_ensemble import NNEnsemble
from src.anchored_ensemble import AnchoredNNEnsemble
from src.mlp import (
    MLP,
    MCDropoutMLP,
    PlattScalingMLP,
)
from src.bbb import BBBMLP
from src.info import (
    AVAILABLE_MODELS,
    AVAILABLE_SCORING_FUNCS,
    TRAIN_PARAMS,
    MODEL_PARAMS,
    NO_ENSEMBLE_NN_MODELS,
    ENSEMBLE_MODELS,
)
from src.novelty_estimator import NoveltyEstimator

# Model classes
MODEL_CLASSES = {
    "NN": MLP,
    "PlattScalingNN": PlattScalingMLP,
    "BBB": BBBMLP,
    "MCDropout": MCDropoutMLP,
    "NNEnsemble": NNEnsemble,
    "BootstrappedNNEnsemble": NNEnsemble,
    "AnchoredNNEnsemble": AnchoredNNEnsemble,
}


def init_models(
    input_dim: int,
    selection: Iterable[str] = AVAILABLE_MODELS,
    model_classes: Dict[str, Type] = MODEL_CLASSES,
    scoring_funcs: Dict[str, Tuple[Optional[str], ...]] = AVAILABLE_SCORING_FUNCS,
):
    """
    Initialize the models for the experiments.

    Parameters
    ----------
    input_dim: int
        Dimensionality of input for models.
    selection: Iterable[str]
        Iterable of strings defining the names of all the models that will be initialized. Default is the set
        AVAILABLE_MODELS defined in uncertainty_estimation.models.infos.
    model_classes: Dict[str, Type]
        Dictionary mapping from model class name to their Python types.
    scoring_funcs: Dict[str, Tuple[Optional[str], ...]]
        Specify the scoring functions by name by model which are being used to determine the novelty of a sample.

    Returns
    -------
    List[ModelInfo]
        A list of tuples containing the model, the names of metrics used for uncertainty estimation as a tuple as well
        as the model name.
    """
    selection = set(selection)
    assert len(AVAILABLE_MODELS.intersection(selection)) == len(
        selection
    ), f"Unknown models specified: {', '.join([name for name in selection if name not in AVAILABLE_MODELS])}"

    def _add_input_dim(model_params, model_name):
        """ Add input_size parameter to the model parameter dict if necessary. """
        if model_name in NO_ENSEMBLE_NN_MODELS:
            return {"input_size": input_dim, **model_params}

        if model_name in ENSEMBLE_MODELS:
            model_params["model_params"]["input_size"] = input_dim

        return model_params

    return [
        (
            NoveltyEstimator(
                model_classes[model_name],
                _add_input_dim(MODEL_PARAMS[model_name], model_name),
                TRAIN_PARAMS[model_name],
                model_name,
            ),
            scoring_funcs[model_name],
            model_name,
        )
        for model_name in selection
    ]
