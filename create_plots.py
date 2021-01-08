"""
Create plots by showing how different uncertainty metrics behave on the half-moons dataset.
"""

# STD
import argparse

# EXT
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
import torch

# PROJECT
from src.info import AVAILABLE_MODELS, MODEL_PARAMS, TRAIN_PARAMS
from src.model_init import MODEL_CLASSES
from src.novelty_estimator import NoveltyEstimator

# CONST
SEED = 123
PLOT_DIR = "./plots"


def plot_scores(
    X_train: np.array,
    y_train: np.array,
    ne: NoveltyEstimator,
    scoring_func: str,
    img_path: str,
    show_cmap: bool = True,
) -> None:
    x = np.arange(-2.5, 3.5, 0.1)
    y = np.arange(-2, 2.5, 0.1)
    # x = np.arange(-3.5, 4.5, 0.1)
    # y = np.arange(-3, 3.5, 0.1)
    xx, yy = np.meshgrid(x, y)
    grid = np.stack([xx, yy], axis=2)
    scores = ne.get_novelty_score(grid, scoring_func)
    fig_x = 7 if show_cmap else 6
    plt.figure(figsize=(fig_x, 6), dpi=800)
    plt.contourf(xx, yy, scores, cmap=plt.cm.Purples, levels=40)
    plt.colorbar()
    plt.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        cmap="Set1",
        s=50,
        edgecolors="k",
        alpha=0.6,
    )
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.show()


if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=AVAILABLE_MODELS,
        choices=AVAILABLE_MODELS,
        help="Determine the models which are being used for this experiment.",
    )

    args = parser.parse_args()

    X_train, y_train = make_moons(n_samples=400, noise=0.125)

    for model_name in args.models:
        model_type = MODEL_CLASSES[model_name]
        param_set = MODEL_PARAMS[model_name]
        param_set["input_size"] = 2
        ne = NoveltyEstimator(
            model_type,
            model_params=param_set,
            train_params=TRAIN_PARAMS[model_name],
            method_name=model_name,
        )
        # TODO: Debug
        ne.train(X_train, y_train, X_val=X_train, y_val=y_train)
        plot_scores(X_train, y_train, ne, "entropy", "")
