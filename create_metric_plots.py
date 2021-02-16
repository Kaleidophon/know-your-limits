"""
Create plots by showing how different uncertainty metrics behave on the half-moons dataset.
"""

# STD
import argparse
from typing import Optional

# EXT
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import roc_auc_score
import torch
from tqdm import tqdm

# PROJECT
from src.info import (
    AVAILABLE_MODELS,
    MODEL_PARAMS,
    TRAIN_PARAMS,
    AVAILABLE_SCORING_FUNCS,
    NO_ENSEMBLE_NN_MODELS,
)
from src.model_init import MODEL_CLASSES
from src.novelty_estimator import NoveltyEstimator

# CONST
SEED = 123
PLOT_DIR = "./plots"
CMAP_RANGES = {
    "var": [0, 0.25],
    "entropy": [0, 1],
    "mutual_information": [4, 5],
    "max_prob": [0, 0.5],
}


def plot_scores(
    X_train: np.array,
    y_train: np.array,
    ne: NoveltyEstimator,
    scoring_func: str,
    img_path: str,
    show_cmap: bool = True,
    add_roc_auc: Optional[float] = None,
) -> None:
    # x = np.arange(-2.5, 3.5, 0.1)
    # y = np.arange(-2, 2.5, 0.1)
    x = np.arange(-3.5, 4.5, 0.1)
    y = np.arange(-3, 3.5, 0.1)
    xx, yy = np.meshgrid(x, y)
    grid = torch.Tensor(np.stack([xx, yy], axis=2))

    # Create figure with uncertainty scores
    scores = ne.get_novelty_score(grid, scoring_func).detach().numpy()
    fig_x = 9 if show_cmap else 8
    plt.figure(figsize=(fig_x, 8), dpi=200)
    vmin, vmax = CMAP_RANGES[scoring_func]
    plt.contourf(xx, yy, scores, cmap=plt.cm.Purples, levels=40, vmin=vmin, vmax=vmax)

    if show_cmap:
        plt.colorbar()

    colors = np.array(["#FFD700"] * y_train.shape[0])
    colors[y_train == 1] = "#3CB371"
    plt.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=colors,
        s=50,
        edgecolors="k",
        alpha=0.6,
    )
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    if add_roc_auc is not None:
        ax.text(
            0.025,
            0.025,
            f"AUC-ROC: {add_roc_auc:.2f}",
            horizontalalignment="left",
            verticalalignment="bottom",
            transform=ax.transAxes,
            fontsize=24,
            bbox=dict(
                facecolor="white",
                alpha=0.6,
                edgecolor="black",
                boxstyle="round,pad=0.5",
            ),
        )

    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()

    # Create figure with gradient magnitude of uncertainty scores w.r.t. data
    x = np.arange(-5.5, 6.5, 0.1)
    y = np.arange(-5, 5.5, 0.1)
    xx, yy = np.meshgrid(x, y)
    grid = torch.Tensor(np.stack([xx, yy], axis=2))
    grad_magnitudes = (
        ne.get_novelty_score_grad_magnitude(grid, scoring_func).detach().numpy()
    )
    plt.figure(figsize=(9, 8), dpi=200)
    plt.contourf(xx, yy, grad_magnitudes, cmap=plt.cm.YlGn, levels=40)

    # if show_cmap:
    plt.colorbar()

    plt.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=colors,
        s=50,
        edgecolors="k",
        alpha=0.6,
    )
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    if add_roc_auc is not None:
        ax.text(
            0.025,
            0.025,
            f"AUC-ROC: {add_roc_auc:.2f}",
            horizontalalignment="left",
            verticalalignment="bottom",
            transform=ax.transAxes,
            fontsize=24,
            bbox=dict(
                facecolor="white",
                alpha=0.6,
                edgecolor="black",
                boxstyle="round,pad=0.5",
            ),
        )

    plt.tight_layout()
    plt.savefig(img_path.replace(".png", "_grads.png"))
    plt.close()


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
    parser.add_argument(
        "--plot_dir",
        type=str,
        default=PLOT_DIR,
        help="Directory for plots.",
    )

    args = parser.parse_args()
    X_train, y_train = make_moons(n_samples=750, noise=0.125)
    X_train, X_val = torch.Tensor(X_train[:500, :]), torch.Tensor(X_train[500:, :])
    y_train, y_val = torch.Tensor(y_train[:500]), torch.Tensor(y_train[500:])

    for model_name in tqdm(args.models):
        model_type = MODEL_CLASSES[model_name]
        param_set = MODEL_PARAMS[model_name]

        if model_name in NO_ENSEMBLE_NN_MODELS:
            param_set["input_size"] = 2
        else:
            param_set["model_params"]["input_size"] = 2

        ne = NoveltyEstimator(
            model_type,
            model_params=param_set,
            train_params=TRAIN_PARAMS[model_name],
            method_name=model_name,
        )
        ne.train(X_train, y_train, X_val=X_val, y_val=y_val)
        y_hat = torch.argmax(ne.model.predict_proba(X_train), dim=1).numpy()
        roc_auc = roc_auc_score(y_train, y_hat)

        for scoring_func in AVAILABLE_SCORING_FUNCS[model_name]:
            plot_scores(
                X_train,
                y_train,
                ne,
                scoring_func=scoring_func,
                img_path=f"{args.plot_dir}/{model_name.lower()}_{scoring_func}.png",
                add_roc_auc=roc_auc,
                show_cmap=False,
            )
