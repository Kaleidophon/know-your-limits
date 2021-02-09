"""
Define some utility functions.
"""

# STD
from typing import Tuple, Union

# EXT
import numpy as np
import torch
from torch.utils.data import Dataset


def entropy(probabilities: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Entropy of a probability distribution.

    Parameters
    ----------
    probabilities: torch.Tensor
        Probabilities per class.
    dim: int
        Axis over which the entropy should be calculated.

    Returns
    -------
    torch.Tensor
        Entropy of the predicted distribution.
    """
    return -torch.sum(probabilities * torch.log2(probabilities + 1e-8), dim=dim)


def max_prob(probabilities: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Implement the baseline from [1], which just uses the maximum (softmax) probability as a OOD detection score.

    [1] https://arxiv.org/abs/1610.02136

    Parameters
    ----------
    probabilities: torch.Tensor
        Probabilities per class.
    dim: int
        Axis over which the max should be taken.

    Returns
    -------
    torch.Tensor
        Max class probability per sample.
    """
    return 1 - torch.max(probabilities, dim)[0]


class SimpleDataset(Dataset):
    """
    Create a new (simple) PyTorch Dataset instance.

    Parameters
    ----------
    X: torch.Tensor
        Predictors
    y: torch.Tensor
        Target
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self):
        """Return the number of items in the dataset.

        Returns
        -------
        type: int
            The number of items in the dataset.
        """
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return X and y at index idx.

        Parameters
        ----------
        idx: int
            Index.

        Returns
        -------
        type: Tuple[torch.Tensor, torch.Tensor]
            X and y at index idx
        """
        return self.X[idx], self.y[idx]
