"""
Module containing implementations of (different variations of) multi-layer perceptrons:

* Vanilla MLP
* MLP with Platt Scaling (Guo et al., 2017)
* MC Dropout MLP (Gal et al., 2016)

https://arxiv.org/abs/1706.04599
http://proceedings.mlr.press/v48/gal16.pdf
"""

# STD
from typing import Dict, Any, List, Type, Callable, Optional

# EXT
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader

# PROJECT
from src.utils import entropy, SimpleDataset

# CONST
# CONST
DEFAULT_LEARNING_RATE: float = 1e-2
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_N_EPOCHS: int = 10
DEFAULT_EARLY_STOPPING_PAT: int = 4


class MLPModule(nn.Module):
    """
    Base class for a multilayer perceptron.
    """

    def __init__(
        self,
        hidden_sizes: List[int],
        input_size: int,
        dropout_rate: float,
        output_size: int = 1,
        layer_class: nn.Module = nn.Linear,
        layer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a multi-layer perceptron.

        Parameters
        ----------
        hidden_sizes: List[int]
            List specifying the sizes of hidden layers.
        input_size: int
            Dimensionality of input samples.
        dropout_rate: float
            Dropout rate for linear layers.
        output_size: int
            Number of output units, default is 1.
        layer_class: Type
            Class of the linear layer, default is nn.Linear.
        layer_kwargs: Optional[Dict[str, Any]]
            Key-word arguments for layer class.
        """
        super().__init__()
        layer_kwargs = {} if layer_kwargs is None else layer_kwargs

        layers = []

        hidden_sizes = [input_size] + hidden_sizes + [output_size]

        for l, (in_dim, out_dim) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:])):
            layers.append(layer_class(in_dim, out_dim, **layer_kwargs))

            if l < len(hidden_sizes) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))

        self.mlp = nn.Sequential(*layers)

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass of the MLP.

        Parameters
        ----------
        _input: torch.Tensor
            The input of the model.

        Returns
        -------
        type: torch.Tensor
            The output of the model.
        """
        out = self.mlp(_input)
        return out


class MLP:
    """
    Handles training of an MLPModule.

    Parameters
    ----------
    hidden_sizes: list
        The sizes of the hidden layers.
    input_size: int
        The input size.
    dropout_rate: float
        The dropout rate applied after each layer (except the output layer)
    output_size: int
        The output size.
    """

    def __init__(
        self,
        hidden_sizes: list,
        input_size: int,
        dropout_rate: float,
        class_weight: bool = True,
        output_size: int = 1,
        lr: float = 1e-3,
        mlp_module: MLPModule = MLPModule,
        **mlp_kwargs,
    ):
        self.model = mlp_module(
            hidden_sizes, input_size, dropout_rate, output_size, **mlp_kwargs
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.class_weight = class_weight
        self.lr = lr

    def _initialize_dataloader(
        self, X_train: torch.Tensor, y_train: torch.Tensor, batch_size: int
    ):
        """
        Initialize the dataloader of the train data.

        Parameters
        ----------
        X_train: torch.Tensor
            The training data.
        y_train: torch.Tensor
            The labels corresponding to the training data.
        batch_size:
            The batch size.
        """
        train_set = SimpleDataset(X_train, y_train)
        self.train_loader = DataLoader(train_set, batch_size, shuffle=True)

    def get_loss(
        self, X: torch.Tensor, y: torch.Tensor, train: bool = True
    ) -> torch.Tensor:
        """
        Obtain the loss for the current batch.

        Parameters
        ----------
        X: torch.Tensor
            Data sample for which the loss should be computed for.
        y: torch.Tensor
            Labels for the current batch.
        train: bool
            Specify whether the training or validation loss function is used (differs for BBB).

        Returns
        -------
        loss: torch.FloatTensor
            Loss for current batch.
        """
        y_pred = self.model(X)
        mean_y = y.mean()

        if self.class_weight:
            if mean_y == 0:
                pos_weight = torch.tensor(0.0)
            elif mean_y == 1:
                pos_weight = torch.tensor(1.0)
            else:
                pos_weight = (1 - mean_y) / mean_y

        else:
            # When not using class weighting, the weight is simply 1.
            pos_weight = torch.tensor(1.0)

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = loss_fn(y_pred, y)

        return loss

    def validate(self, X_val: np.ndarray, y_val: np.ndarray) -> torch.Tensor:
        """
        Calculate the validation loss.

        Parameters
        ----------
        X_val: np.ndarray
            The validation data.
        y_val: np.ndarray
            The labels corresponding to the validation data.

        Returns
        -------
        type: torch.Tensor
            The validation loss.
        """
        self.model.eval()
        X = X_val.float()
        y = y_val.float().view(-1, 1)

        val_loss = self.get_loss(X, y, train=False)

        return val_loss

    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        n_epochs: int = DEFAULT_N_EPOCHS,
        early_stopping: bool = True,
        early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PAT,
    ):
        """
        Train the MLP.

        Parameters
        ----------
        X_train: torch.Tensor
            The training data.
        y_train: torch.Tensor
            The labels corresponding to the training data.
        X_val: Optional[torch.Tensor]
            The validation data.
        y_val: Optional[torch.Tensor]
            The labels corresponding to the validation data.
        batch_size: int
            The batch size, default 256
        n_epochs: int
            The number of training epochs, default 30
        early_stopping: bool
            Whether to perform early stopping, default True
        early_stopping_patience: int
            The early stopping patience, default 2.
        """
        self._initialize_dataloader(X_train, y_train, batch_size)
        prev_val_loss = float("inf")
        n_no_improvement = 0
        for epoch in range(n_epochs):

            self.model.train()

            for batch_X, batch_y in self.train_loader:

                loss = self.get_loss(batch_X.float(), batch_y.float().view(-1, 1))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if early_stopping and X_val is not None and y_val is not None:
                val_loss = self.validate(X_val, y_val)

                if val_loss >= prev_val_loss:
                    n_no_improvement += 1

                else:
                    n_no_improvement = 0
                    prev_val_loss = val_loss

            if n_no_improvement >= early_stopping_patience:
                print("Early stopping after", epoch, "epochs.")
                break

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor, **train_kwargs):
        """
        Fit an MLP to a dataset. Implemented to ensure compatibility to scikit-learn.

        Parameters
        ----------
        X_train: torch.Tensor
            The training data.
        y_train: torch.Tensor
            The labels corresponding to the training data.
        """
        self.train(X_train, y_train, **train_kwargs)

    def predict_proba(self, X_test: torch.Tensor) -> torch.Tensor:
        """
        Predict the probabilities for a batch of samples.

        Parameters
        ----------
        X_test: np.array
            Batch of samples as torch tensor.

        Returns
        -------
        np.array
            Predictions for every sample.
        """
        X_test_tensor = X_test.float()

        self.model.eval()
        predictions = torch.sigmoid(self.model(X_test_tensor)).squeeze()

        return torch.stack([1 - predictions, predictions], dim=1)

    def predict(self, X_test: torch.Tensor) -> torch.Tensor:
        """
        Same as predict_proba(). Implement for compatibility with scikit learn.

        Parameters
        ----------
        X_test: np.array
            Batch of samples as torch tensor.

        Returns
        -------
        torch.Tensor
            Predictions for every sample.
        """
        return self.predict_proba(X_test)

    def eval(self) -> None:
        self.model.eval()


class PlattScalingMLP(MLP):
    """
    Handles the training of a MLP module with Platt scaling.
    """

    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        n_epochs: int = DEFAULT_N_EPOCHS,
        early_stopping: bool = True,
        early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PAT,
    ):
        """
        Train the MLP.

        Parameters
        ----------
        X_train: torch.Tensor
            The training data.
        y_train: torch.Tensor
            The labels corresponding to the training data.
        X_val: torch.Tensor
            The validation data.
        y_val: torch.Tensor
            The labels corresponding to the validation data.
        batch_size: int
            The batch size, default 256
        n_epochs: int
            The number of training epochs, default 30
        early_stopping: bool
            Whether to perform early stopping, default True
        early_stopping_patience: int
            The early stopping patience, default 2.
        """
        # Do the regular training first
        super().train(
            X_train,
            y_train,
            X_val,
            y_val,
            batch_size,
            n_epochs,
            early_stopping,
            early_stopping_patience,
        )

        # Now learn the platt scaling on the validation set
        self.model.eval()
        scaling_layer = nn.Linear(1, 1).train()

        val_set = SimpleDataset(X_val, y_val)
        val_loader = DataLoader(val_set, batch_size, shuffle=True)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(scaling_layer.parameters(), lr=0.1)

        prev_val_loss = float("inf")
        n_no_improvement = 0

        for epoch in range(n_epochs):

            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.float(), batch_y.float().view(-1, 1)
                optimizer.zero_grad()

                model_out = self.model(batch_X)
                out = scaling_layer(model_out)
                loss = loss_fn(out, batch_y)
                loss.backward()
                optimizer.step()

                if loss >= prev_val_loss:
                    n_no_improvement += 1

                else:
                    n_no_improvement = 0
                    prev_val_loss = loss

            if n_no_improvement >= early_stopping_patience:
                print(f"Early stopping platt scale training after {epoch} epochs.")
                break

        # Add scaling layer to model
        self.model.mlp.add_module("platt_scaling", scaling_layer)
        self.model.train()


class MultiplePredictionsMixin:
    """
    Mixin class adding functions that are used for models that are able to produce multiple, different predictions
    (but which are not ensembles).
    """

    def __init__(self, pred_sources_func: Optional[Callable] = None):
        """
        Initialize a multi-class prediction model.

        Parameters
        ----------
        pred_sources_func: Optional[Callable]
            Function that return the models that are going to produce n different predictions. In the case of BBB or a
            MCDropout model, this just return the model instance n times. In the case of an ensemble, this returns
            all the models of the ensemble.
        """
        self.pred_sources_func = (
            (lambda n_samples: [self.model] * n_samples)
            if pred_sources_func is None
            else pred_sources_func
        )

    def predict_proba(self, X_test: torch.Tensor, n_samples: int = 50) -> torch.Tensor:
        """
        Predict the probabilities for a batch of samples.

        Parameters
        ----------
        X_test: torch.Tensor
            Batch of samples as torch tensor.
        n_samples: Optional[int]
            Number of forward passes in the case of MC Dropout.

        Returns
        -------
        torch.Tensor
            Predictions for every sample.
        """
        X_test_tensor = X_test.float()

        if n_samples:
            # perform multiple forward passes with dropout activated.
            predictions = self._predict_n_times(X_test_tensor, n_samples)
            predictions = torch.stack(predictions, dim=0)
            predictions = torch.mean(predictions, dim=0)

        else:
            predictions = torch.sigmoid(self.model(X_test_tensor)).squeeze()

        return torch.stack([1 - predictions, predictions], dim=1)

    def get_var(self, X_test: torch.Tensor, n_samples: int = 10) -> torch.Tensor:
        """
        Predict standard deviation between predictions.

        Parameters
        ----------
        X_test: torch.Tensor
            Batch of samples as torch tensor.
        n_samples: int
            Number of forward passes.

        Returns
        -------
        torch.Tensor
            Predictions for every sample.
        """
        X_test_tensor = X_test.float()

        predictions = self._predict_n_times(X_test_tensor, n_samples)
        predictions = torch.stack(predictions, dim=2)
        predictions = torch.stack([1 - predictions, predictions], dim=3)

        return torch.mean(torch.var(predictions, dim=2), dim=2)

    def get_mutual_information(
        self, X_test: torch.Tensor, n_samples: int = 10
    ) -> torch.Tensor:
        """
        Compute the mutual information for over multiple predictions based on the approximation of [1] (eq. 7 / 8).

        [1] https://arxiv.org/pdf/1803.08533.pdf

        Parameters
        ----------
        X_test: torch.Tensor
            Batch of samples as torch tensor.
        n_samples: int
            Number of forward passes.

        Returns
        -------
        torch.Tensor
            Approximate mutual information.
        """
        X_test_tensor = X_test.float()

        predictions = self._predict_n_times(X_test_tensor, n_samples)
        predictions = torch.stack(predictions, dim=2)
        predictions = torch.stack([1 - predictions, predictions], dim=3)

        return entropy(predictions.mean(dim=3), dim=2) - entropy(
            predictions, dim=3
        ).mean(dim=2)

    def _predict_n_times(self, X: torch.Tensor, n: int) -> List[torch.Tensor]:
        """
        Make predictions based on n forward passes.

        Parameters
        ----------
        X: torch.Tensor
            Input.
        n: int
            Number of forward passes.
        """
        predictions = []

        for model in self.pred_sources_func(n):
            predictions.append(torch.sigmoid(model(X)).squeeze())

        return predictions


class MCDropoutMLP(MultiplePredictionsMixin, MLP):
    """
    Class for a MLP using MC Dropout.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        MLP.__init__(self, *args, **kwargs)

    def eval(self):
        """
        Ensure that dropout is still being used even if model is in eval mode.
        """
        for m in self.model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()
