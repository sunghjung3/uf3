from typing import Any, Dict, List

import numpy as np
import torch

from uf3.regression import least_squares
from uf3.regression import regularize


def alchemy_als(X: np.ndarray,
                Y: np.ndarray,
                C_init: np.ndarray,
                W_init: np.ndarray,
                C_reg_matrix: np.ndarray,
                n_iter: int,
                normalize: bool = True,
                ):
    """
    Alternating least squares (block coordinate descent) for the toy alchemical
    model:

    Y_hat = np.sum((X @ C) * W, axis=(1, 2)).reshape(-1, 1)

    Args:
        X: np.ndarray
            The input data matrix of shape (n_data, n_ituples, n_basis)
        Y: np.ndarray
            The target data matrix of shape (n_data, 1)
        C_init: np.ndarray
            The initial value of the coefficient matrix of shape
            (n_basis, n_pseudo)
        W_init: np.ndarray
            The initial value of the weight matrix of shape (n_ituples, n_pseudo)
        C_reg_matrix: np.ndarray
            The regularization matrix for the coefficient matrix of shape
            (n_reg_C, n_basis, n_pseudo)
        n_iter: int
            The number of iterations to run the ALS algorithm
        normalize: bool
            Whether to normalize each column of C and W by the maximum absolute
            value of the W column.
        
    Returns:
        Cs: List[np.ndarray]
            List of coefficient matrices of shape (n_basis, n_pseudo) from each
            half iteration. In each iteration, it is saved after fitting C,
            after fitting W, and after normalization. Length of the list is
            3*n_iter+1 (including the initial value).
        Ws: List[np.ndarray]
            List of weight matrices of shape (n_ituples, n_pseudo) from each
            half iteration. In each iteration, it is saved after fitting C,
            after fitting W, and after normalization. Length of the list is
            3*n_iter+1 (including the initial value).
    """
    # sanity check
    n_data, n_ituples, n_basis = X.shape
    assert Y.shape[0] == n_data and Y.shape[1] == 1
    assert C_init.shape[0] == n_basis
    n_pseudo = C_init.shape[1]
    assert W_init.shape[0] == n_ituples and W_init.shape[1] == n_pseudo
    assert C_reg_matrix.shape[1] == n_basis and C_reg_matrix.shape[2] == n_pseudo
    n_reg_C = C_reg_matrix.shape[0]

    # initialize
    C = C_init.copy()
    W = W_init.copy()
    Cs = [C.copy()]
    Ws = [W.copy()]

    # run ALS
    for iter in range(n_iter):
        print(f'Iteration {iter+1}/{n_iter}')

        # update C
        WX = least_squares.broad_row_krp_sum(W, X)
        gram = WX.T @ WX
        ordinate = WX.T @ Y
        regularizer = C_reg_matrix.T.reshape(n_reg_C, n_pseudo * n_basis)
        gram += regularizer.T @ regularizer
        C = np.linalg.solve(gram, ordinate).reshape(n_pseudo, n_basis).T
        Cs.append(C.copy())
        Ws.append(W.copy())

        # update W
        XC = (X @ C).reshape(n_data, n_ituples * n_pseudo)
        gram = XC.T @ XC
        ordinate = XC.T @ Y
        W = np.linalg.solve(gram, ordinate).reshape(n_ituples, n_pseudo)
        Cs.append(C.copy())
        Ws.append(W.copy())

        # normalize
        if normalize:
            normalization_factor = np.max(np.abs(W), axis=0)
            W /= normalization_factor
            C *= normalization_factor
        Cs.append(C.copy())
        Ws.append(W.copy())

    return Cs, Ws


def alchemy_torch(X: np.ndarray,
                  Y: np.ndarray,
                  C_init: np.ndarray,
                  W_init: np.ndarray,
                  C_reg_matrix: np.ndarray,
                  n_epoch: int,
                  optimizer_class: torch.optim.Optimizer,
                  optimizer_kwargs: Dict[str, Any],
                  freeze_params: str = None,
                  ):
    """
    PyTorch implementation of the alchemical model.

    Args:
        X: np.ndarray
            The input data matrix of shape (n_data, n_ituples, n_basis)
        Y: np.ndarray
            The target data matrix of shape (n_data, 1)
        C_init: np.ndarray
            The initial value of the coefficient matrix of shape
            (n_basis, n_pseudo)
        W_init: np.ndarray
            The initial value of the weight matrix of shape (n_ituples, n_pseudo)
        C_reg_matrix: np.ndarray
            The regularization matrix for the coefficient matrix of shape
            (n_reg_C, n_basis, n_pseudo)
        n_epoch: int
            The number of epochs to train the model
        optimizer_class: torch.optim.Optimizer
            The optimizer class to use for training
        optimizer_kwargs: Dict[str, Any]
            The keyword arguments to pass to the optimizer class
        freeze_params: str
            Parameters to freeze during training (e.g., 'C' or 'W')

    Returns:
        Cs: List[np.ndarray]
            List of coefficient matrices of shape (n_basis, n_pseudo) from each
            iteration. Length of the list is n_iter+1 (including the initial value).
        Ws: List[np.ndarray]
            List of weight matrices of shape (n_ituples, n_pseudo) from each
            iteration. Length of the list is n_iter+1 (including the initial value).
    """
    # sanity check
    n_data, n_ituples, n_basis = X.shape
    assert Y.shape[0] == n_data and Y.shape[1] == 1
    assert C_init.shape[0] == n_basis
    n_pseudo = C_init.shape[1]
    assert W_init.shape[0] == n_ituples and W_init.shape[1] == n_pseudo
    assert C_reg_matrix.shape[1] == n_basis and C_reg_matrix.shape[2] == n_pseudo
    n_reg_C = C_reg_matrix.shape[0]

    # initialize
    X = torch.tensor(X, dtype=torch.float64, requires_grad=False)
    Y = torch.tensor(Y, dtype=torch.float64, requires_grad=False)
    train_C = False if freeze_params == 'C' else True
    train_W = False if freeze_params == 'W' else True
    C = torch.tensor(C_init, dtype=torch.float64, requires_grad=train_C)
    W = torch.tensor(W_init, dtype=torch.float64, requires_grad=train_W)
    C_reg_matrix = torch.tensor(C_reg_matrix, dtype=torch.float64, requires_grad=False)
    optimizer = optimizer_class([C, W], **optimizer_kwargs)
    Cs = [C.clone().detach().numpy()]
    Ws = [W.clone().detach().numpy()]

    # run training
    for epoch in range(n_epoch):
        print(f'Epoch {epoch+1}/{n_epoch}')

        def closure():
            optimizer.zero_grad()
            loss = total_loss(X, Y, C, W, C_reg_matrix)
            loss.backward()
            return loss
        optimizer.step(closure)

        # save
        Cs.append(C.clone().detach().numpy())
        Ws.append(W.clone().detach().numpy())

    return Cs, Ws

def forward(X: np.ndarray | torch.Tensor,
            C: np.ndarray | torch.Tensor,
            W: np.ndarray | torch.Tensor,
            ):
    """
    Forward pass of the alchemical model.
    """
    return ((X @ C) * W).sum((1, 2)).reshape(-1, 1)  # the (1, 2) is the axis/dim

def calc_sse(Y: np.ndarray | torch.Tensor,
             Y_hat: np.ndarray | torch.Tensor,
             ):
    """
    Calculate the sum of squared errors between the target and the prediction.
    """
    return ((Y - Y_hat) ** 2).sum()

def reg_loss(C: np.ndarray | torch.Tensor,
             C_reg_matrix: np.ndarray | torch.Tensor,
             ):
    """
    Calculate the regularization loss for the coefficient matrix.
    """
    return ((C_reg_matrix * C).sum((1, 2)) ** 2).sum()

def total_loss(X: np.ndarray | torch.Tensor,
               Y: np.ndarray | torch.Tensor,
               C: np.ndarray | torch.Tensor,
               W: np.ndarray | torch.Tensor,
               C_reg_matrix: np.ndarray | torch.Tensor,
               ):
    """
    Calculate the total loss of the alchemical model.
    """
    Y_hat = forward(X, C, W)
    mse = calc_sse(Y, Y_hat)
    reg = reg_loss(C, C_reg_matrix)
    return mse + reg

def loss_grad(X: np.ndarray | torch.Tensor,
              Y: np.ndarray | torch.Tensor,
              C: np.ndarray | torch.Tensor,
              W: np.ndarray | torch.Tensor,
              C_reg_matrix: np.ndarray | torch.Tensor,
              ):
    """
    Calculate the loss and the gradient of the loss w.r.t. the coefficient matrix
    and the weight matrix.
    """
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float64, requires_grad=False)
        Y = torch.tensor(Y, dtype=torch.float64, requires_grad=False)
        C = torch.tensor(C, dtype=torch.float64, requires_grad=True)
        W = torch.tensor(W, dtype=torch.float64, requires_grad=True)
        C_reg_matrix = torch.tensor(C_reg_matrix, dtype=torch.float64, requires_grad=False)

    loss = total_loss(X, Y, C, W, C_reg_matrix)
    loss.backward()

    loss = loss.clone().detach().numpy()
    grad_C = C.grad.clone().detach().numpy()
    grad_W = W.grad.clone().detach().numpy()
    return loss, grad_C, grad_W

def get_C_reg_matrix(n_pseudo: int,
                     n_basis: int,
                     ridge: float,
                     curvature: float,
                     ):
    """
    Get the regularization matrix for the coefficient matrix.
    """
    matrices = []
    for _ in range(n_pseudo):
        matrix = regularize.get_ridge_penalty_matrix(n_basis)
        matrix *= np.sqrt(ridge)
        if curvature > 0:
            matrix_c = regularize.get_curvature_penalty_matrix_1D(n_basis)
            matrix_c *= np.sqrt(curvature)
            matrix = np.vstack((matrix, matrix_c))
        matrices.append(matrix)
    combined_matrix = regularize.combine_regularizer_matrices(matrices)
    n_reg = combined_matrix.shape[0]
    combined_matrix = combined_matrix.reshape(n_reg, n_pseudo, n_basis).transpose(0, 2, 1)
    return combined_matrix