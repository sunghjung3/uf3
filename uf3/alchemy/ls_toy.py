import numpy as np
import torch

from uf3.regression import regularize


def ls_fit(X: np.ndarray,
           Y: np.ndarray,
           reg_matrix: np.ndarray,
           ):
    """
    Fit a least squares model to the data.

    Args:
        X: np.ndarray
            The input data. Shape (n_data, n_features)
        Y: np.ndarray
            The target data. Shape (n_data, 1)
        reg_matrix: np.ndarray
            The regularization matrix. Shape (n_reg, n_features)
    
    Returns:
        beta: np.ndarray
            The coefficient matrix. Shape (n_features, 1)
    """
    # sanity check
    n_data, n_features = X.shape
    assert Y.shape[0] == n_data and Y.shape[1] == 1
    assert reg_matrix.shape[1] == n_features

    # fit
    gram = X.T @ X
    ordinate = X.T @ Y
    regularizer = reg_matrix.T @ reg_matrix
    gram += regularizer
    beta = np.linalg.solve(gram, ordinate)

    return beta

def ls_torch(X: np.ndarray,
             Y: np.ndarray,
             beta_init: np.ndarray,
             reg_matrix: np.ndarray,
             n_epoch: int,
             optimizer_class: torch.optim.Optimizer,
             optimizer_kwargs: dict,
             ):
    """
    Fit a least squares model to the data using PyTorch.

    Args:
        X: np.ndarray
            The input data. Shape (n_data, n_features)
        Y: np.ndarray
            The target data. Shape (n_data, 1)
        beta_init: np.ndarray
            The initial value of the coefficient matrix. Shape (n_features, 1)
        reg_matrix: np.ndarray
            The regularization matrix. Shape (n_reg, n_features)
        n_epoch: int
            The number of epochs to train the model
        optimizer_class: torch.optim.Optimizer
            The optimizer class to use for training
        optimizer_kwargs: dict
            The keyword arguments to pass to the optimizer

    Returns:
        betas: List[np.ndarray]
            The coefficient matrix at each epoch. Shape (n_features, 1)
    """
    # sanity check
    n_data, n_features = X.shape
    assert Y.shape[0] == n_data and Y.shape[1] == 1
    assert reg_matrix.shape[1] == n_features
    assert beta_init.shape[0] == n_features and beta_init.shape[1] == 1

    # initialize
    X = torch.tensor(X, dtype=torch.float64, requires_grad=False)
    Y = torch.tensor(Y, dtype=torch.float64, requires_grad=False)
    beta = torch.tensor(beta_init, dtype=torch.float64, requires_grad=True)
    reg_matrix = torch.tensor(reg_matrix, dtype=torch.float64, requires_grad=False)
    optimizer = optimizer_class([beta], **optimizer_kwargs)
    betas = [beta.clone().detach().numpy()]

    # train
    for epoch in range(n_epoch):
        print(f"Epoch {epoch+1}/{n_epoch}")

        def closure():
            optimizer.zero_grad()
            loss = total_loss(X, Y, beta, reg_matrix)
            loss.backward()
            return loss
        optimizer.step(closure)
        
        # save
        betas.append(beta.clone().detach().numpy())

    return betas


def forward(X: np.ndarray | torch.Tensor,
            beta: np.ndarray | torch.Tensor,
            ):
    """
    Forward pass of the least squares model.
    """
    return X @ beta

def calc_sse(Y: np.ndarray | torch.Tensor,
             Y_hat: np.ndarray | torch.Tensor,
             ):
    """
    Calculate the sum of squared errors between the target and the prediction.
    """
    return ((Y - Y_hat) ** 2).sum()

def reg_loss(beta: np.ndarray | torch.Tensor,
             reg_matrix: np.ndarray | torch.Tensor,
             ):
    """
    Calculate the regularization loss for the coefficients.
    """
    #return ((reg_matrix @ beta) ** 2).sum()  # same as below
    return ((reg_matrix.T * beta).sum((0)) ** 2).sum()

def total_loss(X: np.ndarray | torch.Tensor,
               Y: np.ndarray | torch.Tensor,
               beta: np.ndarray | torch.Tensor,
               reg_matrix: np.ndarray | torch.Tensor,
               ):
    """
    Calculate the total loss of the least squares model.
    """
    Y_hat = forward(X, beta)
    mse = calc_sse(Y, Y_hat)
    reg = reg_loss(beta, reg_matrix)
    return mse + reg

def loss_grad(X: np.ndarray | torch.Tensor,
              Y: np.ndarray | torch.Tensor,
              beta: np.ndarray | torch.Tensor,
              reg_matrix: np.ndarray | torch.Tensor,
              ):
    """
    Calculate the loss and the gradient of the loss w.r.t. the coefficients.
    """
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float64, requires_grad=False)
        Y = torch.tensor(Y, dtype=torch.float64, requires_grad=False)
        beta = torch.tensor(beta, dtype=torch.float64, requires_grad=True)
        reg_matrix = torch.tensor(reg_matrix, dtype=torch.float64, requires_grad=False)

    loss = total_loss(X, Y, beta, reg_matrix)
    loss.backward()

    loss = loss.clone().detach().numpy()
    grad_beta = beta.grad.clone().detach().numpy()
    return loss, grad_beta

def get_reg_matrix(n_ituples: int,
                   n_basis: int,
                   ridge: float,
                   curvature: float,
                   ):
    """
    Get the regularization matrix for the coefficients.
    """
    matrices = []
    for _ in range(n_ituples):
        matrix = regularize.get_ridge_penalty_matrix(n_basis)
        matrix *= np.sqrt(ridge)
        if curvature > 0:
            matrix_c = regularize.get_curvature_penalty_matrix_1D(n_basis)
            matrix_c *= np.sqrt(curvature)
            matrix = np.vstack((matrix, matrix_c))
        matrices.append(matrix)
    combined_matrix = regularize.combine_regularizer_matrices(matrices)
    return combined_matrix