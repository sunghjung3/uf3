import numpy as np
import matplotlib.pyplot as plt

import warnings
from concurrent.futures import ProcessPoolExecutor


### ===========================================================================
### Hessian calculation ###

def _write_hessian_row(atoms, delta, i, j):
    """
    Calculate a row of the hessian matrix corresponding to the i-th atom and
    j-th coordinate.

    Parameters
    ----------
    atoms : ase.Atoms
        atoms object with defined calculator
    delta : float
        step size (in Angstroms) for numeric differentiation
    i : int
        index of the atom
    j : int
        index of the coordinate (0, 1, or 2)

    Returns
    -------
    row_index: int
        index of the row in the hessian matrix
    row: numpy array
        row of the hessian matrix
    """
    atoms1 = atoms.copy()
    atoms1[i].position[j] += (delta/2)
    atoms1.set_calculator(atoms.get_calculator())
    forces_forward = atoms1.get_forces()
    atoms1[i].position[j] -= delta
    forces_backward = atoms1.get_forces()
    return (i*3 + j), (-1 * (forces_forward - forces_backward).flatten() / delta)


def get_hessian(atoms, delta=0.005, symm_tol=1e-3, nprocs=4):
    """
    Calculate hessian using central difference formula.

    Parameters
    ----------
    atoms : ase.Atoms
        atoms object with defined calculator
    delta : float
        step size (in Angstroms) for numeric differentiation
    symm_tol : float
        tolerance for checking symmetry of the hessian. If the maximum
        elementwise absolute difference between the hessian and its transpose
        is greater than this value, a warning is printed.

    Returns
    -------
    numpy square symmetric array
    """
    H = np.zeros((len(atoms) * 3, len(atoms) * 3), dtype=np.float64)
    with ProcessPoolExecutor(max_workers=nprocs) as executor:
        futures = [executor.submit(_write_hessian_row, atoms, delta, i, j)
                   for i in range(len(atoms)) for j in range(3)]
        # Wait for all tasks to complete
        for future in futures:
            row_index, row = future.result()
            H[row_index] = row
        
    # Check symmetry
    symm_diff = np.max(np.abs(H - H.T))
    if symm_diff > symm_tol:
        warnings.warn(f"Warning: Hessian is not symmetric. Maximum elementwise "
                      f"difference between H and H.T is {symm_diff}.")
    return 0.5 * (H + H.T)

### ===========================================================================

### Calculations ###

def quad_approx_min(H, F, x0):
    """
    Quadratic approximation of the minimum given the Hessian and the force at
    a given point.
    
    Parameters
    ----------
    H : numpy array
        Hessian matrix at x0 (symmetric, square, and 3N x 3N where N is the
        number of atoms in the system)
    F : numpy array
        Force vector evaluated at x0 (N x 3)
    x0 : numpy array
        Initial position (N x 3)
    
    Returns
    -------
    numpy array
        minimum of the quadratic approximation (N x 3)
    """
    H_inv = np.linalg.pinv(H)
    dx = H_inv @ F.flatten()
    return x0 + dx.reshape(-1, 3)

### ===========================================================================
