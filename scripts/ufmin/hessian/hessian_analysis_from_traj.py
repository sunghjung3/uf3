from ase.io import trajectory

import numpy as np
import matplotlib.pyplot as plt

import time, os

import cv2  # movies

from uf3.regression import least_squares
from uf3.forcefield import calculator

import hessian


traj = trajectory.Trajectory("truemin.traj")
true_model = least_squares.WeightedLinearModel.from_json("/home/sung/UFMin/sung/representability_test/fit_2b/true_model_2b.json")
true_calc = calculator.UFCalculator(true_model)
hessians = np.load("hessian.npz")

convergence_fmax = 0.01
eigval_print_num = 10


### ===========================================================================
# Plots:


#   +------------------------------------------------------------------------+
#   |                                    |                                   |
#   |                                    |                                   |
#   |                                    |                                   |
#   |                                    |                                   |
#   |                                    |                                   |
#   |           Hessian Matrix           |        Sorted Eigenvalue          |
#   |                                    |                                   |
#   |                                    |                                   |
#   |                                    |                                   |
#   |                                    |                                   |
#   |                                    |                                   |
#   |                                    |                                   |
#   |                                    |                                   |
#   +------------------------------------+-----------------------------------+
#   |                                    |                                   |
#   |                                    |                                   |
#   |                                    |                                   |
#   |                                    |                                   |
#   |               fmax                 |            eigenvectors           |
#   |        (true, predicted min        |                                   |
#   |                                    |                                   |
#   |                                    |                                   |
#   |                                    |                                   |
#   |                                    |                                   |
#   |                                    |                                   |
#   |                                    |                                   |
#   |                                    |                                   |
#   +------------------------------------------------------------------------+

### ===========================================================================


# first round through traj to get data for plotting
fmax_array = np.zeros(len(traj))
max_fmax = 0
predicted_min_fmax_array = np.zeros(len(traj))
eigvals_list = []
eigvecs_list = []
max_hessian_value = 0
min_hessian_value = np.inf
max_eigenvalue = 0
min_eigenvalue = np.inf
for i, image in enumerate(traj):
    forces = image.get_forces()
    fmax_array[i] = np.sqrt(np.max(np.sum(forces**2, axis=1)))
    if fmax_array[i] > max_fmax:
        max_fmax = fmax_array[i]

    hessian_mat = hessians[f'hessian_{i}']
    if np.max(hessian_mat) > max_hessian_value:
        max_hessian_value = np.max(hessian_mat)
    if np.min(hessian_mat) < min_hessian_value:
        min_hessian_value = np.min(hessian_mat)
    predicted_min = hessian.quad_approx_min(hessian_mat, image.get_forces(), image.get_positions())
    atoms = image.copy()
    atoms.set_calculator(true_calc)
    atoms.set_positions(predicted_min)
    predicted_min_forces = atoms.get_forces()
    predicted_min_fmax = np.sqrt(np.max(np.sum(predicted_min_forces**2, axis=1)))
    predicted_min_fmax_array[i] = predicted_min_fmax
    print(f"Image {i} predicted minimum forces: {predicted_min_fmax}")

    # eigenvalues from largest to smallest
    eigvals, eigvecs = np.linalg.eigh(hessian_mat)
    eigidx = eigvals.argsort()[::-1]
    eigvals = eigvals[eigidx]
    eigvecs = eigvecs[:, eigidx]
    if eigvals[0] > max_eigenvalue:
        max_eigenvalue = eigvals[0]
    if eigvals[-1] < min_eigenvalue:
        min_eigenvalue = eigvals[-1]
    eigvals_list.append(eigvals)
    eigvecs_list.append(eigvecs)

# plotting
hessian_plot_frames = []
for i, image in enumerate(traj):
    print(f"Plotting hessian for image {i}")
    hessiam_mat = hessians[f'hessian_{i}']

    # plot: (0, 0) - hessian matrix, (0, 1) - eigenvalues, (1, 0) - fmax, (1, 1): eigenvectors
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    hess_ax = ax[0, 0].matshow(hessian_mat, cmap='bwr',
                               vmin=min_hessian_value,
                               vmax=max_hessian_value,
                               )
    fig.colorbar(hess_ax)
    ax[0, 0].set_title(f'hessian_{i}')
    ax[0, 1].plot(eigvals_list[i], 'o')
    ax[0, 1].plot(np.where(eigvals_list[i] >= 0, eigvals_list[i], np.nan), 'bo')
    ax[0, 1].plot(np.where(eigvals_list[i] < 0, eigvals_list[i], np.nan), 'ro')
    ax[0, 1].axhline(y=0.0, color='k', linestyle='--')
    ax[0, 1].set_title('Eigenvalues')
    ax[0, 1].set_xlabel('Eigenvalue number')
    ax[0, 1].set_ylabel('Eigenvalue')
    ax[0, 1].set_ylim(min_eigenvalue, max_eigenvalue)
    extreme_eigvals = ''
    for j in range(0, eigval_print_num):
        extreme_eigvals += str(eigvals_list[i][j]) + '\n'
    extreme_eigvals += '...\n'
    for j in range(-eigval_print_num, 0):
        extreme_eigvals += str(eigvals_list[i][j]) + '\n'
    ax[0, 1].text(0.95, 0.95, extreme_eigvals, fontsize=8, color='black', ha='right', va='top', transform=ax[0, 1].transAxes)
    ax[1, 0].plot(fmax_array, 'r-', label='fmax true')
    ax[1, 0].plot(i, fmax_array[i], 'ro')
    ax[1, 0].plot(predicted_min_fmax_array, 'y-', label='fmax at min predicted by hessian')
    ax[1, 0].plot(i, predicted_min_fmax_array[i], 'yo')
    ax[1, 0].axhline(y=convergence_fmax, color='r', linestyle='--')
    ax[1, 0].set_title('fmax')
    ax[1, 0].set_xlabel('forcecall number')
    ax[1, 0].set_ylabel('fmax (eV/A)')
    ax[1, 0].set_yscale('log')
    bottom, top = ax[1, 0].get_ylim()
    ax[1, 0].set_ylim(bottom=bottom, top=max_fmax * 1.2)
    ax[1, 0].legend(loc='best')
    eigvec_ax = ax[1, 1].matshow(np.abs(eigvecs_list[i]), cmap='coolwarm', vmin=0, vmax=1)
    fig.colorbar(eigvec_ax)
    ax[1, 1].set_title('Eigenvectors (column)')
    tmp_filename = str(time.time_ns()) + '.png'
    fig.savefig(tmp_filename)
    plt.close(fig)
    hessian_plot_frames.append( cv2.imread(tmp_filename) )
    os.remove(tmp_filename)

# make movie
height, width, layers = hessian_plot_frames[-1].shape
fps = 5
movie = cv2.VideoWriter("hessian_plot.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
for frame in hessian_plot_frames: movie.write(frame)
cv2.destroyAllWindows()
movie.release()
traj.close()