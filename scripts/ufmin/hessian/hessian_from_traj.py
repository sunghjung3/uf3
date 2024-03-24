from ase.io import trajectory

from hessian import get_hessian

from uf3.regression import least_squares
from uf3.forcefield import calculator

import numpy as np

import os, sys

traj = trajectory.Trajectory("truemin.traj")
model = least_squares.WeightedLinearModel.from_json("/home/sung/UFMin/sung/representability_test/fit_2b/true_model_2b.json")
hessian_calc = calculator.UFCalculator(model)
hessian_file = "hessian.npz"

# check if hessian file exists
if os.path.exists(hessian_file):
    sys.exit(f"{hessian_file} already exists. Exiting...")    

hessians_dict = {}
for i, image in enumerate(traj):
    print(f"Calculating hessian for image {i}")
    image.set_calculator(hessian_calc)
    hessian = get_hessian(image, delta=0.001, nprocs=32)
    hessians_dict[f'hessian_{i}'] = hessian

np.savez(hessian_file, **hessians_dict)
