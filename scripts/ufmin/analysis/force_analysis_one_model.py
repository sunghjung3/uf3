from ase.io import trajectory

import numpy as np
import matplotlib.pyplot as plt

from uf3.regression import least_squares
from uf3.forcefield import calculator

import os, pickle
import concurrent.futures


results_dir = "."
init_traj_file = "../../../initial_data.traj"
opt_traj_file = "../../../ufmin.traj"
model_file = "model.json"


init_traj = trajectory.Trajectory( os.path.join(results_dir, init_traj_file), 'r' )
nimages_start = len(init_traj)
print(f"{nimages_start} images to start")

init_true_energies = list()
init_true_forces = list()
for image in init_traj:
    init_true_energies.append(image.get_potential_energy())
    init_true_forces.append(image.get_forces())

opt_traj = trajectory.Trajectory( os.path.join(results_dir, opt_traj_file), 'r' )
nimages = len(opt_traj)
print(f"{nimages} forcecalls")

true_energies = list()
true_forces = list()
for image in opt_traj:
    true_energies.append(image.get_potential_energy())
    true_forces.append(image.get_forces())

nmodels = nimages
forcecalls = np.arange(-nimages_start, nimages)

# model evaluation
model = least_squares.WeightedLinearModel.from_json( os.path.join(results_dir, model_file) )
model_calc = calculator.UFCalculator(model)
nprocs = 32
def model_eval(image):
    image.calc = model_calc
    return image.get_potential_energy(), image.get_forces()

init_model_energies = list()
init_model_forces = list()
print("Evaluating initial traj")
with concurrent.futures.ProcessPoolExecutor(max_workers=nprocs) as executor:
    for energy, forces in executor.map(model_eval, init_traj):
        init_model_energies.append(energy)
        init_model_forces.append(forces)
model_energies = list()
model_forces = list()
print("Evaluating ufmin.traj")
with concurrent.futures.ProcessPoolExecutor(max_workers=nprocs) as executor:
    for energy, forces in executor.map(model_eval, opt_traj):
        model_energies.append(energy)
        model_forces.append(forces)
print("Finished evaluations")

nAtoms = len(opt_traj[-1])
init_traj.close()
opt_traj.close()

#=============================================================
force_comp_errors = np.zeros((nimages_start + nimages, nAtoms*3))
for i in range(0, nimages_start):
    force_comp_error = np.abs(init_model_forces[i] - init_true_forces[i])
    force_comp_errors[i] = force_comp_error.flatten()
for i in range(0, nimages):
    force_comp_error = np.abs(model_forces[i] - true_forces[i])
    force_comp_errors[i+nimages_start] = force_comp_error.flatten()


force_comp_error_fig, force_comp_error_ax = plt.subplots()
force_comp_error_ax.plot(forcecalls, force_comp_errors)

force_comp_error_ax.set_xlabel("Image number")
force_comp_error_ax.set_ylabel("Force component error (eV/A)")
force_comp_error_ax.set_yscale("log")
#force_comp_error_ax.set_ylim(bottom=ufmin_true_fmax/20)
force_comp_error_ax.yaxis.set_label_position("right")
force_comp_error_ax.yaxis.tick_right()
force_comp_error_ax.legend(loc="lower left")

plt.show()
