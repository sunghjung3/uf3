from ase.io import trajectory

import numpy as np
import matplotlib.pyplot as plt

import os, pickle


results_dir = "."
init_traj_file = "initial_data.traj"
opt_traj_file = "ufmin.traj"
model_traj_file = "ufmin_model.traj"
model_calc_file = "model_calc.pckl"


init_traj = trajectory.Trajectory( os.path.join(results_dir, init_traj_file), 'r' )
nimages_start = len(init_traj)
print(f"{nimages_start} images to start")

init_energies = list()
init_forces = list()
for image in init_traj:
    init_energies.append(image.get_potential_energy())
    init_forces.append(image.get_forces())

opt_traj = trajectory.Trajectory( os.path.join(results_dir, opt_traj_file), 'r' )
nimages = len(opt_traj)
print(f"{nimages} forcecalls")

true_energies = list()
true_forces = list()
for image in opt_traj:
    true_energies.append(image.get_potential_energy())
    true_forces.append(image.get_forces())

with open( os.path.join(results_dir, model_calc_file), 'rb' ) as f:
    model_calc_E = list()
    model_calc_F = list()
    while True:
        try:
            Es, Fs = pickle.load(f)
        except EOFError:
            break
        model_calc_E.append(Es)
        model_calc_F.append(Fs)    
print(len(model_calc_E))

nmodels = nimages
forcecalls = np.arange(0, nimages)

#=============================================================
nAtoms = len(opt_traj[-1])
force_comp_errors = np.zeros((nimages, nAtoms*3))
for true_call_step in range(0, nimages):
    i = true_call_step
    force_pair_list = model_calc_F[i]  # list of tuples of pairs (model F, true F)
    force_comp_error = np.abs(force_pair_list[0][1] - force_pair_list[0][0])
    force_comp_errors[i] = force_comp_error.flatten()

# indices of force components with the highest errors at the last 10 percent of the trajectory
avg_window = np.max([nimages // 10, 1])
cumulative_final_errors = np.sum(force_comp_errors[-avg_window:], axis=0)
highest_end_error_idx = np.argsort(-cumulative_final_errors)[0:10]  # only plot force components with highest errors at the end

force_comp_error_fig, force_comp_error_ax = plt.subplots()
for rank, i in enumerate(highest_end_error_idx):
    atom_number = i // 3
    component = 'x' if i % 3 == 0 else 'y' if i % 3 == 1 else 'z'
    label = str(rank) + ": " + str(atom_number) + component
    force_comp_error_ax.plot(forcecalls, force_comp_errors[:, i], label=label)

force_comp_error_ax.set_xlabel("True force call step")
force_comp_error_ax.set_ylabel("Force component error (eV/A)")
force_comp_error_ax.set_yscale("log")
#force_comp_error_ax.set_ylim(bottom=ufmin_true_fmax/20)
force_comp_error_ax.yaxis.set_label_position("right")
force_comp_error_ax.yaxis.tick_right()
force_comp_error_ax.legend(loc="lower left")

plt.show()
