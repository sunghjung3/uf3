import os, pickle, time

from ase.io import trajectory, read

import uf3_run
from uf3.data import io
from uf3.regression import least_squares
from uf3.util import plotting

import pandas as pd
import matplotlib.pyplot as plt
import imageio
import numpy as np

#==========================================================

results_dir = "."
live_features_file = "live_features.h5"
model_file_prefix = "model"
init_traj_file = "initial_data.traj"
opt_traj_file = "ufmin.traj"
model_traj_file = "ufmin_model.traj"
model_calc_file = "model_calc.pckl"
spline_2b_gif = "spline_2b.gif"   # will write at results_dir/spline_2b_gif
parity_E_gif = "parity_E.gif"  # will write at results_dir/parity_E_gif
parity_F_gif = "parity_F.gif"  # will write at results_dir/parity_F_gif
opt_plot_png = "opt_plot.png"
opt_plot_detailed_png = "opt_plot_detailed.png"
call_ratio_png = "call_ratio.png"
error_plot_png = "error.png"
model_diff_png = "model_diff.png"
settings_file = "settings.yaml"

#md_features_path = "entire_traj_training/features.h5"
md_features_path = None
truemin_traj_path = "1/truemin.traj"

plot_2b_xlim = [1.5, 6.0]  # for 2 body potential plotting
plot_2b_ylim = [-6, 11]  # for 2 body potential plotting
plotting_pair = ("Pt", "Pt")
gif_fps = 0.2  # framerate of gif
ufmin_true_fmax = 0.01

#=======================================================================================================
def create_2b_potential_frame(model, bspline_config, pair, label, img_filename):
    solutions = least_squares.arrange_coefficients(model.coefficients, bspline_config)
    coefficients = solutions[pair]
    knot_sequence = bspline_config.knots_map[pair]
    fig, ax = plotting.visualize_splines(coefficients, knot_sequence)
    plt.vlines([0.0], -100, 100, color="orange", linewidth=2)
    ax.set_ylim(plot_2b_ylim)
    ax.set_xlim(plot_2b_xlim)
    ax.set_ylabel("Pair energy (eV)")
    ax.set_xlabel("Pair distance (A)")
    fig.suptitle("True force call step " + str(label))
    plt.savefig(img_filename, transparent=False, facecolor='white')
    plt.close()


def create_parity_plot_frame(y_i, p_i, rmse_i, y_all, p_all, rmse_all, y_md=None, p_md=None, rmse_md=None, units="units", title="title", img_filename="parity_plot.png"):
    fig, ax = plt.subplots()
    ax.scatter(p_i, y_i, s=5, c='b', zorder=3)
    ax.scatter(p_all, y_all, s=5, c='r', zorder=2)
    if y_md is not None:
        ax.scatter(p_md, y_md, s=5, c='g', zorder=1)
    ax.axline([0,0], c="k", ls="dashed", slope=1, linewidth=0.5)
    rmse_text = f"Training RMSE = {'%.5f'%(rmse_i)}\nFull traj RMSE = {'%.5f'%(rmse_all)}"
    if rmse_md is not None:
        rmse_text += f"\nMD traj RMSE={'%.5f'%(rmse_md)}"
    ax.text(0.55, 0.05, rmse_text, transform=ax.transAxes)
    ax.set_xlabel(f"Predicted ({units})")
    ax.set_ylabel(f"True ({units})")
    legend_list = ["Training", "Full", "MD", "Ideal"]
    if y_md is None:
        legend_list.remove("MD")
    ax.legend(legend_list)
    fig.suptitle(title)
    plt.savefig(img_filename, transparent=False, facecolor='white')
    plt.close()

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

skip_frames = nmodels // 1000 + 1  # skip frames for gif if too many

forcecalls = np.arange(0, nimages)  # for plotting


bspline_config = uf3_run.initialize( os.path.join(results_dir, settings_file) )

#======================================================================================================
### Compare final structure with true optimization ###
'''
ref_structure = read(truemin_traj_path)
ufmin_final_structure = opt_traj[-1]
max_deviation = np.sqrt(np.max(np.sum( np.square(ref_structure.positions - ufmin_final_structure.positions), axis=1)))
print(f"Max atomic position deviation: {max_deviation}")
'''

#========================================================================================================
### Run through each UF3 model ###
live_features_path = os.path.join(results_dir, live_features_file)
all_features = uf3_run.load_all_features(live_features_path)
n_chunks, n_entries, chunk_names, chunk_lengths = io.analyze_hdf_tables(live_features_path)
live_features = all_features.loc[[]]  # features at stage i. empty to begin with

# load featurized MD trajectory
if md_features_path is not None:
    md_features = uf3_run.load_all_features(md_features_path)
else:
    md_features = None

# initialize lists to hold data
rmse_e_i_list = list()
rmse_f_i_list = list()
rmse_e_all_list = list()
rmse_f_all_list = list()
rmse_e_md_list = list()
rmse_f_md_list = list()
spline_frames_2b = list()
energy_rmse_parity_frames = list()
force_rmse_parity_frames = list()
previous_model_coeffs = None
model_diffs = list()
model_idx = list()

for i in range(0, nmodels):  # for each model
    print("model", i)

    # load model
    model_path = os.path.join(results_dir, model_file_prefix + '_' + str(i) + '.json')
    if not os.path.isfile(model_path):
        raise Exception(f"{model_path} does not exist.")
    model = least_squares.WeightedLinearModel.from_json(model_path)
    
    # load live features dataframe for this model
    if i == 0:  # 1st model (`nimages_start` data points)
        init_keys = [str(i-1) + "_" + str(j) for j in range(nimages_start)]  # "-1_0", "-1_1", ...
        live_features = pd.concat( [live_features, all_features.loc[init_keys]] )
    else:  # not 1st model (new data point)
        live_features = pd.concat( [live_features, all_features.loc[[str(i-1) + "_0"]]] )  # model i takes in data point i-1

    # calculate training error at each stage i
    y_e_i, p_e_i, y_f_i, p_f_i, rmse_e_i, rmse_f_i, mae_e_i, mae_f_i = uf3_run.calculate_errors(model, live_features)
    rmse_e_i_list.append(rmse_e_i)
    rmse_f_i_list.append(rmse_f_i)

    # calculate error of stage i model with respect to the entire ufmin trajectory
    y_e_all, p_e_all, y_f_all, p_f_all, rmse_e_all, rmse_f_all, mae_e_all, mae_f_all = uf3_run.calculate_errors(model, all_features)
    rmse_e_all_list.append(rmse_e_all)
    rmse_f_all_list.append(rmse_f_all)

    # calculate error of stage i model with respect to MD trajectory
    if md_features is not None:
        y_e_md, p_e_md, y_f_md, p_f_md, rmse_e_md, rmse_f_md, mae_e_md, mae_f_md = uf3_run.calculate_errors(model, md_features)
    else:
        y_e_md, p_e_md, y_f_md, p_f_md, rmse_e_md, rmse_f_md, mae_e_md, mae_f_md = None, None, None, None, None, None, None, None
    rmse_e_md_list.append(rmse_e_md)
    rmse_f_md_list.append(rmse_f_md)

    # append to frame list to make gif of interaction potentials
    temp_filename = str(time.time_ns()) + ".png"
    create_2b_potential_frame(model, bspline_config, plotting_pair, i, temp_filename)  # label is true force call number
    spline_frames_2b.append( imageio.v2.imread(temp_filename) )
    os.remove(temp_filename)

    # append to frame list to make gif of parity plots
    temp_filename = str(time.time_ns()) + ".png"
    title = "Energy at true force call step " + str(i)
    create_parity_plot_frame(y_e_i, p_e_i, rmse_e_i, y_e_all, p_e_all, rmse_e_all, y_e_md, p_e_md, rmse_e_md, units="eV", title=title, img_filename=temp_filename)
    energy_rmse_parity_frames.append( imageio.v2.imread(temp_filename) )
    os.remove(temp_filename)
    temp_filename = str(time.time_ns()) + ".png"
    title = "Force at true force call step " + str(i+1)
    create_parity_plot_frame(y_f_i, p_f_i, rmse_f_i, y_f_all, p_f_all, rmse_f_all, y_f_md, p_f_md, rmse_f_md, units="eV/A", title=title, img_filename=temp_filename)
    force_rmse_parity_frames.append( imageio.v2.imread(temp_filename) )
    os.remove(temp_filename)

    # model difference
    if previous_model_coeffs is not None:
        try:
            valid_coeffs = model.coefficients[model.data_coverage]
            valid_previous_model_coeffs = previous_model_coeffs[model.data_coverage]
            not_zero = np.where(valid_coeffs != 0)
            valid_coeffs = valid_coeffs[not_zero]
            valid_previous_model_coeffs = valid_previous_model_coeffs[not_zero]
            model_diff = np.max(np.abs(1  - valid_previous_model_coeffs / valid_coeffs))
            model_diffs.append(model_diff)
            model_idx.append(i)
        except ValueError:  # shapes not compatible for subtraction
            pass
    previous_model_coeffs = model.coefficients


# 2b pair potential gif
spline_frames_2b = spline_frames_2b[::skip_frames]
imageio.mimsave( os.path.join(results_dir, spline_2b_gif), spline_frames_2b, duration=gif_fps )

# parity plot gif
energy_rmse_parity_frames = energy_rmse_parity_frames[::skip_frames]
force_rmse_parity_frames = force_rmse_parity_frames[::skip_frames]
imageio.mimsave( os.path.join(results_dir, parity_E_gif), energy_rmse_parity_frames, duration=gif_fps )
imageio.mimsave( os.path.join(results_dir, parity_F_gif), force_rmse_parity_frames, duration=gif_fps )

# RMSE plot
rmse_fig, (rmse_e_ax, rmse_f_ax) = plt.subplots(2)
rmse_e_ax.plot(forcecalls, rmse_e_i_list, color='b', label="Training error")
rmse_e_ax.plot(forcecalls, rmse_e_all_list, 'r', label="Error wrt full ufmin traj")
if md_features is not None:
    rmse_e_ax.plot(forcecalls, rmse_e_md_list, 'g', label="Error wrt separate MD traj")
#rmse_e_ax.set_xlabel("True force call step")
rmse_e_ax.set_ylabel("Energy RMSE (eV)")
rmse_e_ax.tick_params(axis ='y', which = 'minor')
rmse_e_ax.set_yscale("log")
#rmse_e_ax.legend(loc="upper right")
rmse_f_ax.plot(forcecalls, rmse_f_i_list, color='b', label="Training error")
rmse_f_ax.plot(forcecalls, rmse_f_all_list, 'r', label="Error wrt full ufmin traj")
if md_features is not None:
    rmse_f_ax.plot(forcecalls, rmse_f_md_list, 'g', label="Error wrt separate MD traj")
rmse_f_ax.set_xlabel("True force call step")
rmse_f_ax.set_ylabel("Force RMSE (eV/A)")
rmse_f_ax.tick_params(axis ='y', which = 'minor')
rmse_f_ax.set_yscale("log")
#rmse_f_ax.legend(loc="upper right")
handles, labels = rmse_f_ax.get_legend_handles_labels()
rmse_fig.legend(handles, labels, loc="upper right")
rmse_fig.suptitle("Model Errors")
plt.savefig( os.path.join(results_dir, error_plot_png) )

#========================================================================================================
### True energies and max forces at each true force call step ###
# get true fmax at each true force call step
init_fmax_list = list()
for init_force in init_forces:
    init_force_squared = np.sum( np.square(init_force), axis=1 )
    init_fmax_list.append( np.sqrt(np.max(init_force_squared)) )

true_fmax_list = list()
for true_force in true_forces:
    true_force_squared = np.sum( np.square(true_force), axis=1 )
    true_fmax_list.append( np.sqrt(np.max(true_force_squared)) )

# plot
true_calc_fig, true_E_ax = plt.subplots()
color = 'tab:cyan'
true_E_ax.plot(np.ones(nimages_start) * -1, init_energies, 'c.')
true_E_ax.plot(forcecalls, true_energies, 'c')
true_E_ax.plot([-1, 0], [init_energies[-1], true_energies[0]], 'c')
true_E_ax.set_xlabel("True force call step")
true_E_ax.set_ylabel("Energy (eV)", color = 'c')
true_E_ax.tick_params(axis ='y', labelcolor = 'c', which = 'minor')
#true_E_ax.set_yscale("log")

true_F_ax = true_E_ax.twinx()
color = 'tab:magenta'
true_F_ax.plot(np.ones(nimages_start) * -1, init_fmax_list, 'm.')
true_F_ax.plot(forcecalls, true_fmax_list, 'm')
true_F_ax.plot([-1, 0], [init_fmax_list[-1], true_fmax_list[0]], 'm')
true_F_ax.set_ylabel("Max force (eV/A)", color = 'm')
true_F_ax.tick_params(axis ='y', labelcolor = 'm', which = 'minor')
true_F_ax.set_yscale("log")
true_F_ax.axhline(y=ufmin_true_fmax, color='r', linestyle='--')

true_calc_fig.suptitle("UF3-accelerated Optimization")
plt.savefig( os.path.join(results_dir, opt_plot_png) )
with open( os.path.join(results_dir, opt_plot_png + "_obj.pckl"), 'wb' ) as f:
    pickle.dump([true_calc_fig, true_E_ax, true_F_ax], f)

#======================================================================================================
### True and UF3 energies and max forces at each UF3 force call step ###
# finishing touches on model_calc_E and calculate UF3 fmax
model_calc_fmax_list = list()
for true_call_step in range(0, nimages):
    #if true_call_step == 1 or true_call_step == 2:  # 1st 2 steps (does not use UF3)
    #    i = true_call_step-1
    #    t = [ (true_energies[i], true_energies[i]) ]
    #    model_calc_E.insert(i, t)
    #    t = [ (true_fmax_list[i], true_fmax_list[i]) ]
    #    model_calc_fmax_list.append(t)
    i = true_call_step
    force_pair_list = model_calc_F[i]  # list of tuples of pairs (model F, true F)
    each_step_fmax_list = list()
    for force_pair in force_pair_list:
        uf3_fmax = np.sqrt(np.max(np.sum( np.square(force_pair[0]), axis=1 )))
        true_fmax = np.sqrt(np.max(np.sum( np.square(force_pair[1]), axis=1 )))
        each_step_fmax_list.append( (uf3_fmax, true_fmax) )
    model_calc_fmax_list.append(each_step_fmax_list)
assert len(model_calc_fmax_list) == nimages
assert len(model_calc_E) == nimages

# plot
model_calc_fig, model_calc_E_ax = plt.subplots()
model_calc_fmax_ax = model_calc_E_ax.twinx()
#model_calc_E_ax.set_yscale("log")
model_calc_fmax_ax.set_yscale("log")
model_calc_E_ax.set_xlabel("Force call step (true and UF3)")
model_calc_fmax_ax.set_ylabel("Max force (eV/A)", color='r')
model_calc_E_ax.set_ylabel("Energy (eV)", color='b')
p1, p2, p3, p4 = None, None, None, None  # uninitialized
all_calls_counter = 0
for true_call_step in range(0, nimages):
    i = true_call_step
    e = np.array(model_calc_E[i])
    f = np.array(model_calc_fmax_list[i])
    assert len(e) == len(f)
    call_numbers = np.arange(all_calls_counter, all_calls_counter+len(e))
    all_calls_counter += len(e) - 1  # -1 because of overlap between steps
    p1 = model_calc_fmax_ax.plot(call_numbers, f[:, 0], 'r-', linewidth=0.6, label="UF3 max force")
    p2 = model_calc_fmax_ax.plot(call_numbers, f[:, 1], 'm--', linewidth=0.6, label="True max force")
    p3 = model_calc_E_ax.plot(call_numbers, e[:, 0], 'b-', linewidth=0.6, label="UF3 energy")
    p4 = model_calc_E_ax.plot(call_numbers, e[:, 1], 'c--', linewidth=0.6, label="True energy")
    
    # plot each true call step with bolder markers
    model_calc_fmax_ax.plot(call_numbers[0], f[0, 0], 'r.', markersize=5)
    model_calc_fmax_ax.plot(call_numbers[0], f[0, 1], 'm.', markersize=5)
    model_calc_E_ax.plot(call_numbers[0], e[0, 0], 'b.', markersize=5)
    model_calc_E_ax.plot(call_numbers[0], e[0, 1], 'c.', markersize=5)
    model_calc_fmax_ax.plot(call_numbers[-1], f[-1, 0], 'r.', markersize=5)
    model_calc_fmax_ax.plot(call_numbers[-1], f[-1, 1], 'm.', markersize=5)
    model_calc_E_ax.plot(call_numbers[-1], e[-1, 0], 'b.', markersize=5)
    model_calc_E_ax.plot(call_numbers[-1], e[-1, 1], 'c.', markersize=5)

model_calc_fmax_ax.axhline(y=ufmin_true_fmax, color='r', linestyle='--')   # horizontal line at force convergence
ps = p1 + p2 + p3 + p4
labs = [l.get_label() for l in ps]
model_calc_fmax_ax.legend(ps, labs, loc='upper right')
model_calc_fig.suptitle("UF3-accelerated Optimization (detailed)")
plt.savefig( os.path.join(results_dir, opt_plot_detailed_png) )
with open( os.path.join(results_dir, opt_plot_detailed_png + "_obj.pckl"), 'wb' ) as f:
    pickle.dump([model_calc_fig, model_calc_E_ax, model_calc_fmax_ax], f)
#================================================================================================
# uf3 force call per true force call plot
forcecalls = np.arange(0, nimages)
n_uf3_calls = [len(step)-1 for step in model_calc_E]
call_ratio_fig, call_ratio_ax = plt.subplots()
call_ratio_ax.plot(forcecalls, n_uf3_calls)
call_ratio_ax.set_xlabel("True force call step")
call_ratio_ax.set_ylabel("Number of UF3 force calls")
plt.savefig( os.path.join(results_dir, call_ratio_png) )

#=================================================================================================

# model difference plot
model_diff_fig, model_diff_ax = plt.subplots()
model_diff_ax.plot(model_idx, model_diffs)
model_diff_ax.set_xlabel("Model number")
model_diff_ax.set_ylabel("Model difference (max abs(1 - previous model / current model))")
model_diff_ax.set_yscale("log")
plt.savefig( os.path.join(results_dir, model_diff_png) )

#plt.show()

init_traj.close()
opt_traj.close()
