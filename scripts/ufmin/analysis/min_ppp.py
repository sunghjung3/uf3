import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
#import imageio  # gifs
import cv2  # movies

import math

from ase.io import read
from ase.io.trajectory import Trajectory
from ase.atoms import Atoms

import functools, copy, time, os, pickle, warnings

from uf3.forcefield import calculator
from uf3.regression import least_squares
from uf3.util import cubehelix

from libufmin_analysis import plot_pair_energy


def lj(r_min, depth, r):
    sigma = r_min*(2 ** (-1/6))
    a = (sigma / r) ** 6
    return 4*depth * ( a*a - a )


def opt_pair_pot_frames(radial_potential, traj, model=None, training_traj=None, label='', xlim=None, ylim=None, combine=True,
                        show_last_frame=False, reslim=(-0.1, 0.1)):

    frames = list()
    if xlim is None:
        xlim = [1.7, 5]
    if ylim is None:
        ylim = [-10, 15]
    if ylim[0] > 0:
        knot_plot_y = ylim[0]  # y value to plot knot tick marks
    elif ylim[1] < 0:
        knot_plot_y = ylim[1]
    else:
        knot_plot_y = 0

    xs = np.linspace(*xlim, 200)
    nAtoms = len(traj[0])


    print("Processing data...")
    if model is not None:  # data for model
        if isinstance(model, str):
            model = least_squares.WeightedLinearModel.from_json(model)
        if len(model.bspline_config.element_list) > 1:
            raise Exception("Only 1 element system supported in this script.")
        if model.bspline_config.degree > 2:
            raise Exception("Only up to 2 body interactions supported by this scirpt")
        solutions = least_squares.arrange_coefficients(model.coefficients, model.bspline_config)
        element = model.bspline_config.element_list[0]
        pair = (element, element)
        coefficient_1b = solutions[element]
        coefficients_2b = solutions[pair]
        knot_sequence = model.bspline_config.knots_map[pair]
        pair_model = copy.deepcopy(model)
        pair_model.coefficients[0] /= (nAtoms * (nAtoms-1) / 2)  # 1 body offset corrected to 2 atoms
        model_calc = calculator.UFCalculator(pair_model)
        model_calc_atoms = Atoms(element * model_calc.bspline_config.degree)
        model_calc_atoms.positions = np.zeros((len(model_calc_atoms), 3))
        model_calc_atoms.calc = model_calc

        if training_traj is not None:
            training_rs = np.empty(0)
            for training_image in training_traj:
                for i in range(len(training_image)-1):
                    d = training_image[i].position - training_image.positions[i+1:]
                    d = np.sqrt(np.sum(np.square(d), axis=1))
                    training_rs = np.concatenate( (training_rs, d) )
            #training_energies = list()  # slower than below
            #for r in training_rs:
            #    model_calc_atoms[1].position[2] = r
            #    training_energies.append(model_calc_atoms.get_potential_energy())
            #training_energies = calc_pair_energy(training_rs, coefficient_1b, coefficients_2b, knot_sequence)
            training_energies = radial_potential(training_rs)


    indices = list(range(len(traj[0])))
    pairs = [(a, b) for idx, a in enumerate(indices) for b in indices[idx + 1:]]
    n_pairs = len(pairs)

    # plot potential energy curves in the background
    print("Plotting energy curves...")
    if combine:
        plot_dim = 1
    else:
        plot_dim = math.ceil(n_pairs ** 0.5)

    orig_fig, orig_axs = plt.subplots(nrows=plot_dim, ncols=plot_dim, figsize=(10*plot_dim, 5*plot_dim))
    for idx, pair in enumerate(pairs):
        plot_coord = [ idx//plot_dim, idx%plot_dim ]
        try:
            ax = orig_axs[plot_coord[0], plot_coord[1]]
        except IndexError as ie:
            ax = orig_axs[idx]
        except TypeError as te:
            ax = orig_axs
        ax.plot(xs, radial_potential(xs), 'm')
        if model is not None:
            #orig_fig, ax = plotting.visualize_splines(2 * coefficients, knot_sequence, ax=ax)
            #plt.vlines([0.0], -100, 100, color="red", linewidth=2)
            orig_fig, ax = plot_pair_energy(coefficient_1b, coefficients_2b, knot_sequence, nAtoms, ax=ax, xlim=xlim, ylim=ylim)
            ax.plot(knot_sequence, np.ones(len(knot_sequence)) * knot_plot_y, marker="|", color='k')
            if training_traj is not None:
                ax.plot(training_rs, training_energies, 'mx')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if combine:
            break  # only need to execute this loop for 1 iteration

    print("Plotting traj r points...")
    prev_r_pairs = [None] * n_pairs
    prev_r_energies = [None] * n_pairs
    traj_length = len(traj)
    residuals = np.empty((n_pairs, 2))  # only used if show_last_frame=True
    for step, image in enumerate(traj):
        if show_last_frame and step != traj_length - 1:
            continue  # skip all but last frame
        fig = copy.deepcopy(orig_fig)
        axs = copy.deepcopy(orig_axs)
        for idx, pair in enumerate(pairs):
            plot_coord = [ idx//plot_dim, idx%plot_dim ]
            try:
                ax = axs[plot_coord[0], plot_coord[1]]
            except IndexError as ie:
                ax = axs[idx]
            except TypeError as te:
                ax = axs
            r_pair = np.sqrt(np.sum(np.square(image[pair[0]].position - image[pair[1]].position)))
            if model is None:
                r_energy = radial_potential(r_pair)  # TODO: change if using model
            else:  # model is provided
                model_calc_atoms[1].position[2] = r_pair
                r_energy = model_calc_atoms.get_potential_energy()
                #r_energy = calc_pair_energy(r_pair, coefficient_1b, coefficients_2b, knot_sequence, nAtoms)  # this is slower than above in this loop layout. If distances are calculated like the training_traj, maybe a different story
                if show_last_frame and step == traj_length - 1:
                    residuals[idx] = [r_pair, radial_potential(r_pair) - r_energy]
            ax.plot(r_pair, r_energy, 'ro')
            if prev_r_pairs[idx] is not None:
                ax.plot(prev_r_pairs[idx], prev_r_energies[idx], 'r.')
            prev_r_pairs[idx] = r_pair
            prev_r_energies[idx] = r_energy
            if step == traj_length - 1:
                if combine:
                    title = label + " " + str(step) + " CONVERGED"
                else:
                    title = label + " " + str(step) + " " + str(pair) + " CONVERGED"
            else:
                if combine:
                    title = label + " " + str(step)
                else:
                    title = label + " " + str(step) + " " + str(pair)
            ax.set_title(title)

        tmp_filename = str(time.time_ns()) + ".png"
        plt.savefig(tmp_filename)
        #frames.append( imageio.v2.imread(tmp_filename) )
        frames.append( cv2.imread(tmp_filename) )
        os.remove(tmp_filename)

    if show_last_frame:
        res_fig, res_ax = plt.subplots()
        res_ax.plot(residuals[:, 0], residuals[:, 1], 'r.')
        res_ax.set_title(f"Residuals for model {label}")
        res_ax.set_xlabel("r")
        res_ax.set_ylabel("Residual (true - pred)")
        res_ax.set_ylim(reslim)
        res_ax.set_xlim(xlim)
        res_ax.axhline(y=0.0, color='k', linestyle='--')  # plot x axis
        plt.show()

        
    plt.close('all')
           
    return frames


def test():
    #atoms = read("1/truemin.traj")
    #traj = [atoms]
    traj = Trajectory("1/truemin.traj", mode='r')
    r_min = 2.22
    well_depth = 9
    lj_p = functools.partial(lj, r_min, well_depth)
    frames = opt_pair_pot_frames(lj_p, traj, xlim=[1.7, 5], ylim=[-10, 15])
    print("Writing movie...")
    #imageio.mimsave("temp.gif", frames, duration=1)
    height, width, layers = frames[-1].shape
    fps = 5
    movie = cv2.VideoWriter("1/min_ppp.avi", 0, fps, (width, height))
    for frame in frames: movie.write(frame)
    cv2.destroyAllWindows()
    movie.release()
    traj.close()


if __name__ == "__main__":
    #test()

    #import sys
    #sys.exit()

    #============================================

    results_dir = "."
    model_file_prefix = "model"
    opt_traj_file = "ufmin.traj"
    model_traj_file = "ufmin_model.traj"
    movie_file = "min_ppp.avi"   # will write at results_dir/movie_file
    plot_xlim = [1.5, 5]
    plot_ylim = [-15, 20]
    combine_subplots = True
    movie_fps = 3  # framerate of movie

    r_min = 2.22
    well_depth = 9
    lj_p = functools.partial(lj, r_min, well_depth)  # the true radial potential

    #============================================

    opt_traj = Trajectory( os.path.join(results_dir, opt_traj_file), 'r' )
    print(len(opt_traj))

    with open( os.path.join(results_dir, model_traj_file), 'rb' ) as f:
        model_traj = list()
        while True:
            try:
                image = pickle.load(f)
            except EOFError:
                break
            model_traj.append(image)
    print(len(model_traj))

    n_models = len(model_traj)
    n_images = len(opt_traj)  # also the number of true force calls
    assert n_models + 2 == n_images


    frames = list()
    for i in range(1, n_models+1):  # for each model
        print("model", i)

        # load model
        model_path = os.path.join(results_dir, model_file_prefix + '_' + str(i) + '.json')
        if not os.path.isfile(model_path):
            raise Exception(f"{model_path} does not exist.")
        model = least_squares.WeightedLinearModel.from_json(model_path)

        min_traj = model_traj[i-1]
        training_traj = opt_traj[0:i+1]
        frames += opt_pair_pot_frames(lj_p, min_traj, model=model, training_traj=training_traj, xlim=plot_xlim, ylim=plot_ylim, label=str(i), combine=combine_subplots)
        frames.append(frames[-1])  # duplicate last frame of each UF3 minimization
        frames.append(frames[-1])  # duplicate again
        frames.append(frames[-1])  # duplicate again

    print("Writing movie...")
    height, width, layers = frames[-1].shape
    movie = cv2.VideoWriter( os.path.join(results_dir, movie_file), 0, movie_fps, (width, height) )
    for frame in frames: movie.write(frame)
    cv2.destroyAllWindows()
    movie.release()

    opt_traj.close()
