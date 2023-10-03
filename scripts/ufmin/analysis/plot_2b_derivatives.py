from ase.atoms import Atoms
import ase.data as ase_data
from ase.io.trajectory import Trajectory
import numpy as np
import matplotlib.pyplot as plt 
from scipy import interpolate, optimize

import functools

from uf3.regression import least_squares
from uf3.forcefield import zbl

from libufmin_analysis import calc_pair_energy, plot_pair_energy


def lj(r_min, depth, r):
    sigma = r_min*(2 ** (-1/6))
    a = (sigma / r) ** 6
    return 4*depth * ( a*a - a )

def morse(r_e, D_e, a, r):
    return D_e * (1 - np.exp(-a * (r-r_e)))**2 - D_e


def make_pair_energy_data(traj, radial_potential):
    # get list of all pair distances seen in trajectory and their corresponding pair energies
    pair_distances = []
    pair_energies = []
    for atoms in traj:
        for i in range(len(atoms)-1):
            for j in range(i+1, len(atoms)):
                r = atoms.get_distance(i, j)
                pair_distances.append(r)
                pair_energies.append(radial_potential(r))
    return pair_distances, pair_energies


def plot_single_model(model, training_data=None, nAtoms=None, label='',
                      # calculation parameters
                      calc_rlim=None, rspacing=0.01, epsilon=0.00000001,
                      # plot parameters
                      plot_rlim=None, E_lim=None, E_prime_lim=None, E_double_prime_lim=None, reslim=None,
                      plot_components=False
                      ):
    solutions = least_squares.arrange_coefficients(model.coefficients, model.bspline_config)
    element = model.bspline_config.element_list[0]
    pair = (element, element)
    z1, z2 = (ase_data.atomic_numbers[el] for el in pair)
    coefficient_1b = solutions[element]
    coefficients_2b = solutions[pair]
    knot_sequence = model.bspline_config.knots_map[pair]
    if hasattr(model, 'zbl_scale') and model.zbl_scale:
        zbl_obj = zbl.LJSwitchingZBL(z1, z2, scale=model.zbl_scale)
    else:
        zbl_obj = None

    if plot_components:
        fig, ax = plot_pair_energy(coefficient_1b, coefficients_2b, knot_sequence,
                                   nAtoms, zbl=zbl_obj,
                                   show_components=True, show_total=True,
                                   xlim=plot_rlim, ylim=E_lim)

    # calculate derivative and curvature of pair energy using central finite difference
    if calc_rlim is None:
        calc_rlim = (1.4, 6.0)
    r_min = calc_rlim[0]
    r_max = calc_rlim[1]
    rs = np.arange(r_min, r_max, rspacing)
    energies = []
    derivatives = []
    curvatures = []
    local_minima = {'r': [], 'E': [], 'E_prime': [], 'E_double_prime': []}
    for k, r in enumerate(rs):
        pair_energy_plus = calc_pair_energy(r + epsilon, coefficient_1b, coefficients_2b, knot_sequence, nAtoms, zbl=zbl_obj)
        pair_energy_minus = calc_pair_energy(r - epsilon, coefficient_1b, coefficients_2b, knot_sequence, nAtoms, zbl=zbl_obj)
        pair_energy_prime = (pair_energy_plus - pair_energy_minus) / (2 * epsilon)
        derivatives.append(pair_energy_prime)

        pair_energy_center = calc_pair_energy(r, coefficient_1b, coefficients_2b, knot_sequence, nAtoms, zbl=zbl_obj)
        energies.append(pair_energy_center)

        pair_energy_plusplus = calc_pair_energy(r + 2*epsilon, coefficient_1b, coefficients_2b, knot_sequence, nAtoms, zbl=zbl_obj)
        pair_energy_minusminus = calc_pair_energy(r - 2*epsilon, coefficient_1b, coefficients_2b, knot_sequence, nAtoms, zbl=zbl_obj)
        pair_energy_plus_prime = (pair_energy_plusplus - pair_energy_center) / (2*epsilon)
        pair_energy_minus_prime = (pair_energy_center - pair_energy_minusminus) / (2*epsilon)
        pair_energy_double_prime = (pair_energy_plus_prime - pair_energy_minus_prime) / (2*epsilon)
        curvatures.append(pair_energy_double_prime)

        # find r where local minimum occurs
        if len(derivatives) > 1:
            if len(derivatives) > 2 and derivatives[-2] == 0:
                if derivatives[-3] < 0 and derivatives[-1] > 0:
                    #r_zero = optimize.brentq(calc_pair_energy, rs[k-2], rs[k], args=(coefficient_1b, coefficients_2b, knot_sequence, 2))  # acccurate method
                    r_zero = rs[k-1]  # approximate method
                    local_minima['r'].append(r_zero)
                    local_minima['E'].append(energies[-2])
                    local_minima['E_prime'].append(0)
                    local_minima['E_double_prime'].append(curvatures[-2])
            elif derivatives[-1] > 0 and derivatives[-2] < 0:
                #r_zero = optimize.brentq(calc_pair_energy, rs[k-1], rs[k], args=(coefficient_1b, coefficients_2b, knot_sequence, 2))  # acccurate method
                r_zero = (rs[k] + rs[k-1]) / 2  # approximate method
                local_minima['r'].append(r_zero)
                local_minima['E'].append( (energies[-1] + energies[-2]) / 2 )
                local_minima['E_prime'].append(0)
                local_minima['E_double_prime'].append( (curvatures[-1] + curvatures[-2]) / 2 )

    fig, axs = plt.subplots(4, 1, figsize=(8, 8), sharex=True)

    # plot training data and residual if available
    if training_data is not None:
        residual = []
        for r, E in zip(training_data[0], training_data[1]):
            residual.append((E - calc_pair_energy(r, coefficient_1b, coefficients_2b, knot_sequence, nAtoms, zbl=zbl_obj))/2)
        axs[0].plot(training_data[0], residual, 'r.')
        axs[0].set_ylabel("true - fit (eV)")
        if reslim is not None:
            axs[0].set_ylim(reslim)
        axs[0].plot(knot_sequence, np.zeros(len(knot_sequence)), 'k|')
        axs[1].plot(training_data[0], training_data[1], 'g,')

    # plot energy and its derivative and its curvature
    if plot_rlim is None:
        plot_rlim = (knot_sequence[0], knot_sequence[-1])
    axs[1].plot(rs, energies)
    axs[1].set_ylabel("energy (eV)")
    if E_lim is not None:
        axs[1].set_ylim(E_lim)
    axs[1].plot(local_minima['r'], local_minima['E'], 'rx')
    axs[1].plot(knot_sequence, np.zeros(len(knot_sequence)), 'k|')
    axs[2].plot(rs, derivatives)
    axs[2].set_ylabel("derivative (eV/A)")
    if E_prime_lim is not None:
        axs[2].set_ylim(E_prime_lim)
    axs[2].plot(local_minima['r'], local_minima['E_prime'], 'rx')
    axs[2].plot(knot_sequence, np.zeros(len(knot_sequence)), 'k|')
    axs[3].plot(rs, curvatures)
    axs[3].set_ylabel("curvature (eV/A^2)")
    if E_double_prime_lim is not None:
        axs[3].set_ylim(E_double_prime_lim)
    axs[3].plot(local_minima['r'], local_minima['E_double_prime'], 'rx')
    axs[3].plot(knot_sequence, np.zeros(len(knot_sequence)), 'k|')
    axs[3].set_xlabel("r (A)")
    axs[3].set_xlim(plot_rlim)
    axs[0].set_title(label)

    return fig, axs



if __name__ == '__main__':
    n_models = 7
    for i in range(1, n_models+1):
    #i = 7
    #if True:
        print(i)
        model_file = "model_" + str(i) + ".json"
        model_number = i
        opt_traj_file = "ufmin.traj"
        rspacing = 0.01
        epsilon = 0.0000001
        calc_rlim = [1.4 + epsilon, 6 + epsilon]
        plot_rlim = [1.4, 6]
        E_lim = (-2, 5)
        E_prime_lim = (-12, 12)
        E_double_prime_lim = (-50, 50)
        reslim = (-1, 1)
        plot_components = True

        r_min = 2.22
        well_depth = 9
        lj_p = functools.partial(lj, r_min, well_depth)  # the true radial potential

        r_e = 2.897
        D_e = 0.7102
        exp_prefactor = 1.6047
        rho0 = exp_prefactor * r_e
        lj_p = functools.partial(morse, r_e, D_e, exp_prefactor)

        #============================================


        model = least_squares.WeightedLinearModel.from_json(model_file)

        if opt_traj_file is not None:
            opt_traj = Trajectory(opt_traj_file, 'r')
            training_traj = make_pair_energy_data(opt_traj[0:model_number+1], lj_p)
            nAtoms = len(opt_traj[0])
            opt_traj.close()
        else:
            training_traj = None
            nAtoms = None

        plot_single_model(model, training_traj, nAtoms=nAtoms, label=model_file,
                          calc_rlim=calc_rlim, rspacing=rspacing, epsilon=epsilon, plot_rlim=plot_rlim,
                          E_lim=E_lim, E_prime_lim=E_prime_lim, E_double_prime_lim=E_double_prime_lim, reslim=reslim,
                          plot_components=plot_components)
    plt.show()
