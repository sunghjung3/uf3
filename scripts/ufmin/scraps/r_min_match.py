from ase.atoms import Atoms
from ase.io.trajectory import Trajectory
import numpy as np
from scipy import interpolate, optimize

from uf3.regression import least_squares


def bspline_interpolate(xs, c, kn):
    # x, coefficient for basis, knot sequence for this basis
    kno = np.concatenate([np.repeat(kn[0], 3), 
                          kn, 
                          np.repeat(kn[-1], 3)])
    bs = interpolate.BSpline(kno,    
                             np.array([0, 0, 0, c, 0, 0, 0]),
                             3,  
                             extrapolate=False)
    y_plot = bs(xs)
    return y_plot

def bsplines_interpolate(xs, coefficients, knot_sequence, sum=True):
    basis_components = []
    for i, c in enumerate(coefficients):
        kn = knot_sequence[i:i + 5]
        y_plot = bspline_interpolate(xs, c, kn) 
        y_plot[np.isnan(y_plot)] = 0 
        basis_components.append(y_plot)
    if sum:
        return np.sum(basis_components, axis=0)
    else:
        return basis_components

def calc_pair_energy(rs, coefficient_1b, coefficients_2b, knot_sequence, nAtoms):
    # includes 1 body offset energy
    nBonds = nAtoms * (nAtoms - 1) / 2 
    return bsplines_interpolate(rs, 2*coefficients_2b, knot_sequence) + np.ones(np.shape(rs)) * nAtoms / nBonds * coefficient_1b



atoms = Atoms("Pt2")
atoms.set_positions([[0, 0, 0], [0, 0, 0]])
atoms.cell = np.eye(3) * 20
atoms.pbc = [True, True, True]

from ase.calculators.lj import LennardJones
true_r_min = 2.22
well_depth = 9
r_cut = 8 * true_r_min
true_calc = LennardJones(sigma=true_r_min*(2 ** (-1/6)), epsilon=well_depth, rc=r_cut)
atoms.calc = true_calc

traj = Trajectory("lj.traj", "w")
r_min = 1.4
r_max = 6.0
spacing = 0.0005
rs = np.arange(r_min, r_max, spacing)
for r in rs:
    atoms.positions[1, 0] = r
    atoms.get_potential_energy()
    traj.write(atoms)

traj.close()

# find elements in rs closest to true_r_min and print neighboring values
closest = np.argmin(np.abs(rs - true_r_min))
print("closest:", rs[closest-10:closest+10])

import uf3_run
from uf3.forcefield import calculator

import matplotlib.pyplot as plt

true_energies = list()
for r in rs:
    atoms.positions[1, 0] = r
    true_energies.append(atoms.get_potential_energy())

knots_to_scan = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
model_energies = dict()
model_rmin = dict()
true_force_at_model_rmin = dict()
model_force_at_true_rmin = dict()
epsilon = 0.001
for knots in knots_to_scan:
    print("knots =", knots)
    knots_map = {"Pt-Pt": knots}
    bspline_config = uf3_run.initialize(resolution_map=knots_map, overwrite=True)
    df_features = uf3_run.featurize(bspline_config, overwrite=True)
    model = uf3_run.train(df_features, bspline_config, model_file="model_"+str(knots)+".json", overwrite=True)
    model_calc = calculator.UFCalculator(model)
    atoms.calc = model_calc

    # find the model force at the true rmin
    atoms.positions[1, 0] = true_r_min
    model_force_at_true_rmin[knots] = (atoms.get_forces()[0] ** 2).sum() ** 0.5
    print("model at true:", true_r_min, model_force_at_true_rmin[knots])
    print("\tRaw forces:", atoms.get_forces()[0])

    e = np.zeros((len(rs), 2))
    for i, r in enumerate(rs):
        atoms.positions[1, 0] = r
        e[i] = [r, atoms.get_potential_energy()]
    model_energies[knots] = e

    
    # find value of r where energy is minimum
    rmin_guess = e[np.argmin(e[:, 1]), 0] + epsilon
    solutions = least_squares.arrange_coefficients(model.coefficients, model.bspline_config)
    element = model.bspline_config.element_list[0]
    pair = (element, element)
    coefficient_1b = solutions[element]
    coefficients_2b = solutions[pair]
    knot_sequence = model.bspline_config.knots_map[pair]
    rmin_0 = optimize.minimize(calc_pair_energy, rmin_guess, args=(coefficient_1b, coefficients_2b, knot_sequence, 2)).x[0]
    rmin_guess -= 2*epsilon
    rmin_1 = optimize.minimize(calc_pair_energy, rmin_guess, args=(coefficient_1b, coefficients_2b, knot_sequence, 2)).x[0]
    rmin = (rmin_0 + rmin_1) / 2
    print("rmin =", rmin)
    model_rmin[knots] = rmin

    atoms.positions[1, 0] = rmin
    atoms.calc = true_calc
    true_force_at_model_rmin[knots] = (atoms.get_forces()[0] ** 2).sum() ** 0.5
    print("true at model:", rmin, true_force_at_model_rmin[knots])
    print("\tRaw forces:", atoms.get_forces()[0])

    # numerically estimate the force by taking the difference in energies between two very small r values
    atoms.positions[1, 0] += epsilon
    e1 = atoms.get_potential_energy()
    print(atoms.positions[1, 0], e1)
    atoms.positions[1, 0] -= 2 * epsilon
    e2 = atoms.get_potential_energy()
    print(atoms.positions[1, 0], e2)
    print("numerical force =", np.abs(e1 - e2) / (2 * epsilon) )


    # plot r vs the residual between the true energy and the model energy
    res_fig, res_ax = plt.subplots()
    res_ax.plot(e[:, 0], e[:, 1] - true_energies, 'r,')
    res_ax.set_xlabel("r")
    res_ax.set_ylabel("residual")
    res_ax.set_title("knots = " + str(knots))

    # plot r vs the true energy and the model energy
    energy_fig, energy_ax = plt.subplots()
    energy_ax.plot(e[:, 0], e[:, 1], 'r,-', label="model")
    energy_ax.plot(e[:, 0], true_energies, 'b,-', label="true")
    energy_ax.set_xlabel("r")
    energy_ax.set_ylabel("energy")
    energy_ax.set_title("knots = " + str(knots))
    energy_ax.set_ylim([-10, 25])
    energy_ax.legend()


    plt.show()
    


# plot true_force_at_model_rmin vs knots with x axis on log scale
r_min_fig, r_min_ax = plt.subplots()
r_min_ax.plot(knots_to_scan, list(true_force_at_model_rmin.values()), 'bo-', label="true force at model rmin")
r_min_ax.plot(knots_to_scan, list(model_force_at_true_rmin.values()), 'ro-', label="model force at true rmin")
r_min_ax.set_xscale('log')
r_min_ax.set_yscale('log')
r_min_ax.set_xlabel("knots")
r_min_ax.set_ylabel("true force at model rmin")
r_min_ax.legend()
r_min_ax.hlines(0.05, 1, 10000, colors='k', linestyles='dashed')
r_min_ax.set_xlim([4, 4096])
print(true_force_at_model_rmin)
plt.show()
