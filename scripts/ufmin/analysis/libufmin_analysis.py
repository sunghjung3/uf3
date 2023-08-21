import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from uf3.util import cubehelix


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

def calc_pair_energy(rs, coefficient_1b, coefficients_2b, knot_sequence, nAtoms, zbl=None):
    # includes 1 body offset energy
    nBonds = nAtoms * (nAtoms - 1) / 2
    uf3_energy =  bsplines_interpolate(rs, 2*coefficients_2b, knot_sequence) + np.ones(np.shape(rs)) * nAtoms / nBonds * coefficient_1b
    zbl_energy = np.zeros(rs.shape)
    if zbl is not None:
        zbl_energy += zbl(rs)
    return uf3_energy + zbl_energy

# from uf3.util.plotting.visualize_splines()
def plot_pair_energy(coefficient_1b, coefficients_2b, knot_sequence, nAtoms, zbl=None, ax=None, cmap=None, show_components=True, show_total=True, xlim=None, ylim=None):
    if xlim is None:
        r_min = knot_sequence[0]
        r_max = knot_sequence[-1]
        xlim = [r_min, r_max]
    if ax is None:              
        fig, ax = plt.subplots()
    else:                    
        fig = ax.get_figure()
    if cmap is None:
        cmap = cubehelix.c_rainbow                     
    colors = cmap(np.linspace(0, 1, len(coefficients_2b)+2))  # +2 for 1b coefficient and zbl
    #x_plot = np.linspace(r_min, r_max, 1000)
    x_plot = np.linspace(*xlim, 200)
    basis_components = bsplines_interpolate(x_plot, 2*coefficients_2b, knot_sequence, sum=False)

    # add ZBL
    zbl_energy = np.zeros(x_plot.shape)
    if zbl is not None:
        zbl_energy += zbl(x_plot)

    if show_components:
        for i, basis_component in enumerate(basis_components):
            ax.plot(x_plot,
                    basis_component,
                    color=colors[i],
                    linewidth=1)
        ax.plot(x_plot,
                zbl_energy,
                color=colors[-2],
                linewidth=1)

    y_total = np.sum(basis_components, axis=0)
    if ylim is None:
        s_min = np.min(y_total[~np.isnan(y_total)])
        s_max = np.max(y_total[~np.isnan(y_total)])
        ylim = [s_min, s_max]

    # add 1 body offset
    nBonds = nAtoms * (nAtoms - 1) / 2
    y_plot = np.ones(np.shape(x_plot)) * nAtoms / nBonds * coefficient_1b
    if show_components:
        ax.plot(x_plot,
                y_plot,
                '--',
                color=colors[-1],
                linewidth=1)
    y_total += y_plot

    if show_total:
        ax.plot(x_plot,
                y_total,
                c='k',      
                linewidth=2) 
    #ax.set_xlim(r_min, r_max)
    #ax.set_ylim(s_min, s_max)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("r")
    ax.set_ylabel("E(r)")
    return fig, ax
