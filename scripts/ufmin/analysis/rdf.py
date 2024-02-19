import uf3_run
from uf3.data import analyze
from uf3.data import composition
from uf3.util import parallel

import numpy as np
import matplotlib.pyplot as plt

import os, sys, ase
from ase.io import read, trajectory



def calc_rs(traj_file, bspline_config, traj_index=None):
    analyzer = analyze.DataAnalyzer(bspline_config.chemical_system, r_cut=plot_2b_xlim[1], bins=rdf_bin_res)

    if traj_index is None or traj_index == ":" or traj_index == slice(None):
        try:
            traj = trajectory.Trajectory(traj_file, 'r')
        except ase.io.ulm.InvalidULMFileError:
            atoms = read(traj_file, index=":")
        except:
            raise Exception(f"Could not read {traj_file}.")
    else:
        traj = read(traj_file, index=traj_index)
        if not isinstance(traj, list):
            traj = [traj]

    n_entries = len(traj)
    iterable = parallel.progress_iter(enumerate(traj),
                                    total=n_entries)

    distances = dict()
    for j, geom in iterable:
        dists, hashes = analyzer.get_distances(geom, r_min=0.1, r_max=6.0)
        tmp_distances = composition.hash_gather(dists, hashes)
        for hash in tmp_distances:
            try:
                distances[hash] = np.concatenate((distances[hash], tmp_distances[hash]))
            except KeyError:
                distances[hash] = tmp_distances[hash]
    for hash in distances:
        distances[hash].sort()

    try:
        traj.close()
    except AttributeError:
        pass

    return distances


def plot_rdf(distances, bspline_config, rlim, bin_res, fig_name = None):
    nPairTypes = len(distances)
    fig, axs = plt.subplots(nPairTypes, 1, sharex=True, figsize=(6, 2*nPairTypes))
    if nPairTypes == 1:
        axs = [axs]
    axs[0].set_xlabel("r (Å)")
    hash2pair = {hashed_pair: pair for pair, hashed_pair in  
                 zip(bspline_config.chemical_system.interactions_map[2],
                 bspline_config.chemical_system.interaction_hashes[2]
                 )   
                 }  # from r_uq
    for i, hash in enumerate(distances):
        axs[i].hist(distances[hash], bins=np.arange(rlim[0], rlim[1], bin_res), density=True)
        axs[i].set_title(hash2pair[hash])
        axs[i].set_ylabel("g(r)")
        axs[i].set_xlim(rlim)
    plt.tight_layout()

    if fig_name is not None:
        plt.savefig(fig_name)
    else:
        plt.show()


if __name__ == "__main__":
    results_dir = "."
    traj_file = "ufmin.traj"
    traj_index = -1
    settings_file = "settings.yaml"
    fig_name = "final_rdf.png"

    plot_2b_xlim = [1.0, 6.0]
    rdf_bin_res = 0.01

    bspline_config = uf3_run.initialize( os.path.join(results_dir, settings_file) )
    traj_file = os.path.join(results_dir, traj_file)
    distances = calc_rs(traj_file, bspline_config, traj_index)
    plot_rdf(distances, bspline_config, plot_2b_xlim, rdf_bin_res, fig_name=fig_name)
