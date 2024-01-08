from ase.atoms import Atoms
import ase.data as ase_data
from ase.optimize import LBFGS

import myopts

from uf3.data import geometry
from uf3.representation import distances
from uf3.forcefield import calculator

import numpy as np

from typing import List, Tuple
from copy import deepcopy


class QRP:
    """
    A quadratic repulsive potential.

    A half quadratic potential function with a leftmost minimum at r=r0 and a
    flat potential for r>r0. The potential is scaled by a factor of scale.

    Args:
        r (float | np.ndarray): The interatomic distance(s).
        r0 (float): The minimum of the potential.
        scale (float): The scale factor of the potential (>0).
    """
    def __init__(self,
                 r0: float,
                 scale: float = 100,
                 ):
        self.r0 = r0
        self.scale = scale
        self.r_max = self.r0

    def __call__(self,
                 r: float | np.ndarray,
                 ) -> float | np.ndarray:
        return np.where(r < self.r0, self.scale * (r - self.r0) ** 2, 0)

    def d(self,
          r: float | np.ndarray
          ) -> float | np.ndarray:
        return np.where(r < self.r0, 2 * self.scale * (r - self.r0), 0)


class LRP:
    """
    A linear repulsive potential.

    Linear with a negative slope for r<r0 and a flat potential valued at 0 for
    r>=r0. The potential is scaled by a factor of scale.
    
    Args:
        r (float | np.ndarray): The interatomic distance(s).
        r0 (float): The minimum of the potential.
        scale (float): The scale factor of the potential (>0).
    """
    def __init__(self,
                 r0: float,
                 scale: float = 1.0,
                 ):
        self.r0 = r0
        self.scale = scale
        self.r_max = self.r0

    def __call__(self,
                 r: float | np.ndarray,
                 ) -> float | np.ndarray:
        return np.where(r < self.r0, self.scale * (self.r0 - r), 0)

    def d(self,
          r: float | np.ndarray
          ) -> float | np.ndarray:
        return np.where(r < self.r0, -1 * self.scale, 0)


def preprocess(atoms: Atoms,
               pair_tuples: List[Tuple[str, str]],
               strength: float = 0.5,
               ) -> Atoms:
    """
    Preprocess the atoms object with a repulsive quadratic potential to expand
    small interatomic distances.

    Args:
        atoms (ase.atoms.Atoms): The atoms object to preprocess.
        strength (float): How much to preprocess.
            Represents the fraction of the estimated bond length of a
            pair interaction, which is used as the threshold for preprocessing.
            i.e. The preprocessed atoms object will have all interatomic
            distances above this threshold.

    Returns:
        The preprocessed atoms object.
    """
    atoms = deepcopy(atoms)

    # r_max_map for all pair types
    r_max_map = {}
    for pair in pair_tuples:
        z1 = ase_data.atomic_numbers[pair[0]]
        z2 = ase_data.atomic_numbers[pair[1]]
        approximate_bond_length = ase_data.covalent_radii[z1] + ase_data.covalent_radii[z2]
        r_max_map[pair] = approximate_bond_length * strength
    r_cut = max(r_max_map.values())

    epsilon = 1e-6
    r_min_map = {pair: 0.0 for pair in pair_tuples}

    def get_distances_map(atoms: Atoms,):
        if any(atoms.pbc):
            supercell = geometry.get_supercell(atoms, r_cut=r_cut)
        else:
            supercell = atoms
        distances_map = distances.distances_by_interaction(atoms,
                                                        pair_tuples,
                                                        r_min_map,
                                                        r_max_map,
                                                        supercell)
        return distances_map


    # determine if preprocessing is necessary
    distances_map = get_distances_map(atoms)
    if all([len(distances_map[pair]) == 0 for pair in pair_tuples]):
        print("No preprocessing necessary.")
        return atoms
    
    # make repulsive potential objects for each pair
    RPClass = LRP
    rp_objs = {}
    for pair in pair_tuples:
        r_max = r_max_map[pair]
        rp_objs[pair] = RPClass(r_max)
    
    # create ASE calculator and optimizer for QRP
    qrp_calc = calculator.UFStylePairCalculator(rp_objs,)
    atoms.calc = qrp_calc
    #dyn = LBFGS(atoms, trajectory="preprocess.traj")
    dyn = myopts.GD(atoms, trajectory="preprocess.traj")

    # preprocess
    max_steps = 10  # only a couple should be necessary
    counter = 1
    print("Preprocessing...")
    while True:
        dyn.run(steps=counter, fmax=0.0)
        distances_map = get_distances_map(atoms)
        counter += 1
        if all([len(distances_map[pair]) == 0 for pair in pair_tuples]):
            print("Preprocessing successfully complete.")
            break
        elif counter > max_steps:
            print("Preprocessing failed to complete.")
            break

    atoms.calc = None  # remove the fictitious calculator
    return atoms


if __name__ == '__main__':
    from ase.io import read
    from uf3.data import composition

    atoms = read("POSCAR")
    element_list = list(set(atoms.get_chemical_symbols()))
    chemical_system = composition.ChemicalSystem(element_list, degree=2)
    pair_tuples = chemical_system.get_interactions_map()[2]

    atoms = preprocess(atoms, pair_tuples)