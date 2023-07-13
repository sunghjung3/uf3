from typing import Iterable, List
from collections import defaultdict, Counter
import heapq

import numpy as np
from scipy import spatial

import ase
from ase.atoms import Atoms
from ase.neighborlist import PrimitiveNeighborList, NeighborList

from uf3.representation import bspline
from uf3.data import composition, geometry


global_dr_trust = 0.64


class R_UQ_NeighborList(PrimitiveNeighborList):
    """
    A class to hold neighbor lists for the R_UQ class.

    It is inherited from ASE's PrimitiveNeighborList class with these changes:
        * The `self_interaction` parameter is False by default.
        * The `bothways` parameters is True by default.
        * The default value for the `skin` parameter is equal to the
            half of the `global_dr_trust` global variable by default.
        * A rebuild of the neighbor list is triggered if the sum of the largest
            2 atomic displacements exceed twice the skin depth.
    """ 

    def __init__(self, cutoffs, skin=global_dr_trust/2, sorted=False,
                 self_interaction=False, bothways=True,
                 use_scaled_positions=False):
        super().__init__(cutoffs, skin, sorted, self_interaction,
                         bothways, use_scaled_positions)

    
    def update(self, pbc, cell, coordinates):
        """Make sure the list is up to date."""

        if self.nupdates == 0:
            self.build(pbc, cell, coordinates)
            return True

        largest_2_displacements = np.sqrt(
                heapq.nlargest(2, ((self.coordinates - coordinates)**2).sum(1))
        )

        if ((self.pbc != pbc).any() or (self.cell != cell).any()
            or np.sum(largest_2_displacements) > 2*self.skin):
            self.build(pbc, cell, coordinates)
            return True

        return False


class R_UQ:
    """
    A class for the r-based UQ method to be used during UF3-accelerated
    geometry optimization.
    
    Intended to be instantiated only once for a single optimization job.

    This assumes that the atomic order and composition as well as the following
    attributes of `bspline_config` will remain constant:
        * r_min_map
        * r_max_map
        * chemical_system

    Attributes
    ----------
    atoms : ase.atoms.Atoms
        Current geometry.
        Referenced in memory but not modified within this class.
    trained_traj : List[ase.atoms.Atoms]
        List of Atoms objects that the currently-used UF3 model
        was trained with.
        Referenced in memory but not modified within the class.
    bspline_config : uf3.representation.bspline.BSplieBasis
        BSplineBasis object containing UF3 information, such as pair hashes and
        cutoff distances.
    dr_trust : float
        Trust radius of how far a pair distance can be away from trained
        distances before a "high uncertainty" is triggered.

    Methods
    -------
    nl_cutoffs()
        Creates a list of cutoffs for neighbor list construction
    """


    def __init__(self, atoms: Atoms,
                       trained_traj: List[Atoms],
                       bspline_config: bspline.BSplineBasis,
                       dr_trust: float = global_dr_trust):
        """
        Parameters
        ----------
        atoms : ase.atoms.Atoms
            See class description.
        trained_traj : List[ase.atoms.Atoms]
            See class description.
        bspline_config : uf3.representation.bspline.BSplineBasis
            See class description.
        dr_trust : float
            See class description (default = `global_dr_trust`).
        """
        self.atoms = atoms
        self.trained_traj = trained_traj
        self.bspline_config = bspline_config
        self.dr_trust = dr_trust
        if Counter(set(atoms.get_chemical_symbols())) != Counter(self.bspline_config.element_list):
            raise Exception("Elements in atoms do not match those in bspline_config.")

        # Neighbor list things
        ase_nl_skin = self.dr_trust / 2  # half because ASE-style neighbor list
                                         # no half with LAMMPS-style NL
        self.nl = NeighborList(cutoffs=self.nl_cutoffs(),
                               skin=ase_nl_skin,
                               sorted=False,
                               self_interaction=False,
                               bothways=True,
                               primitive=R_UQ_NeighborList
                               )

        # Hash things
        self.pair2hash2b = {pair: hashed_pair for pair, hashed_pair in 
                            zip(self.bspline_config.chemical_system.interactions_map[2],
                            self.bspline_config.chemical_system.interaction_hashes[2]
                            )
                           }
        if self.bspline_config.degree > 2:
            self.pair2hash3b = {pair: hashed_pair for pair, hashed_pair in 
                                zip(self.bspline_config.chemical_system.interactions_map[3],
                                self.bspline_config.chemical_system.interaction_hashes[3]
                                )
                               }
        self.r_min_2b_hashed = {self.pair2hash2b[pair]: value for pair, value in self.bspline_config.r_min_map.items() if len(pair) == 2}  # e.g.: {("Pt", "Pt"): 0.1} --> {6240: 0.1}
        self.r_max_2b_hashed = {self.pair2hash2b[pair]: value for pair, value in self.bspline_config.r_max_map.items() if len(pair) == 2}
        self.pair_hash_array = self.build_pair_hash_array()
        # TODO: implement similar "hash tensor" for 3 body


        # Reference (trained) r's things
        self.r2_gaps = dict()  # store gaps in r^2 space for 2 body
        self.trained_triples = defaultdict(list)  # 3 body
        self.trained_traj_len = 0  # how many images in self.trained_traj have I considered in update_trained_rs()?
        self.no_more_gaps = defaultdict(bool)  # True when no gaps remaining; {pair_hash: bool}


    def build_pair_hash_array(self):
        # Hash matrices of pairs between self.atoms and a supercell (nRows < nCols)
        # NOTE: When working with ASE-style neighborlists, the leftmost (nRows x nRows) submatrix can be used.
        nAtoms = len(self.atoms)
        largest_r = max(self.r_max_2b_hashed.values())
        if any(self.atoms.pbc):
            tmp_supercell = geometry.get_supercell(self.atoms, r_cut=largest_r)
        else:
            tmp_supercell = self.atoms
        nSupercell = len(tmp_supercell)
        upper_tri_idx = np.triu_indices(nAtoms, k=1, m=nSupercell)
        species_set = (self.atoms.get_atomic_numbers(),
                       tmp_supercell.get_atomic_numbers())
        symbols_set = (self.atoms.get_chemical_symbols(),
                       tmp_supercell.get_chemical_symbols())
        pair_hashes_by_idx = composition.get_pair_hashes(species_set, symbols_set, upper_tri_idx)
        pair_hash_array = np.zeros((nAtoms, nSupercell))
        pair_hash_array[upper_tri_idx] = pair_hashes_by_idx
        lower_tri_idx = np.tril_indices(nAtoms, k=-1)
        truncated_upper_tri_idx = (lower_tri_idx[1], lower_tri_idx[0])
        pair_hash_array[lower_tri_idx] = pair_hash_array[truncated_upper_tri_idx]  # fill lower part
        return pair_hash_array


    def nl_cutoffs(self) -> List:
        """
        Creates a list of cutoffs for neighbor list construction based on
        self.bspline_config.r_max_map.

        Parameters
        ----------
        None

        Returns
        -------
        cutoffs : List
            Cutoff radii for the neighbor list for each atom in self.atoms.
            Since we are using ASE neighbor lists, where it defines "neighbors"
            as touching spheres, the cutoff would be something like half the
            largest r_max for each element.

        """
        element_list = self.bspline_config.element_list
        r_max_map = self.bspline_config.r_max_map

        # for each element in element_list, find the largest possible cutoff
        cutoffs_by_element = dict()
        for element in element_list:
            max_r_max = -1  # initialize
            for interaction, r_max in r_max_map.items():
                if element not in interaction:
                    continue
                if len(interaction) == 2 and r_max > max_r_max:  # 2 body
                    max_r_max = r_max  # new high score!
                elif len(interaction) == 3:  # 3 body
                    if interaction[0] == element:  # is center atom
                        trial_r_max = max(r_max[0], r_max[1])
                    elif interaction[1] == element:
                        trial_r_max = r_max[0]
                    elif interaction[2] == element:
                        trial_r_max = r_max[1]
                    else:
                        raise Exception("Something is going horribly wrong.")
                    if trial_r_max > max_r_max:
                        max_r_max = trial_r_max  # new high score!

            # divide by 2 for ASE-style neighbor list
            cutoffs_by_element[ase.data.atomic_numbers[element]] = max_r_max/2
        
        atomic_numbers = self.atoms.get_atomic_numbers()
        cutoffs = [cutoffs_by_element[i] for i in atomic_numbers]
        return cutoffs


    def distances_by_hash(self, geom):
        """ 
        Computes lists of observed distances separated by pair hash. Each pair distance is not double counted.

        Args:
            geom (ase.atoms.Atoms): Atoms object of interest

        Returns:
            distances_lists (dict): list of observed pair distances by pair hash
        """
        smallest_r = min(self.r_min_2b_hashed.values())
        largest_r = max(self.r_max_2b_hashed.values())

        if any(geom.pbc):
            supercell = geometry.get_supercell(geom, r_cut=largest_r)
        else:
            supercell = geom

        nAtoms = len(geom)
        nSupercell = len(supercell)

        dist_matr = spatial.distance.cdist(geom.get_positions(),
                                            supercell.get_positions())  # distance of atoms in geom to atoms on supercell

        # apply r_min and r_max cut masks for each pair type
        ###
        hash2r_min = np.vectorize(lambda key: self.r_min_2b_hashed.get(key, largest_r))  # default to largest_r so that it will get filtered out later
        r_min_cut = hash2r_min(self.pair_hash_array)
        hash2r_max = np.vectorize(lambda key: self.r_max_2b_hashed.get(key, smallest_r))  # default to smallest_r so that it will get filtered out later
        r_max_cut = hash2r_max(self.pair_hash_array)
        filter = (dist_matr > r_min_cut) & (dist_matr <= r_max_cut)
        
        dists_cut = dist_matr[filter]  # filter applied, make 1D array
        hashes_cut = self.pair_hash_array[filter]  # hashes corresponding to filtered distances
        distances_lists = composition.hash_gather(dists_cut, hashes_cut)

        return distances_lists

    
    def triples_by_hash(self):
        raise NotImplementedError("Not written yet.")


    def update_trained_rs(self):
        self.update_gaps()
        #if self.bspline_config.degree > 2:  # 3 body
        #    self.update_trained_rs_3b()
        self.trained_traj_len = len(self.trained_traj)


    def update_gaps(self):  # fill in gaps with new r's

        # Gaps in r space:
        #   |---------*===============*---------|
        #       - are buffer regions of length self.dr_trust
        #       = is the core region
        #       A new r is uncertain if it is in the core region

        lower_traj_idx = self.trained_traj_len
        upper_traj_idx = len(self.trained_traj)
        trained_pairs = defaultdict(list)
        for image in self.trained_traj[lower_traj_idx:upper_traj_idx]:
            for pair_hash, rs in self.distances_by_hash(image).items():
                trained_pairs[pair_hash].extend(rs)

        # Numerical error tolerance to mitigate error accumulation from repeated squaring and rooting
        num_tol = 0.000001  

        # use self.r_max_2b_hashed instead of trained_pairs because possibility of missing hash
        for pair_hash in self.r_max_2b_hashed:
            r_min = self.r_min_2b_hashed[pair_hash]
            r_max = self.r_max_2b_hashed[pair_hash]
            rs = trained_pairs.get(pair_hash, list())
            assert not rs or rs[0] >= r_min-num_tol  # either empty or meet sanity check
            assert not rs or rs[-1] <= r_max+num_tol
            min_r_spacing = 2 * self.dr_trust
            assert r_max - r_min >= min_r_spacing

            try:
                tmp_gaps = self.r2_gaps[pair_hash]
                # Temporarily bring it to r space and include buffer region
                for gap in tmp_gaps:
                    gap[0] = gap[0] ** 0.5 - self.dr_trust
                    gap[1] = gap[1] ** 0.5 + self.dr_trust
            except KeyError:
                tmp_gaps = [ [r_min, r_max] ]  # first time (buffers included)
            for r in rs:
                gap_idx_to_remove = list()
                gaps_to_add = list()
                for gap_idx, gap in enumerate(tmp_gaps):
                    if r > gap[0] and r < gap[1]:  # time to split this gap
                        gap_idx_to_remove.append(gap_idx)
                        if r - gap[0] > min_r_spacing:
                            gaps_to_add.append( [gap[0], r] )
                        if gap[1] - r > min_r_spacing:
                            gaps_to_add.append( [r, gap[1]] )
                for gap_idx in gap_idx_to_remove:  # remove broken gaps
                    del tmp_gaps[gap_idx][:]
                    del tmp_gaps[gap_idx]
                tmp_gaps.extend(gaps_to_add)  # add new gaps
            del rs[:]

            # Remove buffers again and take it to r^2 space
            for gap in tmp_gaps:
                gap[0] = (gap[0] + self.dr_trust) ** 2
                gap[1] = (gap[1] - self.dr_trust) ** 2
        
            self.r2_gaps[pair_hash] = tmp_gaps


    def update_trained_rs_3b(self):
        raise NotImplementedError("Not written yet.")
        for image in self.trained_traj[self.trained_traj_len:]:
            for hash, triples in self.triples_by_hash(image):  # 2 body
                pass
                # 3b: bin the triples space into voxels with width self.dr_trust, and check the adjaent 8 voxels. Can I work in r^2 space?


    def check_r_2b(self):
        # check based on r2_gaps
        for atom in range(len(self.atoms)):
            neighbors, offsets = self.nl.get_neighbors(atom)

            # XXX: use these for vectorized but not filtered version
            #diff = atoms.positions[neighbors] + offsets@atoms.cell - atoms.positions[atom]
            #r2s = (diff**2).sum(1)  # array of r^2 for all neighbors
            #uncertain_list = np.ones(np.shape(r2s), dtype=bool)  # True if uncertain

            # Filter to avoid double counting of pairs
            for i, neighbor in enumerate(neighbors):
                offset = offsets[i]
                if neighbor <= atom and (not any(offset)):
                    continue
                diff = self.atoms.positions[neighbor] +  offset@self.atoms.cell - self.atoms.positions[atom]
                r2 = (diff**2).sum()  # r^2
                pair_hash = self.pair_hash_array[atom, neighbor]
                for gap in self.r2_gaps[pair_hash]:
                    if r2 > gap[0] and r2 < gap[1]:  # in the gap
                        return True  # uncertain
        return False  # not uncertain


    def check_r_3b(self):
        # check based on trained_triples
        raise NotImplementedError("Not written yet.")
        return False  # not uncertain

    def too_uncertain(self):
        # Check neighbor list
        self.nl.update(self.atoms)  # build new neighbor list if needed

        # If there is new training data
        if len(self.trained_traj) != self.trained_traj_len:
            self.update_trained_rs()

        # Check epistemic uncertainty
        if self.check_r_2b():
            return True
        #if self.bspline_config.degree > 2:
        #    return self.check_r_3b()
        return False
        


if __name__ == "__main__":
    # tests at /home/sung/UFMin/sung/data/lj_pt38/1/5/test/

    #atoms = Atoms("PtCH")
    atoms = Atoms("Pt3")
    print(atoms.get_atomic_numbers())
    atoms.positions = np.array([ [0, 0, 0], [3.5, 0, 0], [9, 0, 0] ])
    atoms.cell = np.array([ [10, 0, 0], [0, 10, 0], [0, 0, 10] ])
    atoms.pbc = [True, True, True]

    import uf3_run
    import copy
    bspline_config = uf3_run.initialize()

    traj = [copy.deepcopy(atoms), copy.deepcopy(atoms)]
    traj[1].positions = np.array([ [0, 0, 0], [3.4, 0, 0], [8.9, 0, 0] ])
    traj.append(copy.deepcopy(atoms))
    traj[-1].positions = np.array([ [0, 0, 0], [3.3, 0, 0], [8.8, 0, 0] ])


    r_uq = R_UQ(atoms, traj, bspline_config)

    print(r_uq.atoms.positions)

    atoms.positions = np.array([ [0, 0, 0], [3.2, 0, 0], [8.7, 0, 0] ])
    print(r_uq.atoms.positions)
    print(r_uq.too_uncertain())


    atoms.positions = np.array([ [0, 0, 0], [2.2, 0, 0], [8.6, 0, 0] ])
    print(r_uq.atoms.positions)
    print(r_uq.too_uncertain())