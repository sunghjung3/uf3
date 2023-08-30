from typing import Iterable, List, Tuple
from collections import defaultdict, Counter
import heapq

import numpy as np
from scipy import spatial

import ase
from ase.atoms import Atoms
from ase.neighborlist import PrimitiveNeighborList, NeighborList

from uf3.representation import bspline
from uf3.data import composition, geometry




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

    def __init__(self, cutoffs, skin=0.32, sorted=False,
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
    A superclass for the r-based UQ method to be used during UF3-accelerated
    geometry optimization.

    Intended to be instantiated only once for a single optimization job.
    
    This assumes that the atomic order and composition as well as the following
    attributes of `bspline_config` will remain constant:
        * r_min_map
        * r_max_map
        * chemical_system

    This superclass should be inherited as a general framework for body-order-specific subclasses.
    The subclasses should implement the following methods:
        * determine_dr_trust()
        * update_trained_rs()
        * check_r()

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
        cutoff distances. The uncertainty trust radius is determined by the 
        cutoff and resolution information in this object.
    uq_tolerance : float
        The uncertainty tolerance level. The trust radius (dr_trust) is given
        by uq_tolerance * 
    """

    def __init__(self, atoms: Atoms,
                 trained_traj: List[Atoms],
                 bspline_config: bspline.BSplineBasis,
                 uq_tolerance: float = 1.0,
                 ):
        """
        Parameters
        ----------
        atoms : ase.atoms.Atoms
            See class description.
        trained_traj : List[ase.atoms.Atoms]
            See class description.
        bspline_config : uf3.representation.bspline.BSplineBasis
            See class description.
        """
        self.atoms = atoms
        self.trained_traj = trained_traj
        self.bspline_config = bspline_config
        if Counter(set(atoms.get_chemical_symbols())) > Counter(self.bspline_config.element_list):
            raise Exception("There are more elements in atoms than bspline_config.")

        # Neighbor list things
        dr_trust = self.determine_dr_trust(self.bspline_config, scale=uq_tolerance)
        dr_trust_values = list(dr_trust.values())
        if isinstance(dr_trust_values[0], Iterable):
            max_value = max(max(lst) for lst in dr_trust_values)
        else:
            max_value = max(dr_trust_values)
        ase_nl_skin = max_value / 2  # half because ASE-style neighbor list
                                         # no half with LAMMPS-style NL
        self.nl = NeighborList(cutoffs=self.nl_cutoffs(),
                               skin=ase_nl_skin,
                               sorted=False,
                               self_interaction=False,
                               bothways=True,
                               primitive=R_UQ_NeighborList
                               )

        # Hash things
        degree = self.bspline_config.degree
        self.interaction2hash = {interaction: hashed_interaction for interaction, hashed_interaction in
                                    zip(self.bspline_config.chemical_system.interactions_map[degree],
                                    self.bspline_config.chemical_system.interaction_hashes[degree]
                                    )
                                    }
        self.r_min_hashed = {self.interaction2hash[interaction]: value for interaction, value in self.bspline_config.r_min_map.items() if len(interaction) == degree}  # e.g.: {("Pt", "Pt"): 0.1} --> {6240: 0.1}
        self.r_max_hashed = {self.interaction2hash[interaction]: value for interaction, value in self.bspline_config.r_max_map.items() if len(interaction) == degree}
        self.dr_trust = {self.interaction2hash[interaction]: value for interaction, value in dr_trust.items() if len(interaction) == degree}

        self.trained_traj_len = 0  # how many images in self.trained_traj have been considered in update_trained_rs()

    def determine_dr_trust(self,
                           bspline_config: bspline.BSplineBasis,
                           scale: float = 1.0,
                           ) -> dict:
        """
        Determines the trust radius of how far a pair distance can be away from
        trained distances before a "high uncertainty" is triggered.

        This method should be implemented by the subclass.

        Parameters
        ----------
        bspline_config : uf3.representation.bspline.BSplineBasis
            BSplineBasis object containing UF3 information.

        scale : float
            A scaling factor for the trust radius. The default value is 1.0.

        Returns
        -------
        dr_trust : dict
            Trust radius for each highest-degree interaction of how far a pair
            distance can be away from trained distances before a
            "high uncertainty" is triggered.
        """
        raise NotImplementedError("This should be implemented by the subclass.")

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

    def update_trained_rs(self):
        """
        Updates the record of training data (distances) on file.
        """
        raise NotImplementedError("This should be implemented by the subclass.")
        self.trained_train_len = len(self.trained_traj)  # should be updated by subclass

    def check_r(self):
        """
        Checks if the current geometry is too epistemically uncertain based on
        the r-based UQ method.
        """
        raise NotImplementedError("This should be implemented by the subclass.")
    
    def too_uncertain(self):
        """
        Updates neighbor lists, training data, and checks uncertainty.
        """
        # Check neighbor list
        self.nl.update(self.atoms)  # build new neighbor list if needed

        # If there is new training data
        if len(self.trained_traj) != self.trained_traj_len:
            self.update_trained_rs()
        
        # Check epistemic uncertainty
        return self.check_r()


class R_UQ_2B(R_UQ):
    """
    A class for the r-based UQ method up to 2-body interactions to be used
    during UF3-accelerated geometry optimization. The trust radius is set to be
    the smallest knot interval of all 2-body interactions.

    See superclass `R_UQ` for more details.
    """

    def __init__(self, atoms: Atoms,
                       trained_traj: List[Atoms],
                       bspline_config: bspline.BSplineBasis,
                       uq_tolerance: float = 1.0,
                       ):
        self.degree = 2
        if bspline_config.degree > self.degree:
            raise Exception(f"This class only supports up to {self.degree}-body interactions.")
        super().__init__(atoms, trained_traj, bspline_config, uq_tolerance)

        self.pair_hash_array = self.build_pair_hash_array()

        # Reference (trained) r's things
        self.r2_gaps = dict()  # store gaps in r^2 space for 2 body


    def determine_dr_trust(self,
                           bspline_config: bspline.BSplineBasis,
                           scale: float = 1.0,
                           ) -> dict:
        """
        See superclass `R_UQ` for more details.
        """
        r_max_map = bspline_config.r_max_map
        r_min_map = bspline_config.r_min_map
        resolution_map = bspline_config.resolution_map
        pairs = bspline_config.interactions_map[2]
        dr_trust = {pair: scale * (r_max_map[pair] - r_min_map[pair]) / resolution_map[pair]
                    for pair in pairs}
        return dr_trust


    def build_pair_hash_array(self):
        # Hash matrices of pairs between self.atoms and a supercell (nRows < nCols)
        # NOTE: When working with ASE-style neighborlists, the leftmost (nRows x nRows) submatrix can be used.
        nAtoms = len(self.atoms)
        largest_r = max(self.r_max_hashed.values())
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


    def distances_by_hash(self, geom):
        """ 
        Computes lists of observed distances separated by pair hash. Each pair distance is not double counted.

        Args:
            geom (ase.atoms.Atoms): Atoms object of interest

        Returns:
            distances_lists (dict): list of observed pair distances by pair hash
        """
        smallest_r = min(self.r_min_hashed.values())
        largest_r = max(self.r_max_hashed.values())

        if any(geom.pbc):
            supercell = geometry.get_supercell(geom, r_cut=largest_r)
        else:
            supercell = geom

        dist_matr = spatial.distance.cdist(geom.get_positions(),
                                           supercell.get_positions())  # distance of atoms in geom to atoms on supercell

        # apply r_min and r_max cut masks for each pair type
        ###
        hash2r_min = np.vectorize(lambda key: self.r_min_hashed.get(key, largest_r))  # default to largest_r so that it will get filtered out later
        r_min_cut = hash2r_min(self.pair_hash_array)
        hash2r_max = np.vectorize(lambda key: self.r_max_hashed.get(key, smallest_r))  # default to smallest_r so that it will get filtered out later
        r_max_cut = hash2r_max(self.pair_hash_array)
        filter = (dist_matr > r_min_cut) & (dist_matr <= r_max_cut)
        
        dists_cut = dist_matr[filter]  # filter applied, make 1D array
        hashes_cut = self.pair_hash_array[filter]  # hashes corresponding to filtered distances
        distances_lists = composition.hash_gather(dists_cut, hashes_cut)

        return distances_lists


    def update_trained_rs(self):
        self.update_gaps()
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
        for pair_hash in self.r_max_hashed:
            r_min = self.r_min_hashed[pair_hash]
            r_max = self.r_max_hashed[pair_hash]
            rs = trained_pairs.get(pair_hash, list())
            assert not rs or rs[0] >= r_min-num_tol  # either empty or meet sanity check
            assert not rs or rs[-1] <= r_max+num_tol
            min_r_spacing = 2 * self.dr_trust[pair_hash]
            assert r_max - r_min >= min_r_spacing

            try:
                tmp_gaps = self.r2_gaps[pair_hash]
                # Temporarily bring it to r space and include buffer region
                for gap in tmp_gaps:
                    gap[0] = gap[0] ** 0.5 - self.dr_trust[pair_hash]
                    gap[1] = gap[1] ** 0.5 + self.dr_trust[pair_hash]
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
                gap[0] = (gap[0] + self.dr_trust[pair_hash]) ** 2
                gap[1] = (gap[1] - self.dr_trust[pair_hash]) ** 2
        
            self.r2_gaps[pair_hash] = tmp_gaps


    def check_r(self):
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


class R_UQ_3B(R_UQ):
    """
    A class for the r-based UQ method up to 3-body interactions to be used
    during UF3-accelerated geometry optimization. The trust radius is set to be
    the smallest knot interval of all 2- and 3-body interactions.

    See superclass `R_UQ` for more details.
    """

    def __init__(self, atoms: Atoms,
                       trained_traj: List[Atoms],
                       bspline_config: bspline.BSplineBasis,
                       uq_tolerance: float = 1.0,
                       ):
        self.degree = 3
        if bspline_config.degree > self.degree or bspline_config.degree < 3:
            raise Exception(f"This class only supports up to {self.degree}-body interactions"
                            " and at least 3-body interactions."
                            )
        super().__init__(atoms, trained_traj, bspline_config, uq_tolerance)

        #self.pair_hash_array = self.build_pair_hash_array()
        # TODO: implement similar "hash tensor" for 3 body

        # Reference (trained) r's things
        self.trained_triples = defaultdict(list)  # 3 body


    def determine_dr_trust(self,
                           bspline_config: bspline.BSplineBasis,
                           scale: float = 1.0,
                           ) -> dict:
        """
        See superclass `R_UQ` for more details.
        """
        r_max_map = bspline_config.r_max_map
        r_min_map = bspline_config.r_min_map
        resolution_map = bspline_config.resolution_map
        pairs = bspline_config.interactions_map[2]
        higher_interactions = [t for i in range(3, self.degree) for t in bspline_config.interactions_map[i]]
        highest_interactions = bspline_config.interactions_map[self.degree]

        ## Find smallest knot interval for each pair type among all interaction orders
        smallest_pair_interval = {pair: (r_max_map[pair] - r_min_map[pair]) / resolution_map[pair]
                                  for pair in pairs
                                  }  # start with 2 body
        
        for interaction in higher_interactions + highest_interactions:
            # For higher order interactions, the flat list (i.e. r_max_map[interaction],
            # r_min_map[interaction], resolution_map[interaction]) can be mapped to the
            # strictly-upper-triangular part of a 2-dimensional array whose indices
            # correspond to the 2 elements that make up the pair distance.
            #
            # Example: Given a 5-body interaction (A, B, C, D, E), there are 10 possible
            # pair distances: AB, AC, AD, AE, BC, BD, BE, CD, CE, DE. The flat r_max_map
            # for this interaction would be a list of 10 number with indices [0, 1, 2, 3,
            # 4, 5, 6, 7, 8, 9]. We can pair each number in this flat list with the
            # corresponding pair type by doing the following:
            #
            #     A | B | C | D | E
            #   --------------------
            # A | - | 0 | 1 | 2 | 3
            # B | - | - | 4 | 5 | 6
            # C | - | - | - | 7 | 8
            # D | - | - | - | - | 9
            # E | - | - | - | - | -
            #
            # So, index 5 of the flat list would map to row 1 column 3 of the 2D array,
            # where row 1 represents 'B' and column 3 represents 'D', giving the pair
            # type 'BD'.

            # Calculate intervals as flat list for this interaction
            intervals = [
                (r_max_map[interaction][i] - r_min_map[interaction][i]) / resolution_map[interaction][i]
                for i in range(len(r_max_map[interaction]))
                         ]

            # Find pair type that correspond to each element in intervals (note: there may be repeating pair tuples)
            row_indices, col_indices = np.triu_indices(len(interaction), k=1)
            mapped_pairs = [(interaction[row], interaction[col]) for row, col in zip(row_indices, col_indices)]

            # Update smallest_pair_interval
            for pair, interval in zip(mapped_pairs, intervals):
                if pair not in pairs:
                    pair = (pair[1], pair[0])
                smallest_pair_interval[pair] = interval if interval < smallest_pair_interval[pair] \
                                                        else smallest_pair_interval[pair]

        ## Build dr_trust for all the highest-order interactions (with scaling)
        dr_trust = dict()
        for interaction in highest_interactions:
            # Similar to above
            row_indices, col_indices = np.triu_indices(len(interaction), k=1)
            mapped_pairs = [(interaction[row], interaction[col]) for row, col in zip(row_indices, col_indices)]
            tmp = list()
            for pair in mapped_pairs:
                if pair not in pairs:
                    pair = (pair[1], pair[0])
                tmp.append(scale * smallest_pair_interval[pair])
            dr_trust[interaction] = tmp

        return dr_trust


    def triples_by_hash(self):
        raise NotImplementedError("Not written yet.")


    def update_trained_rs(self):
        raise NotImplementedError("Not written yet.")
        for image in self.trained_traj[self.trained_traj_len:]:
            for hash, triples in self.triples_by_hash(image):  # 2 body
                pass
                # 3b: bin the triples space into voxels with width self.dr_trust, and check the adjaent 8 voxels. Can I work in r^2 space?


    def check_r(self):
        # check based on trained_triples
        raise NotImplementedError("Not written yet.")
        return False  # not uncertain



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
