from ase.neighborlist import primitive_neighbor_list, first_neighbors
from ase.atoms import Atoms
from ase.data import atomic_numbers as ase_atomic_numbers

import numpy as np

import heapq
from typing import Iterable, List, Tuple
from collections import defaultdict
from itertools import combinations

from uf3.representation import bspline


class R_UQ_NeighborList:
    """
    A neighborlist class for the R_UQ class.

    Attributes and methods are largely copied from the ASE
    NewPrimitiveNeighborList class with some modifications:
        - `self.cutoffs` is made more versatile by allowing independent
            settings for each interaction type.
        - The definition of `self.cutoffs` and `self.skin` is changed to
            reflect LAMMPS-style neighborlists instead of ASE-style.
            - LAMMPS-style: If atom `j` is within `cutoff` of atom `i`, then
                atom `j` is a neighbor of atom `i`.
            - ASE-style: If the sphere around atom `i` with radius `cutoff`
                and the sphere around atom `j` with radius `cutoff` overlap,
                then atom `j` is a neighbor of atom `i`.
        - The `self_interaction` parameter is False by default.
        - The `bothways` parameters is True by default.
        - `self.get_neighbors()` returns a tuple of 4 arrays:
            - indices: indices of neighbors
            - offsets: offsets of neighbors
            - distance_mag: distance magnitudes of neighbors
            - distance_vec: distance vectors of neighbors
        - `self.pair2idx` attribute maps a pair (i, j) to to the index of
            `self.pair_first` and `self.pair_second` where the pair is located.
        - `self.ntuplets` and `self.tabulate_ntuplets()` for optionally
            tabulating all n-tuplets of `self.atoms` for n between 2 and
            `max_n` (both ends inclusive).


    cutoffs: list of float
        Dict of cutoff radii - one for each pair interaction type. The values
        signify the maximum distance between the CENTERs of the atoms, not the
        radius of the neighbor list sphere around an atom (which is the default
        behavior of ASE's NewPrimitiveNeighborList class)
            e.g {("O", "O"): 5.0, ("O", "H"): 1.0, ("H", "H"): 5.0}
                With this setting and `skin=0.0` and
                `Atoms('H2O', positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])`,
                the H-O interaction between atom 1 and 2 are ignored.
    skin: float
        If the sum of the largest 2 atomic displacements since the last
        neighbor list build time exceeds the skin-distance, a new build
        will be triggered. This will save some expensive rebuilds of the list,
        but extra neighbors outside the cutoff will be returned.
    sorted: bool
        Sort neighbor list (default: False)
    self_interaction: bool
        Should an atom return itself as a neighbor? (default: False)
    bothways: bool
        Return all neighbors (default: True)
    """
    def __init__(self, cutoffs, skin=0.6, sorted=False, self_interaction=False,
                 bothways=True, use_scaled_positions=False):
        self.cutoffs = {pair: cutoff + skin for pair, cutoff in cutoffs.items()}
        self.skin = skin
        self.sorted = sorted
        self.self_interaction = self_interaction
        self.bothways = bothways
        self.nupdates = 0
        self.use_scaled_positions = use_scaled_positions
        self.nneighbors = 0
        self.npbcneighbors = 0

    def update(self, atoms, max_n=0):
        """
        Make sure the list is up to date.
        
        Parameters
        ----------
        atoms : ase.atoms.Atoms
            Current geometry.
        max_n : int
            Maximum order of n-tuplets to tabulate. If less than 3, then no
            n-tuplets are tabulated.
        """

        if self.nupdates == 0:
            self.build(atoms)
            self.tabulate_ntuplets(max_n=max_n,
                                   atomic_numbers=atoms.get_atomic_numbers()
                                   )
            return True

        largest_2_displacements = np.sqrt(
                heapq.nlargest(2, ((self.positions - atoms.positions)**2).sum(1))
        )

        if ((self.pbc != atoms.pbc).any() or (self.cell != atoms.cell).any() or
                np.sum(largest_2_displacements) > self.skin):
            self.build(atoms)
            self.tabulate_ntuplets(max_n=max_n,
                                   atomic_numbers=atoms.get_atomic_numbers()
                                   )
            return True

        return False

    def build(self, atoms, max_nbins=1e6):
        """Build the list.
        """
        pbc = atoms.pbc
        cell = atoms.get_cell(complete=True)
        positions = atoms.positions
        numbers = atoms.numbers

        self.pbc = np.array(pbc, copy=True)
        self.cell = np.array(cell, copy=True)
        self.positions = np.array(positions, copy=True)

        pair_first, pair_second, offset_vec, distance_mag, distance_vec = \
            primitive_neighbor_list(
                'ijSdD', pbc, cell, positions, self.cutoffs, numbers=numbers,
                self_interaction=self.self_interaction,
                use_scaled_positions=self.use_scaled_positions,
                max_nbins=max_nbins)

        if len(positions) > 0 and not self.bothways:
            offset_x, offset_y, offset_z = offset_vec.T

            mask = offset_z > 0
            mask &= offset_y == 0
            mask |= offset_y > 0
            mask &= offset_x == 0
            mask |= offset_x > 0
            mask |= (pair_first <= pair_second) & (offset_vec == 0).all(axis=1)

            pair_first = pair_first[mask]
            pair_second = pair_second[mask]
            offset_vec = offset_vec[mask]

        if len(positions) > 0 and self.sorted:
            mask = np.argsort(pair_first * len(pair_first) +
                              pair_second)
            pair_first = pair_first[mask]
            pair_second = pair_second[mask]
            offset_vec = offset_vec[mask]

        self.pair_first = pair_first
        self.pair_second = pair_second
        assert isinstance(self.pair_first, np.ndarray)
        assert isinstance(self.pair_second, np.ndarray)
        assert np.issubdtype(self.pair_first.dtype, np.integer)
        assert np.issubdtype(self.pair_second.dtype, np.integer)
        self.offset_vec = offset_vec
        self.distance_mag = distance_mag  # values at build time
        self.distance_vec = distance_vec  # values at build time
        self.pair2idx = {pair: i for i, pair in enumerate(zip(pair_first, pair_second))}

        # Compute the index array point to the first neighbor
        self.first_neigh = first_neighbors(len(positions), pair_first)

        self.nupdates += 1

    def get_neighbors(self, a):
        """Return neighbors of atom number a.

        A list of indices, offsets, distance magnitudes, and distance vectors
        to neighboring atoms is returned.  The positions of the neighbor atoms
        can be calculated like this:

        >>>  indices, offsets, dist_mag, dist_vec = nl.get_neighbors(42)
        >>>  for i, offset in zip(indices, offsets):
        >>>      print(atoms.positions[i] + dot(offset, atoms.get_cell()))

        Notice that if get_neighbors(a) gives atom b as a neighbor,
        then get_neighbors(b) will not return a as a neighbor - unless
        bothways=True was used."""

        return (self.pair_second[self.first_neigh[a]:self.first_neigh[a + 1]],
                self.offset_vec[self.first_neigh[a]:self.first_neigh[a + 1]],
                self.distance_mag[self.first_neigh[a]:self.first_neigh[a + 1]],
                self.distance_vec[self.first_neigh[a]:self.first_neigh[a + 1]])

    def tabulate_ntuplets(self, max_n=0, atomic_numbers=None, algo=1):
        """
        Tabulate all n-tuplets of `self.atoms` given the current neighbor list
        for n between 2 and `max_n` (both ends inclusive).

        Updates `self.ntuplets` attribute:
            - `self.ntuplets['which_d']`
                - key: n (2 <= n <= max_n)
                - value: list of lists of pair indices (i, j) for each side
                    of the n-tuplet
                    - e.g. For triplets (0, 1, 2) and (2, 1, 3):
                        [[(0, 1), (0, 2), (1, 2)], [(2, 1), (2, 3), (1, 3)]]
                    - For n=2, just a tuple of pair indices (i, j)
                    - The length of the outer tuple is n*(n-1)/2
            - `self.ntuplets['type']`
                - key: n (2 <= n <= max_n)
                - value: list of tuple of n-tuplet types
                    - e.g. For triplets (0, 1, 2) and (2, 1, 3) where the
                        atomic numbers are given by
                        {0: 1, 1: 6, 2: 6, 3: 8}:
                            [(1, 6, 6), (6, 6, 8)]
                    - The length of the tuple is n
        
        If `atomic_numbers` is provided, then the pairs and higher-order
        neighbors are sorted by atomic numbers, and `self.ntuplets['type']` is
        made. If not, then `self.ntuplets['type']` is not made, and the pairs
        and higher-order neighbors are not sorted.

        Parameters
        ----------
        max_n : int
            Maximum order of n-tuplets to tabulate. If less than 2, then no
            n-tuplets are tabulated.
        atomic_numbers : np.ndarray
            To sort by atomic number
        algo : int
            Algorithm to use. (1, 2, or 3)
        """
        if max_n < 2:
            self.ntuplets = None
            return
        self.ntuplets = dict()
        self.ntuplets['which_d'] = dict()
        self.ntuplets['type'] = dict()
        cache = dict()  # used to generate higher-order tuplets from lower.
                        # will store tuplets sorted by index (not atomic
                        # number) for algorithmic efficiency.

        # Pairs first (n=2)
        pairs = np.vstack((self.pair_first, self.pair_second)).T
        cache[2] = [pairs]
        unique_pair_mask = pairs[:, 0] < pairs[:, 1]
        pairs = pairs[unique_pair_mask]  # rebinding. Does not change cache[2]
        if atomic_numbers is not None:
            sort_priority = atomic_numbers[pairs]
            sort_indices = np.argsort(sort_priority, axis=1)
            sorted_pairs = np.take_along_axis(pairs,
                                              sort_indices,
                                              axis=1)  # sorted by atomic number
            sorted_atomic_numbers = np.take_along_axis(sort_priority,
                                                       sort_indices,
                                                       axis=1)
            sorted_atomic_symbols = [tuple(row) for row in
                                     sorted_atomic_numbers.tolist()]
            self.ntuplets['type'][2] = sorted_atomic_symbols
            self.ntuplets['which_d'][2] = [tuple(row) for row in
                                           sorted_pairs.tolist()]
        else:
            self.ntuplets['which_d'][2] = [tuple(row) for row in pairs.tolist()]

        # Triplets (n=3)
        if max_n < 3:
            return
        if algo == 1:
            n = 3
            if True:
                self.ntuplets['which_d'][n] = list()
                if atomic_numbers is not None:
                    self.ntuplets['type'][n] = list()
                cache[n] = list()
                for i in range(len(self.first_neigh) - 1):
                    neighbors, _, _, _ = self.get_neighbors(i)
                    if len(neighbors) < n-1:
                        cache[n].append(None)
                        continue
                    duplicated_lower = np.repeat(neighbors,
                                                 len(neighbors),
                                                 axis=0)
                    tiled_neighbors = np.tile(neighbors,
                                              len(neighbors))
                    # by induction, the last column has the largest values
                    unique_mask = tiled_neighbors > duplicated_lower
                    neighs = np.column_stack((duplicated_lower[unique_mask],
                                              tiled_neighbors[unique_mask]))
                    cache[n].append(neighs)
                    if atomic_numbers is not None:
                        sort_priority = atomic_numbers[neighs]
                        sort_indices = np.argsort(sort_priority, axis=1)
                        sorted_neighs = np.take_along_axis(neighs,
                                                        sort_indices,
                                                        axis=1)
                        new_tuplets = np.concatenate((i*np.ones((len(neighs), 1), dtype=int),
                                                    sorted_neighs),
                                                    axis=1)  # insert center
                        ds = [list(combinations(r, 2))
                                    for r in new_tuplets.tolist()]
                        self.ntuplets['which_d'][n].extend(ds)
                        sorted_atomic_numbers = np.take_along_axis(sort_priority,
                                                                sort_indices,
                                                                axis=1)
                        _i = atomic_numbers[i]
                        sorted_atomic_numbers = \
                            np.concatenate((_i*np.ones((len(sorted_neighs), 1), dtype=int),
                                                sorted_atomic_numbers),
                                                axis=1)  # insert center
                        sorted_atomic_symbols = [tuple(row) for row in
                                                sorted_atomic_numbers.tolist()]
                        self.ntuplets['type'][n].extend(sorted_atomic_symbols)
                    else:
                        neighs = np.concatenate((i*np.ones((len(neighs), 1), dtype=int),
                                                neighs),
                                                axis=1)  # insert center
                        ds = [list(combinations(r, 2)) for r in neighs.tolist()]
                        self.ntuplets['which_d'][n].extend(ds)
            for n in range(4, max_n+1):
                self.ntuplets['which_d'][n] = list()
                if atomic_numbers is not None:
                    self.ntuplets['type'][n] = list()
                cache[n] = list()
                assert len(cache[n-1]) == len(self.first_neigh) - 1
                for i, lower_tuplet_neighs in enumerate(cache[n-1]):
                    if lower_tuplet_neighs is None or not len(lower_tuplet_neighs):
                        cache[n].append(None)
                        continue
                    neighbors, _, _, _ = self.get_neighbors(i)
                    if len(neighbors) < n-1:
                        cache[n].append(None)
                        continue
                    duplicated_lower = np.repeat(lower_tuplet_neighs,
                                                 len(neighbors),
                                                 axis=0)
                    tiled_neighbors = np.tile(neighbors,
                                              len(lower_tuplet_neighs))
                    # by induction, the last column has the largest values
                    unique_mask = tiled_neighbors > duplicated_lower[:, -1]
                    neighs = np.column_stack((duplicated_lower[unique_mask],
                                              tiled_neighbors[unique_mask]))
                    cache[n].append(neighs)
        elif algo == 2:
            for n in range(3, max_n+1):
                new_tuplets = list()
                for i in range(len(self.first_neigh) - 1):
                    neighbors, _, _, _ = self.get_neighbors(i)
                    if len(neighbors) < n-1:
                        continue
                    new_tuplet_neighs = np.array(
                        list(combinations(neighbors, n-1)), dtype=int
                        )
                    new_tuplet = np.column_stack((
                        i*np.ones((len(new_tuplet_neighs), 1), dtype=int),
                        new_tuplet_neighs
                        ))
                    new_tuplets.append(new_tuplet)
                try:
                    new_tuplets = np.concatenate(new_tuplets, axis=0)
                except ValueError:  # no new tuplets
                    break
                if atomic_numbers is None:
                    ds = [list(combinations(r, 2)) for r in
                                new_tuplets.tolist()]
                    self.ntuplets['which_d'][n] = ds
                else:
                    sort_priority = atomic_numbers[new_tuplets[:, 1:]]
                    sort_indices = np.argsort(sort_priority, axis=1)
                    new_tuplets[:, 1:] = \
                        np.take_along_axis(new_tuplets[:, 1:],
                                           sort_indices,
                                           axis=1)
                    # same lines as in if, but this must be done in the middle
                    # of the else block to reduce data movement in/out of cache
                    ds = [list(combinations(r, 2)) for r in
                            new_tuplets.tolist()]
                    self.ntuplets['which_d'][n] = ds
                    sorted_atomic_numbers = np.zeros((len(new_tuplets), n),
                                                        dtype=int)
                    sorted_atomic_numbers[:, 1:] = \
                        np.take_along_axis(sort_priority,
                                           sort_indices,
                                           axis=1)
                    sorted_atomic_numbers[:, 0] = \
                        atomic_numbers[new_tuplets[:, 0]]
                    sorted_atomic_symbols = [tuple(row) for row in
                                            sorted_atomic_numbers.tolist()]
                    self.ntuplets['type'][n] = sorted_atomic_symbols





class R_UQ:
    """
    A class for the r-based UQ method to be used during UF3-accelerated
    geometry optimization.

    Intended to be instantiated only once for a single optimization job.
    
    This assumes that the atomic order and composition as well as the following
    attributes of `bspline_config` will remain constant:
        - r_min_map
        - r_max_map
        - chemical_system

    Other assumptions include:
        - All knot intervals for a given interaction are uniform.
        - The knot ranges of a given n-body interaction are
            included within the knot ranges of the corresponding m-body
            interactions for m < n.
        - The knot intervals of a given n-body interaction are smaller than
            the knot intervals of the corresponding m-body interactions for
            m < n.
            - Does not break anything, but is more inefficient if this is not
                the case.

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
        if not isinstance(atoms.positions, np.ndarray):
            raise TypeError("The `positions` attribute of `atoms` must be a "
                            "numpy array.")
        self.trained_traj = trained_traj
        self.bspline_config = bspline_config
        if set(atoms.get_chemical_symbols()) > \
            set(self.bspline_config.element_list):
            raise Exception("Element list mismatch between `bspline_config` and"
                            " `atoms`.")
        self.degree = self.bspline_config.degree
        if self.degree > 3:
            raise NotImplementedError("Degree 4 and higher are not supported "
                                      "yet.")
        self.interactions_map = {d: [self.symbols2numbers(interaction)
                                     for interaction in
                                     self.bspline_config.interactions_map[d]]
                                 for d in range(2, self.degree+1)}
        self.r_min_map = {self.symbols2numbers[interaction]: value
                          for interaction, value in
                          self.bspline_config.r_min_map.items()}
        self.r_max_map = {self.symbols2numbers[interaction]: value
                          for interaction, value in
                          self.bspline_config.r_max_map.items()}
        self.resolution_map = {self.symbols2numbers[interaction]: value
                               for interaction, value in
                               self.bspline_config.resolution_map.items()}

        # Neighbor list things
        self.dr_trust = self.determine_dr_trust(scale=uq_tolerance)
        self.nl_cutoffs = {pair: self.r_max_map[pair]
                           for pair in self.interactions_map[2]}
        dr_trust_2b = {pair: self.dr_trust[pair]
                       for pair in self.interactions_map[2]}
        max_dr_trust_2b = max(dr_trust_2b.values())
        self.nl = R_UQ_NeighborList(cutoffs=self.nl_cutoffs,
                                    skin=max_dr_trust_2b,  # arbitrary choice
                                    sorted=False,
                                    self_interaction=False,
                                    bothways=True,
                                    use_scaled_positions=False,
                                    )

        # Training data things
        self.r_gaps = self.initialize_r_gaps()  # 2 body: gaps in r space
        self.data_voxels = self.initialize_data_voxels()  # higher body

        self.trained_traj_len = 0  # how many images in self.trained_traj have
                                   # been considered in update_trained_rs()

    @staticmethod
    def symbols2numbers(symbols: Iterable[str],
                        ) -> Iterable[int]:
        """
        Converts a tuple of chemical symbols to a tuple of atomic numbers.

        Parameters
        ----------
        symbols : Iterable[str]
            Tuple of chemical symbols.

        Returns
        -------
        numbers : Iterable[int]
            Tuple of atomic numbers.
        """
        return tuple(ase_atomic_numbers[symbol] for symbol in symbols)

    def determine_dr_trust(self,
                           scale: float = 1.0,
                           ) -> dict:
        """
        Determines the trust radius of how far a pair distance can be away from
        trained distances before a "high uncertainty" is triggered.

        Parameters
        ----------
        scale : float
            A scaling factor for the trust radius. The default value is 1.0.

        Returns
        -------
        dr_trust : dict
            Trust radius for each highest-degree interaction of how far a pair
            distance can be away from trained distances before a
            "high uncertainty" is triggered.
        """
        pairs = self.interactions_map[2]
        dr_trust = {pair: scale * (self.r_max_map[pair] - self.r_min_map[pair])
                    / self.resolution_map[pair]
                    for pair in pairs}
        for degree in range(3, self.degree+1):
            interactions = self.interactions_map[degree]
            for interaction in interactions:
                rs_max = self.r_max_map[interaction]
                rs_min = self.r_min_map[interaction]
                resolutions = self.resolution_map[interaction]
                dr_trust[interaction] = [
                    scale * (rs_max[i] - rs_min[i]) / resolutions[i]
                    for i in range(len(resolutions))
                    ]
        return dr_trust

    def initialize_r_gaps(self):
        """
        Initializes the r gaps for 2-body interactions.
        """
        r_gaps = dict()
        pairs = self.interactions_map[2]
        for pair in pairs:
            lower_bound = self.r_min_map[pair]
            upper_bound = self.r_max_map[pair]
            r_gaps[pair] = [[lower_bound, upper_bound]]
        return r_gaps

    def initialize_data_voxels(self):
        """
        Initializes the data voxels for higher-body interactions.
        The mapping of r to voxel index is given by `r_to_voxel_idx()`.
        """
        data_voxels = dict()
        for degree in range(3, self.degree+1):
            d = dict()
            interactions = self.interactions_map[degree]
            for interaction in interactions:
                shape = list()
                for r_min, r_max, spacing in zip(self.r_min_map[interaction],
                                                 self.r_max_map[interaction],
                                                 self.dr_trust[interaction]):
                    shape.append(self.r_to_voxel_idx(r_min, r_max, spacing) + 1)
                d[interaction] = np.zeros(shape, dtype=bool)
            data_voxels[degree] = d
        return data_voxels

    @staticmethod
    def r_to_voxel_idx(r: float | np.ndarray,
                       r_max: float | np.ndarray,
                       spacing: float | np.ndarray):
        """
        Mapping of r to voxel index.

        r=r_max is mapped to index 0, and r<r_max is mapped to a nonnegative
        index that increases as r decreases.

        Parameters
        ----------
        r : float | np.ndarray
            Pair distance.
        r_max : float | np.ndarray
            Maximum pair distance in the desired axis.
        spacing : float | np.ndarray
            Voxel spacing in the desired axis.
        """
        tmp = np.floor( (r_max - r) / spacing )
        return tmp.astype(int)

    def update_trained_rs(self):
        """
        Updates the record of training data (distances) on file.
        """
        lower_traj_idx = self.trained_traj_len
        upper_traj_idx = len(self.trained_traj) 
        for image in self.trained_traj[lower_traj_idx:upper_traj_idx]:
            nl = R_UQ_NeighborList(cutoffs=self.nl_cutoffs,
                                   skin=0.0,
                                   sorted=False,
                                   self_interaction=False,
                                   bothways=True,
                                   use_scaled_positions=False,
                                   )
            nl.update(image)
            ntuplets_generator = self.ntuplets_generator(self.degree,
                                                         recalc_distances=False,
                                                         return_type=True,
                                                         )
            for ntuplet, distances in ntuplets_generator:
                if len(ntuplet) == 2:
                    self.update_gaps(ntuplet, distances)
                else:
                    self.update_voxels(ntuplet, distances)   # XXX: take into account self.bspline_config.symmetry in voxel

        self.trained_traj_len = len(self.trained_traj)

    def update_gaps(self, nl: R_UQ_NeighborList):
        """
        Fill gaps with new r's (2-body data space).

        Gaps in r space for `pair`:
          |---------*===============*---------|
              |--* are buffer regions of length `self.dr_trust[pair]`
              *==* is the core region
              A new r is uncertain if it is in the core region

        Parameters
        ----------
        nl : R_UQ_NeighborList
            Neighbor list object for the current geometry.
        """
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

    def check_r(self, updated):
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
        updated = self.nl.update(self.atoms)  # build new neighbor list if needed

        # If there is new training data
        if len(self.trained_traj) != self.trained_traj_len:
            self.update_trained_rs()
        
        # Check epistemic uncertainty
        return self.check_r(updated)


def old_ntuplets_generator(nl: R_UQ_NeighborList,
                       max_n: int,
                       positions: np.ndarray = None,
                       ) -> Iterable[List[Tuple]]:
    """
    Generates n-tuplets of neighbors for each atom in `atoms` for n between
    2 and `max_n` (both ends inclusive).

    Parameters
    ----------
    nl: R_UQ_NeighborList
        Neighbor list object.
    max_n : int
        The maximum number of neighbors in the higher-order n-tuplet.
    positions : np.ndarray
        If provided, then this is used to recalculate the distances and
        distance vectors of the neighbor list.

    Yields
    ------
    ntuplet : List[Tuple[Tuple[int, ...], Tuple[int, ...]]]
        A list of n-tuplet of neighbor indices and distances for each n.
    """
    # Make list of distance magnitudes and vectors from positions if needed
    if positions is None:
        dist_mag = nl.distance_mag
        dist_vec = nl.distance_vec
    else:
        dist_vec = positions[nl.pair_second] + \
                    np.dot(nl.offset_vec, nl.cell) - \
                    positions[nl.pair_first]
        dist_mag = np.linalg.norm(dist_vec, axis=1)
    
    # Yield all pairs first
    dist_cache = dict()
    vec_cache = dict()
    ntuplet = list()
    for i, j, d, v in zip(nl.pair_first, nl.pair_second,
                        dist_mag, dist_vec):
        dist_cache[(i, j)] = d
        vec_cache[(i, j)] = v
        if i > j:  # avoid double counting (self.nl.bothways=True)
            continue
        ntuplet.append( ((i, j), (d,)) )
    
    # Generate triplets
    if max_n > 2:
        for i, j in zip(nl.pair_first, nl.pair_second):
            for k in nl.get_neighbors(i)[0]:
                if j >= k:  # avoid self and double counting neighbors
                    continue
                v_jk = vec_cache[(i, k)] - vec_cache[(i, j)]
                d_jk = np.linalg.norm(v_jk)
                d_ij = dist_cache[(i, j)]
                d_ik = dist_cache[(i, k)]
                yield (i, j, k), (d_ij, d_ik, d_jk)

    # Generate higher-body interactions
    for n in range(4, max_n+1):
        raise NotImplementedError("max_n>3 is not supported yet.")


def ntuplets_generator(nl: R_UQ_NeighborList,
                       max_n: int,
                       positions: np.ndarray = None,
                       ) -> Iterable[List[Tuple]]:
    """
    Generates n-tuplets of neighbors for each atom in `atoms` for n between
    2 and `max_n` (both ends inclusive).

    Parameters
    ----------
    nl: R_UQ_NeighborList
        Neighbor list object.
    max_n : int
        The maximum number of neighbors in the higher-order n-tuplet.
    positions : np.ndarray
        If provided, then this is used to recalculate the distances and
        distance vectors of the neighbor list.

    Yields
    ------
    ntuplet : List[Tuple[Tuple[int, ...], Tuple[int, ...]]]
        A list of n-tuplet of neighbor indices and distances for each n.
    """
    # Make list of distance magnitudes and vectors from positions if needed
    if positions is None:
        dist_mag = nl.distance_mag
        dist_vec = nl.distance_vec
    else:
        dist_vec = positions[nl.pair_second] + \
                    np.dot(nl.offset_vec, nl.cell) - \
                    positions[nl.pair_first]
        dist_mag = np.linalg.norm(dist_vec, axis=1)
    
    # Create distance and vector caches
    dist_cache = dict()
    vec_cache = dict()
    for i, j, d, v in zip(nl.pair_first, nl.pair_second,
                        dist_mag, dist_vec):
        dist_cache[(i, j)] = d
        vec_cache[(i, j)] = v

    # Yield all pairs first 
    #pairs = np.
    
    # Generate triplets
    if max_n > 2:
        for i, j in zip(nl.pair_first, nl.pair_second):
            for k in nl.get_neighbors(i)[0]:
                if j >= k:  # avoid self and double counting neighbors
                    continue
                v_jk = vec_cache[(i, k)] - vec_cache[(i, j)]
                d_jk = np.linalg.norm(v_jk)
                d_ij = dist_cache[(i, j)]
                d_ik = dist_cache[(i, k)]
                yield (i, j, k), (d_ij, d_ik, d_jk)

    # Generate higher-body interactions
    for n in range(4, max_n+1):
        raise NotImplementedError("max_n>3 is not supported yet.")


if __name__ == '__main__':
    from ase.atoms import Atoms
    from uf3.data import composition
    from uf3.representation import bspline

    '''
    atoms = Atoms('HOH', positions=[[0, 0, 0], [0, 1, 0], [14, 0, 0]],
                  cell=[15, 15, 15], pbc=True)
    cutoffs = {("O", "O"): 5.0, ("O", "H"): 0.82, ("H", "H"): 5.0}
    nl = R_UQ_NeighborList(cutoffs=cutoffs,
                           skin=0.6)
    nl.update(atoms, max_n=3)
    print(f"type(nl.pair_first) = {type(nl.pair_first)}")
    print(f"nl.pair_first = \n{nl.pair_first}")
    print(f"nl.pair_second = \n{nl.pair_second}")
    print(f"nl.first_neigh = \n{nl.first_neigh}")
    print(f"nl.pair2idx = \n{nl.pair2idx}")
    print(f"nl.offset_vec = \n{nl.offset_vec}")
    print(f"nl.distance_mag = \n{nl.distance_mag}")
    print(f"nl.distance_vec = \n{nl.distance_vec}")
    print(f"nl.get_neighbors(1) = \n{nl.get_neighbors(1)}")
    print(f"nl.ntuplets = \n{nl.ntuplets}")
    print("===")

    atoms = Atoms("COHHCOOHH")
    atoms.positions = np.array([[0, 0, 0],
                                [2, 2, 0],
                                [0, 2, 0],
                                [2, 0, 0],
                                [6, 0, 0],
                                [12, 0, 0],
                                [7, 7, 0],
                                [0, 7, 0],
                                [7, 8, 0]])
    atoms.cell = np.array([[15, 0, 0], [0, 15, 0], [0, 0, 15]])
    atoms.pbc = True
    cutoffs = {('O', 'C'): 5.0,
               ('O', 'H'): 3.5,
               ('O', 'O'): 4.0,
               ('H', 'H'): 2.0,
               ('H', 'C'): 3.0,
               ('C', 'C'): 7.0}
    nl = R_UQ_NeighborList(cutoffs=cutoffs,
                           skin=0.0)
    #nl.update(atoms, max_n=3)
    nl.build(atoms)
    nl.tabulate_ntuplets(max_n=3,
                         atomic_numbers=atoms.get_atomic_numbers(),
                         algo=3
                         )
    #print(len(nl.ntuplets['which_d'][2]))
    #print(len(nl.ntuplets['which_d'][3]))
    #print(f"nl.pair_first = \n{nl.pair_first}")
    #print(f"nl.pair_second = \n{nl.pair_second}")
    for t, s in zip(nl.ntuplets['which_d'][2], nl.ntuplets['type'][2]):
        print(f"{t}, {s}")
    for t, s in zip(nl.ntuplets['which_d'][3], nl.ntuplets['type'][3]):
        print(f"{t}, {s}")
    print("===")
    '''
    atoms = Atoms("Pt1000C100H100", cell=[30, 30, 30], pbc=True)
    np.random.seed(1)
    atoms.positions = np.random.uniform(0, 30, (1200, 3))
    cutoffs = {('Pt', 'Pt'): 6.0,
               ('Pt', 'C'): 5.5,
               ('Pt', 'H'): 5.0,
               ('C', 'C'): 5.5,
               ('C', 'H'): 5.0,
               ('H', 'H'): 4.0}
    nl = R_UQ_NeighborList(cutoffs=cutoffs,
                           skin=0.0)
    #nl.update(atoms, max_n=3)
    nl.build(atoms)
    nl.tabulate_ntuplets(max_n=3,
                         atomic_numbers=atoms.get_atomic_numbers(),
                         algo=3
                         )
    #print(len(nl.ntuplets['which_d'][2]))
    #print(len(nl.ntuplets['which_d'][3]))
    #print(len(nl.ntuplets['which_d'][4]))
    #for t in nl.ntuplets['which_d'][2]:
    #    print(f"{t}")
    #for t in nl.ntuplets['which_d'][3]:
    #    print(f"{t}")
    #'''