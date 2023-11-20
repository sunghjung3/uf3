from ase.neighborlist import primitive_neighbor_list, first_neighbors
from ase.atoms import Atoms
from ase.data import atomic_numbers as ase_atomic_numbers

import numpy as np

import heapq
from typing import Iterable, List
from collections import defaultdict
from itertools import combinations
import math

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
        - The `self_interaction` parameter is fixed to False.
        - The `bothways` parameter is fixed to True.
        - `self.pair_distances` are added to cache distances between pairs of
            atoms.
            - key: atom number of center atom
            - value: dict
                - key: tuple of (atom number of neighbor atom, tuple of offset)
                - value: distance between center and the neighbor.
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
    """
    def __init__(self, cutoffs, skin=0.6, sorted=False,
                 use_scaled_positions=False):
        self.cutoffs = {pair: cutoff + skin for pair, cutoff in cutoffs.items()}
        self.skin = skin
        self.sorted = sorted
        self.self_interaction = False
        self.bothways = True
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
            mask = self.pair_filter(pair_first, pair_second, offset_vec)
            pair_first = pair_first[mask]
            pair_second = pair_second[mask]
            offset_vec = offset_vec[mask]
            distance_mag = distance_mag[mask]
            distance_vec = distance_vec[mask]

        if len(positions) > 0 and self.sorted:
            mask = np.argsort(pair_first * len(pair_first) +
                              pair_second)
            pair_first = pair_first[mask]
            pair_second = pair_second[mask]
            offset_vec = offset_vec[mask]
            distance_mag = distance_mag[mask]
            distance_vec = distance_vec[mask]

        self.pair_first = pair_first
        self.pair_second = pair_second
        assert isinstance(self.pair_first, np.ndarray)
        assert isinstance(self.pair_second, np.ndarray)
        assert np.issubdtype(self.pair_first.dtype, np.integer)
        assert np.issubdtype(self.pair_second.dtype, np.integer)
        self.offset_vec = offset_vec

        # Distance cache
        self.pair_distances = defaultdict(lambda: defaultdict(float))
        for first, second, offset, d in zip(self.pair_first,
                                            self.pair_second,
                                            self.offset_vec,
                                            distance_mag):
            t = (second, tuple(offset))
            self.pair_distances[first][t] = d

        # Compute the index array point to the first neighbor
        self.first_neigh = first_neighbors(len(positions), pair_first)

        self.nupdates += 1

    @staticmethod
    def pair_filter(pair_first, pair_second, offset_vec):
        """
        Filter mask to remove double counting, considering ghost
        atoms.
        
        Parameters
        ----------
        pair_first: np.ndarray
            Atom numbers of center atoms. Refer to
            `ase.neighborlist.primitive_neighbor_list()`.
        pair_second: np.ndarray
            Atom numbers of neighbor atoms. Refer to
            `ase.neighborlist.primitive_neighbor_list()`.
        offset_vec: np.ndarray
            Offset vectors of neighbor atoms. Refer to
            `ase.neighborlist.primitive_neighbor_list()`.

        Returns
        -------
        mask: np.ndarray
            Mask to remove double counting.
        """
        offset_x, offset_y, offset_z = offset_vec.T

        mask = offset_z > 0
        mask &= offset_y == 0
        mask |= offset_y > 0
        mask &= offset_x == 0
        mask |= offset_x > 0
        mask |= (pair_first <= pair_second) & (offset_vec == 0).all(axis=1)
        return mask

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
                )

    def tabulate_ntuplets(self, max_n=0, atomic_numbers=None, algo=1):
        """
        Tabulate all n-tuplets of `self.atoms` given the current neighbor list
        for n between 2 and `max_n` (both ends inclusive).

        Updates `self.ntuplets` attribute:
            - `self.ntuplets['atoms']`
                - key: n (2 <= n <= max_n)
                - value: row-wise array of n-tuplets.
                    - e.g. For triplets (0, 1, 2) and (2, 1, 3):
                        np.array([[0, 1, 2], [2, 1, 3]])
            - `self.ntuplets['offsets']`
                - key: n (2 <= n <= max_n)
                - value: array of neighbor's periodic offsets given by each
                    n-tuplet in `self.ntuplets['atoms']`
            - `self.ntuplets['type']`
                - key: n (2 <= n <= max_n)
                - value: list of tuple of n-tuplet types, where each tuple is
                    sorted by atomic number.
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
        nneighbors = np.diff(self.first_neigh)
        max_n_ceil = max(nneighbors) + 1  # max_n cannot be larger than this
        max_n = min(max_n, max_n_ceil)

        if max_n < 2:
            self.ntuplets = None
            return
        self.ntuplets = dict()
        self.ntuplets['atoms'] = dict()
        self.ntuplets['offsets'] = dict()
        self.ntuplets['type'] = dict()

        # Pairs first (n=2)
        unfiltered_pairs = np.vstack((self.pair_first, self.pair_second)).T
        unfiltered_offsets = self.offset_vec
        pair_mask = \
            self.pair_filter(self.pair_first,
                             self.pair_second,
                             self.offset_vec)
        pairs = unfiltered_pairs[pair_mask]
        offsets = unfiltered_offsets[pair_mask]
        if atomic_numbers is not None:
            sort_priority = atomic_numbers[pairs]
            sort_indices = np.argsort(sort_priority, axis=1)
            sorted_atomic_numbers = np.take_along_axis(sort_priority,
                                                       sort_indices,
                                                       axis=1)
            self.ntuplets['type'][2] = [tuple(row) for row in
                                        sorted_atomic_numbers.tolist()]
            sorted_pairs = np.take_along_axis(pairs,
                                              sort_indices,
                                              axis=1)  # sorted by atomic number
            self.ntuplets['atoms'][2] = sorted_pairs
            offsets_to_negate = (sort_indices == [1, 0]).all(axis=1)
            offsets[offsets_to_negate] *= -1  # if pair is reversed, negate offset
            self.ntuplets['offsets'][2] = offsets
        else:
            self.ntuplets['atoms'][2] = pairs
            self.ntuplets['offsets'][2] = offsets


        # Higher-order (n > 2)
        if max_n < 3:
            return

        if algo == 1:
            # As the neighbor atom may appear more than once in a given neighbor
            # list with a sufficiently large cutoff, use encoded neighbor atoms
            # to build higher-order tuplets and convert back.
            # The simplest encoding is to use the index of the neighbor atom in
            # `self.pair_second` as the encoded neighbor atom.
            encoded_neighbors = \
                np.split(np.arange(self.first_neigh[-1]), self.first_neigh[1:-1])
                # encoded_neighbors[i] belongs to center atom i

            cache = dict()  # used to generate higher-order tuplets from lower.
                            # will store tuplets neighbors sorted by encoded
                            # index (not atomic number) for algorithmic
                            # efficiency.
                            # key: n (2 <= n <= max_n)
                            # value: list of np.ndarray of shape (x, n-1)
                            #   - cache[i] belongs to center atom i
            cache[2] = [subarray.reshape(-1, 1) for subarray in
                        encoded_neighbors]

            for n in range(3, max_n+1):
                # arrays in these lists will be concatenated at the end
                self.ntuplets['atoms'][n] = list()
                self.ntuplets['offsets'][n] = list()

                if atomic_numbers is not None:
                    self.ntuplets['type'][n] = list()
                cache[n] = list()
                assert len(cache[n-1]) == len(self.first_neigh) - 1

                for i, lower_tuplet_neighs in enumerate(cache[n-1]):
                    if lower_tuplet_neighs is None or not len(lower_tuplet_neighs):
                        cache[n].append(None)
                        continue
                    neighbors = encoded_neighbors[i]
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
                    cache[n].append(neighs)  # still encoded

                    # unencode
                    offsets = self.offset_vec[neighs]
                    neighs = self.pair_second[neighs]

                    if atomic_numbers is not None:
                        # indices for sorting
                        sort_priority = atomic_numbers[neighs]
                        sort_indices = np.argsort(sort_priority, axis=1)

                        # sort atoms
                        sorted_neighs = np.take_along_axis(neighs,
                                                           sort_indices,
                                                           axis=1)
                        new_tuplets = np.concatenate((i*np.ones((len(neighs), 1), dtype=int),
                                                     sorted_neighs),
                                                     axis=1)  # insert center
                        self.ntuplets['atoms'][n].append(new_tuplets)

                        # sort atomic numbers
                        sorted_atomic_numbers = np.take_along_axis(sort_priority,
                                                                   sort_indices,
                                                                   axis=1)
                        _i = atomic_numbers[i]
                        sorted_atomic_numbers = \
                            np.concatenate((_i*np.ones((len(sorted_neighs), 1), dtype=int),
                                                sorted_atomic_numbers),
                                                axis=1)  # insert center
                        sorted_atomic_numbers = [tuple(row) for row in
                                                 sorted_atomic_numbers.tolist()]
                        self.ntuplets['type'][n].extend(sorted_atomic_numbers)

                        # sort offsets
                        sorted_offsets = np.take_along_axis(offsets,
                                                            sort_indices[:, :, np.newaxis],
                                                            axis=1)
                        self.ntuplets['offsets'][n].append(sorted_offsets)
                    else:
                        new_tuplets = np.concatenate((i*np.ones((len(neighs), 1), dtype=int),
                                                     neighs),
                                                     axis=1)  # insert center
                        self.ntuplets['atoms'][n].extend(new_tuplets)
                        self.ntuplets['offsets'][n].extend(offsets)

                self.ntuplets['atoms'][n] = \
                    np.concatenate(self.ntuplets['atoms'][n], axis=0)
                self.ntuplets['offsets'][n] = \
                    np.concatenate(self.ntuplets['offsets'][n], axis=0)

        '''
        HAVE NOT BEEN FIXED YET
        if algo == 3:  # NOTE: in my opinion, this should be better than algo=1, but profiling suggests otherwise
            cache = dict()  # used to generate higher-order tuplets from lower.
                            # will store tuplets neighbors sorted by index (not
                            # atomic number) for algorithmic efficiency.
                            # key: n (2 <= n <= max_n)
                            # value: list of np.ndarray of shape (x, n)
            cache[2] = [
                unfiltered_pairs[self.first_neigh[a]:self.first_neigh[a + 1], :]
                for a in range(len(self.first_neigh) - 1)
                ]

            for n in range(3, max_n+1):
                new_tuplets = list()
                for lower_tuplet in cache[n-1]:
                    if not len(lower_tuplet):
                        continue
                    i = lower_tuplet[0, 0]  # center atom
                    if nneighbors[i] < n-1:
                        continue
                    neighbors, _, _, _ = self.get_neighbors(i)
                    duplicated_lower = np.repeat(lower_tuplet,
                                                 len(neighbors),
                                                 axis=0)
                    tiled_neighbors = np.tile(neighbors,
                                              len(lower_tuplet))
                    # by induction, the last column has the largest values
                    unique_mask = tiled_neighbors > duplicated_lower[:, -1]
                    new_tuplet = np.column_stack((duplicated_lower[unique_mask],
                                                  tiled_neighbors[unique_mask]))
                    new_tuplets.append(new_tuplet)
                del cache[n-1]
                cache[n] = new_tuplets
                new_tuplets = np.concatenate(cache[n], axis=0)
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

        elif algo == 2:
            for n in range(3, max_n+1):
                new_tuplets = list()
                for i in np.where(nneighbors >= n-1)[0]:  # only i with enough
                                                          # neighbors
                    neighbors, _, _, _ = self.get_neighbors(i)
                    new_tuplet = np.empty((math.comb(len(neighbors), n-1), n),
                                           dtype=int)
                    new_tuplet[:, 1:] = np.array(
                        list(combinations(neighbors, n-1)), dtype=int
                        )
                    new_tuplet[:, 0] = i
                    new_tuplets.append(new_tuplet)
                new_tuplets = np.concatenate(new_tuplets, axis=0)
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
        '''


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
        self.r_min_map = {self.symbols2numbers(interaction): np.array(value)
                          for interaction, value in
                          self.bspline_config.r_min_map.items()}
        self.r_max_map = {self.symbols2numbers(interaction): np.array(value)
                          for interaction, value in
                          self.bspline_config.r_max_map.items()}
        self.resolution_map = {self.symbols2numbers(interaction): np.array(value)
                               for interaction, value in
                               self.bspline_config.resolution_map.items()}
        self.symmetry_3b = {self.symbols2numbers(key): value for key, value in
                            self.bspline_config.symmetry.items()}

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
                                    use_scaled_positions=False,
                                    )

        # Training data things
        self.data_voxels = self.initialize_data_voxels()

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

    def initialize_data_voxels(self):
        """
        Initializes the data voxels.
        The mapping of r to voxel index is given by `r_to_voxel_idx()`.
        """
        data_voxels = dict()

        # 2-body
        data_voxels[2] = dict()
        for pair in self.interactions_map[2]:
            r_min = self.r_min_map[pair]
            r_max = self.r_max_map[pair]
            spacing = self.dr_trust[pair]
            data_voxels[2][pair] = np.zeros(self.r_to_voxel_idx(r_min, r_max,
                                                                 spacing) + 1,
                                            dtype=bool)

        # higher-order
        for degree in range(3, self.degree+1):
            d = dict()
            interactions = self.interactions_map[degree]
            for interaction in interactions:
                shape = list()
                d[interaction] = np.zeros(self.r_to_voxel_idx(self.r_min_map[interaction],
                                                              self.r_max_map[interaction],
                                                              self.dr_trust[interaction]) + 1,
                                             dtype=bool)
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
                                   use_scaled_positions=False,
                                   )
            nl.update(image, max_n=self.degree)
            self.update_voxels(nl)
        self.trained_traj_len = len(self.trained_traj)

    def voxel_idx_generator(self,
                            nl: R_UQ_NeighborList,
                            recalc_distances: bool = False,
                            ):
        """
        Update data voxels with distances observed in a given structure.

        Parameters
        ----------
        nl : R_UQ_NeighborList
            Neighbor list object created from a structure.
        recalc_distances : bool
            Whether to recalculate distances. If False, then the distances
            are taken from `nl.pair_distances`.

        Yields
        ------
        degree : int
            Degree of n-tuplet.
        n_type : tuple[int]
            Tuple of atomic numbers of n-tuplet.
        voxel_idx : tuple[int]
            Tuple of voxel indices of n-tuplet.
        """
        # 2-body
        degree = 2
        if recalc_distances:
            dist_cache = defaultdict(lambda: defaultdict(float))
            for pair_tuple, neigh_offset, pair_type in \
                zip(nl.ntuplets['atoms'][degree],
                    nl.ntuplets['offsets'][degree],
                    nl.ntuplets['type'][degree]):
                v_ij = self.atoms.positions[pair_tuple[1]] + \
                       neigh_offset @ self.atoms.cell - \
                       self.atoms.positions[pair_tuple[0]]
                d = np.linalg.norm(v_ij)
                voxel_idx = self.r_to_voxel_idx(d, self.r_max_map[pair_type],
                                                self.dr_trust[pair_type])
                if np.all(voxel_idx >= 0) and \
                np.all(voxel_idx < self.data_voxels[degree][pair_type].shape):
                    yield degree, pair_type, voxel_idx
                t1 = (pair_tuple[1], tuple(neigh_offset))
                t2 = (pair_tuple[0], tuple(-1*neigh_offset))
                dist_cache[pair_tuple[0]][t1] = d
                dist_cache[pair_tuple[1]][t2] = d
        else:
            dist_cache = nl.pair_distances
            for pair_tuple, neigh_offset, pair_type in \
                zip(nl.ntuplets['atoms'][degree],
                    nl.ntuplets['offsets'][degree],
                    nl.ntuplets['type'][degree]):
                d = dist_cache[pair_tuple[0]][(pair_tuple[1], tuple(neigh_offset))]
                voxel_idx = self.r_to_voxel_idx(d, self.r_max_map[pair_type],
                                                self.dr_trust[pair_type])
                if np.all(voxel_idx >= 0) and \
                np.all(voxel_idx < self.data_voxels[degree][pair_type].shape):
                    yield degree, pair_type, voxel_idx

        # 3-body
        if self.degree >= 3:
            degree = 3
            for n_tuple, neigh_offsets, n_type in \
                zip(nl.ntuplets['atoms'][degree],
                    nl.ntuplets['offsets'][degree],
                    nl.ntuplets['type'][degree]):
                ds = np.empty(int(degree * (degree-1) / 2))  # n choose 2
                center_atom = n_tuple[0]

                ds[0] = dist_cache[center_atom][(n_tuple[1], tuple(neigh_offsets[0]))]
                ds[1] = dist_cache[center_atom][(n_tuple[2], tuple(neigh_offsets[1]))]
                v_jk = self.atoms.positions[n_tuple[2]] + \
                        neigh_offsets[1] @ self.atoms.cell - \
                        self.atoms.positions[n_tuple[1]] - \
                        neigh_offsets[0] @ self.atoms.cell
                ds[2] = np.linalg.norm(v_jk)

                voxel_idx = self.r_to_voxel_idx(ds, self.r_max_map[n_type],
                                                self.dr_trust[n_type])
                if np.all(voxel_idx >= 0) and \
                   np.all(voxel_idx < self.data_voxels[degree][n_type].shape):
                    yield degree, n_type, tuple(voxel_idx)
                

    def update_voxels(self,
                      nl: R_UQ_NeighborList,
                      ):
        """
        Update data voxels with distances observed in the training data.

        Parameters
        ----------
        nl : R_UQ_NeighborList
            Neighbor list object from an image from the training data.
        """
        generator = self.voxel_idx_generator(nl, recalc_distances=False)
        for degree, n_type, voxel_idx in generator:
            self.data_voxels[degree][n_type][voxel_idx] = True

        # Take into account self.bspline_config.symmetry in voxel
        if self.degree >= 3:
            degree = 3
            for n_type, data_voxel in self.data_voxels[degree].items():
                if self.symmetry_3b[n_type] == 2:
                    self.data_voxels[degree][n_type] = \
                        np.logical_or(data_voxel,
                                        data_voxel.transpose(1, 0, 2))
                elif self.symmetry_3b[n_type] == 3:
                    self.data_voxels[degree][n_type] = \
                        np.logical_or.reduce((data_voxel,
                                                data_voxel.transpose(1, 0, 2),
                                                data_voxel.transpose(2, 1, 0),
                                                data_voxel.transpose(0, 2, 1),
                                                data_voxel.transpose(2, 0, 1),
                                                data_voxel.transpose(1, 2, 0),
                                                ))
                elif self.symmetry_3b[n_type] != 1:
                    raise Exception("Symmetry of 3-body interaction not "
                                    "recognized.")

    def check_r(self,
                use_cached_distances: bool = False,
                ):
        """
        Checks if the current geometry is too epistemically uncertain based on
        the r-based UQ method.

        Parameters
        ----------
        use_cached_distances : bool
            Whether to use the cached distances from the update of `self.nl`.
        """
        generator = self.voxel_idx_generator(self.nl,
                                             recalc_distances=not use_cached_distances)
        for degree, n_type, voxel_idx in generator:
            if not self.data_voxels[degree][n_type][voxel_idx]:
                return True  # uncertain
        return False  # not uncertain
    
    def too_uncertain(self):
        """
        Updates neighbor lists, training data, and checks uncertainty.
        """
        # Check neighbor list
        updated = self.nl.update(self.atoms, max_n=self.degree)  # build new neighbor list if needed

        # If there is new training data
        if len(self.trained_traj) != self.trained_traj_len:
            self.update_trained_rs()
        
        # Check epistemic uncertainty
        return self.check_r(updated)


if __name__ == '__main__':
    from ase.atoms import Atoms
    from uf3.data import composition
    from uf3.representation import bspline

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
    print(f"nl.offset_vec = \n{nl.offset_vec}")
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
                         algo=1
                         )
    for t, s in zip(nl.ntuplets['atoms'][2], nl.ntuplets['type'][2]):
        print(f"{t}, {s}")
    for t, s in zip(nl.ntuplets['atoms'][3], nl.ntuplets['type'][3]):
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
                         algo=2
                         )
    #print(len(nl.ntuplets['which_d'][2]))
    #print(len(nl.ntuplets['which_d'][3]))
    #print(len(nl.ntuplets['which_d'][4]))
    #for t in nl.ntuplets['which_d'][2]:
    #    print(f"{t}")
    #for t in nl.ntuplets['which_d'][3]:
    #    print(f"{t}")
    '''