from ase.neighborlist import primitive_neighbor_list, first_neighbors

import numpy as np

import heapq


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
        For R_UQ, the skin should be set to the smallest knot interval
        times the UQ tolerance factor.
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

    def update(self, atoms):
        """Make sure the list is up to date."""

        if self.nupdates == 0:
            self.build(atoms)
            return True

        largest_2_displacements = np.sqrt(
                heapq.nlargest(2, ((self.positions - atoms.positions)**2).sum(1))
        )

        if ((self.pbc != atoms.pbc).any() or (self.cell != atoms.cell).any() or
                np.sum(largest_2_displacements) > self.skin):
            self.build(atoms)
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
        self.offset_vec = offset_vec
        self.distance_mag = distance_mag  # values at build time
        self.distance_vec = distance_vec  # values at build time

        # Compute the index array point to the first neighbor
        self.first_neigh = first_neighbors(len(positions), pair_first)

        self.nupdates += 1

    def get_neighbors(self, a):
        """Return neighbors of atom number a.

        A list of indices and offsets to neighboring atoms is
        returned.  The positions of the neighbor atoms can be
        calculated like this:

        >>>  indices, offsets = nl.get_neighbors(42)
        >>>  for i, offset in zip(indices, offsets):
        >>>      print(atoms.positions[i] + dot(offset, atoms.get_cell()))

        Notice that if get_neighbors(a) gives atom b as a neighbor,
        then get_neighbors(b) will not return a as a neighbor - unless
        bothways=True was used."""

        return (self.pair_second[self.first_neigh[a]:self.first_neigh[a + 1]],
                self.offset_vec[self.first_neigh[a]:self.first_neigh[a + 1]])


if __name__ == '__main__':
    from ase.atoms import Atoms

    atoms = Atoms('H2O', positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    nl = R_UQ_NeighborList(cutoffs={("O", "O"): 5.0, ("O", "H"): 0.82, ("H", "H"): 5.0},
                           skin=0.6)
    nl.update(atoms)
    print(f"nl.pair_first = {nl.pair_first}")
    print(f"nl.pair_second = {nl.pair_second}")
    print(f"nl.first_neigh = {nl.first_neigh}")
    print(f"nl.offset_vec = {nl.offset_vec}")
    print(f"nl.distance_mag = {nl.distance_mag}")
    print(f"nl.distance_vec = {nl.distance_vec}")
    print(f"nl.get_neighbors(1) = {nl.get_neighbors(1)}")