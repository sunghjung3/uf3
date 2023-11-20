from r_uq import R_UQ_NeighborList, R_UQ
from ase.atoms import Atoms
from uf3.data import composition
from uf3.representation import bspline
import numpy as np
import pytest


def _nl_tuplets_combo_tester(nl: R_UQ_NeighborList,
                             atoms_2: np.ndarray = None,
                             offsets_2: np.ndarray = None,
                             type_2: list = None,
                             atoms_3: np.ndarray = None,
                             offsets_3: np.ndarray = None,
                             type_3: list = None,
                             atoms_4: np.ndarray = None,
                             offsets_4: np.ndarray = None,
                             type_4: list = None,
                             ):
    """
    Helper function used in `test_nl_ntuplets*` functions to check if the
    `nl.ntuplets` dictionary matches the expected values. Works up to 4-body.

    Parameters
    ----------
    nl : R_UQ_NeighborList
        NeighborList object to test with the `nl.tabulate_ntuplets` method
        already called.
    atoms_2 : np.ndarray
        Expected `nl.ntuplets['atoms'][2]` array.
    offsets_2 : np.ndarray
        Expected `nl.ntuplets['offsets'][2]` array.
    type_2 : list
        Expected `nl.ntuplets['type'][2]` list.
    atoms_3 : np.ndarray
        Expected `nl.ntuplets['atoms'][3]` array.
    offsets_3 : np.ndarray
        Expected `nl.ntuplets['offsets'][3]` array.
    type_3 : list
        Expected `nl.ntuplets['type'][3]` list.
    atoms_4 : np.ndarray
        Expected `nl.ntuplets['atoms'][4]` array.
    offsets_4 : np.ndarray
        Expected `nl.ntuplets['offsets'][4]` array.
    type_4 : list
        Expected `nl.ntuplets['type'][4]` list.
    """
    # 2-body
    if atoms_2 is not None and offsets_2 is not None and type_2 is not None:
        assert nl.ntuplets['atoms'][2].shape == atoms_2.shape
        assert nl.ntuplets['offsets'][2].shape == offsets_2.shape
        assert len(nl.ntuplets['type'][2]) == len(type_2)

        atoms_2 = [tuple(row) for row in atoms_2]
        offsets_2 = [tuple(offset) for offset in offsets_2]
        true_zip = list(zip(atoms_2, offsets_2, type_2))
        ntuplets_atoms_2 = [tuple(row) for row in nl.ntuplets['atoms'][2]]
        ntuplets_offsets_2 = [tuple(offset) for offset in nl.ntuplets['offsets'][2]]
        test_zip = list(zip(ntuplets_atoms_2, ntuplets_offsets_2, nl.ntuplets['type'][2]))
        for true in true_zip:
            try:
                assert true in test_zip
            except AssertionError:
                # Maybe elements are the same but the atoms are in a different order
                true_type = true[2]
                if true_type[0] == true_type[1]:
                    new_atoms = true[0][::-1]
                    new_offsets = tuple(-1*i for i in true[1])  # negate offset vector
                    true = (new_atoms, new_offsets, true_type)
                    assert true in test_zip
                else:
                    raise AssertionError(f"{true} not in {test_zip}")

    # 3-body
    if atoms_3 is not None and offsets_3 is not None and type_3 is not None:
        assert nl.ntuplets['atoms'][3].shape == atoms_3.shape
        assert nl.ntuplets['offsets'][3].shape == offsets_3.shape
        assert len(nl.ntuplets['type'][3]) == len(type_3)

        atoms_3 = [tuple(row) for row in atoms_3]
        offsets_3 = [tuple([tuple(offset) for offset in row]) for row in offsets_3]
        true_zip = list(zip(atoms_3, offsets_3, type_3))
        ntuplets_atoms_3 = [tuple(row) for row in nl.ntuplets['atoms'][3]]
        ntuplets_offsets_3 = [tuple([tuple(offset) for offset in row]) for row in nl.ntuplets['offsets'][3]]
        test_zip = list(zip(ntuplets_atoms_3, ntuplets_offsets_3, nl.ntuplets['type'][3]))
        for true in true_zip:
            try:
                assert true in test_zip
            except AssertionError:
                # Maybe neighbor elements are the same but the neighbors are in a different order
                true_type = true[2]
                if true_type[1] == true_type[2]:
                    new_atoms = (true[0][0], true[0][2], true[0][1])
                    new_offsets = true[1][::-1]
                    true = (new_atoms, new_offsets, true_type)
                else:
                    raise AssertionError(f"{true} not in {test_zip}")

    # 4-body
    if atoms_4 is not None and offsets_4 is not None and type_4 is not None:
        assert nl.ntuplets['atoms'][4].shape == atoms_4.shape
        assert nl.ntuplets['offsets'][4].shape == offsets_4.shape
        assert len(nl.ntuplets['type'][4]) == len(type_4)

        atoms_4 = [tuple(row) for row in atoms_4]
        offsets_4 = [tuple([tuple(offset) for offset in row]) for row in offsets_4]
        true_zip = list(zip(atoms_4, offsets_4, type_4))
        ntuplets_atoms_4 = [tuple(row) for row in nl.ntuplets['atoms'][4]]
        ntuplets_offsets_4 = [tuple([tuple(offset) for offset in row]) for row in nl.ntuplets['offsets'][4]]
        test_zip = list(zip(ntuplets_atoms_4, ntuplets_offsets_4, nl.ntuplets['type'][4]))
        for true in true_zip:
            try:
                assert true in test_zip
            except AssertionError:
                # Maybe some neighbor elements are the same but those neighbors are in a different order
                possible_combos = list()
                true_type = true[2]
                if true_type[1] == true_type[2]:  # 1st and 2nd neighbors can be swapped
                    new_atoms = (true[0][0], true[0][2], true[0][1], true[0][3])
                    new_offsets = (true[1][1], true[1][0], true[1][2])
                    possible_combos.append((new_atoms, new_offsets, true_type))
                if true_type[1] == true_type[3]:  # 1st and 3rd neighbors can be swapped
                    new_atoms = (true[0][0], true[0][3], true[0][2], true[0][1])
                    new_offsets = (true[1][2], true[1][1], true[1][0])
                    possible_combos.append((new_atoms, new_offsets, true_type))
                if true_type[2] == true_type[3]:  # 2nd and 3rd neighbors can be swapped
                    new_atoms = (true[0][0], true[0][1], true[0][3], true[0][2])
                    new_offsets = (true[1][0], true[1][2], true[1][1])
                    possible_combos.append((new_atoms, new_offsets, true_type))
                if true_type[1] == true_type[2] and true_type[1] == true_type[3]:
                    # 2 more possibilities remaining
                    # 1. abc --> bca
                    new_atoms = (true[0][0], true[0][1], true[0][3], true[0][2])
                    new_offsets = (true[1][1], true[1][2], true[1][0])
                    possible_combos.append((new_atoms, new_offsets, true_type))
                    # 2. abc --> cab
                    new_atoms = (true[0][0], true[0][3], true[0][1], true[0][2])
                    new_offsets = (true[1][2], true[1][0], true[1][1])
                    possible_combos.append((new_atoms, new_offsets, true_type))
                if len(possible_combos) == 0:
                    raise AssertionError(f"{true} not in {test_zip}")
                else:
                    for combo in possible_combos:
                        if combo in test_zip:
                            break
                    else:
                        raise AssertionError(f"{true} not in {test_zip}")
    

def test_nl_ntuplets():
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
    nl.tabulate_ntuplets(max_n=4,
                         atomic_numbers=atoms.get_atomic_numbers(),
                         )

    atoms_2 = np.array([(0, 1), (2, 0), (3, 0), (0, 4), (0, 5), (2, 1), (3, 1), (4, 1),
                        (8, 6)])
    offsets_2 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [-1, 0, 0],
                          [0, 0, 0], [0, 0, 0], [0, 0 ,0], [0, 0, 0]])
    type_2 = [(6, 8), (1, 6), (1, 6), (6, 6), (6, 8), (1, 8), (1, 8), (6, 8),
              (1, 8)]
    atoms_3 = np.array([[0, 2, 1], [0, 3, 1], [0, 4, 1], [0, 5, 1], [0, 2, 3], [0, 2, 4],
                        [0, 2, 5], [0, 3, 4], [0, 3, 5], [0, 4, 5], [1, 2, 0], [1, 3, 0],
                        [1, 0, 4], [1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 0, 1], [3, 0, 1],
                        [4, 0, 1]])
    offsets_3 = np.array([[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]],
                          [[0, 0, 0], [-1, 0, 0]], [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]],
                          [[0, 0, 0], [-1, 0, 0]], [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [-1, 0, 0]],
                          [[0, 0, 0], [-1, 0, 0]], [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]],
                          [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]],
                          [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]],
                          [[0, 0, 0], [0, 0, 0]]])
    #which_d_3 = [[(0, 2), (0, 1), (2, 1)], [(0, 3), (0, 1), (3, 1)],[ (0, 4), (0, 1), (4, 1)],
    #             [(0, 1), (0, 5), (1, 5)], [(0, 2), (0, 3), (2, 3)], [(0, 2), (0, 4), (2, 4)],
    #             [(0, 2), (0, 5), (2, 5)], [(0, 3), (0, 4), (3, 4)], [(0, 3), (0, 5), (3, 5)],
    #             [(0, 4), (0, 5), (4, 5)], [(1, 2), (1, 0), (2, 0)], [(1, 3), (1, 0), (3, 0)],
    #             [(1, 0), (1, 4), (0, 4)], [(1, 2), (1, 3), (2, 3)], [(1, 2), (1, 4), (2, 4)],
    #             [(1, 3), (1, 4), (3, 4)], [(2, 0), (2, 1), (0, 1)], [(3, 0), (3, 1), (0, 1)],
    #             [(4, 0), (4, 1), (0, 1)]]
    type_3 = [(6, 1, 8), (6, 1, 8), (6, 6, 8),
              (6, 8, 8), (6, 1, 1), (6, 1, 6),
              (6, 1, 8), (6, 1, 6), (6, 1, 8),
              (6, 6, 8), (8, 1, 6), (8, 1, 6),
              (8, 6, 6), (8, 1, 1), (8, 1, 6),
              (8, 1, 6), (1, 6, 8), (1, 6, 8),
              (6, 6, 8)]
    atoms_4 = np.array([[0, 3, 2, 1], [0, 2, 4, 1], [0, 2, 1, 5], [0, 3, 4, 1], [0, 3, 1, 5],
                        [0, 4, 1, 5], [0, 2, 3, 4], [0, 2, 3, 5], [0, 2, 4, 5], [0, 3, 4, 5],
                        [1, 2, 3, 0], [1, 2, 0, 4], [1, 3, 0, 4], [1, 2, 3, 4]])
    offsets_4 = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                          [[0, 0, 0], [0, 0, 0], [-1, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                          [[0, 0, 0], [0, 0, 0], [-1, 0, 0]], [[0, 0, 0], [0, 0, 0], [-1, 0, 0]],
                          [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [-1, 0, 0]],
                          [[0, 0, 0], [0, 0, 0], [-1, 0, 0]], [[0, 0, 0], [0, 0, 0], [-1, 0, 0]],
                          [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                          [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    #which_d_4 = [[(0, 2), (0, 3), (0, 1), (2, 3), (2, 1), (3, 1)], [(0, 2), (0, 4), (0, 1), (2, 4), (2, 1), (4, 1)],
    #             [(0, 2), (0, 1), (0, 5), (2, 1), (2, 5), (1, 5)], [(0, 3), (0, 4), (0, 1), (3, 4), (3, 1), (4, 1)],
    #             [(0, 3), (0, 1), (0, 5), (3, 1), (3, 5), (1, 5)], [(0, 4), (0, 1), (0, 5), (4, 1), (4, 5), (1, 5)],
    #             [(0, 2), (0, 3), (0, 4), (2, 3), (2, 4), (3, 4)], [(0, 2), (0, 3), (0, 5), (2, 3), (2, 5), (3, 5)],
    #             [(0, 2), (0, 4), (0, 5), (2, 4), (2, 5), (4, 5)], [(0, 3), (0, 4), (0, 5), (3, 4), (3, 5), (4, 5)],
    #             [(1, 2), (1, 3), (1, 0), (2, 3), (2, 0), (3, 0)], [(1, 2), (1, 0), (1, 4), (2, 0), (2, 4), (0, 4)],
    #             [(1, 3), (1, 0), (1, 4), (3, 0), (3, 4), (0, 4)], [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]]
    type_4 = [(6, 1, 1, 8), (6, 1, 6, 8),
              (6, 1, 8, 8), (6, 1, 6, 8),
              (6, 1, 8, 8), (6, 6, 8, 8),
              (6, 1, 1, 6), (6, 1, 1, 8),
              (6, 1, 6, 8), (6, 1, 6, 8),
              (8, 1, 1, 6), (8, 1, 6, 6),
              (8, 1, 6, 6), (8, 1, 1, 6)]
    
    _nl_tuplets_combo_tester(nl=nl,
                             atoms_2=atoms_2,
                             offsets_2=offsets_2,
                             type_2=type_2,
                             atoms_3=atoms_3,
                             offsets_3=offsets_3,
                             type_3=type_3,
                             atoms_4=atoms_4,
                             offsets_4=offsets_4,
                             type_4=type_4,
                             )           
    
def test_nl_ntuplets_double_neighbors():
    atoms = Atoms("OH2")
    atoms.positions = np.array([[0, 0, 0],
                                [1, 1, 0],
                                [2, 0, 0]])
    atoms.cell = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]])
    atoms.pbc = True
    cutoffs = {('O', 'H'): 2.1,
               ('O', 'O'): 2.1,
               ('H', 'H'): 2.1}
    nl = R_UQ_NeighborList(cutoffs=cutoffs,
                           skin=0.0)
    nl.build(atoms)
    nl.tabulate_ntuplets(max_n=3,
                         atomic_numbers=atoms.get_atomic_numbers(),
                         )

    atoms_2 = np.array([(1, 0), (2, 0), (2, 0), (1, 2)])
    offsets_2 = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0]])
    type_2 = [(1, 8), (1, 8), (1, 8), (1, 1)]
    atoms_3 = np.array([(0, 1, 2), (0, 1, 2), (0, 2, 2), (1, 2, 0), (2, 1, 0), (2, 0, 0),
                        (2, 1, 0)])
    offsets_3 = np.array([[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [-1, 0, 0]], [[-1, 0, 0], [0, 0, 0]],
                          [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [1, 0, 0]],
                          [[0, 0, 0], [1, 0, 0]]])
    type_3 = [(8, 1, 1), (8, 1, 1), (8, 1, 1), (1, 1, 8), (1, 1, 8), (1, 8, 8), (1, 1, 8)]

    _nl_tuplets_combo_tester(nl=nl,
                                atoms_2=atoms_2,
                                offsets_2=offsets_2,
                                type_2=type_2,
                                atoms_3=atoms_3,
                                offsets_3=offsets_3,
                                type_3=type_3,
                                atoms_4=None,
                                offsets_4=None,
                                type_4=None,
                                )


def test_voxel_update():
    atoms = Atoms("CHHCHHCCH")
    atoms.positions = np.array([[0, 0, 0],
                                [2, 0, 0],
                                [13, 2, 0],
                                [0, 3, 0],
                                [13, 14.5, 0],
                                [2, 2.999, 0],
                                [12, 0, 0],
                                [0, 0, 7],
                                [0, 0, 7.5]
                                ])
    atoms.cell = np.array([[15, 0, 0], [0, 15, 0], [0, 0, 15]])
    atoms.pbc = True

    # Test 2-body
    chemical_system = composition.ChemicalSystem(element_list=['C', 'H'],
                                                 degree=2)
    r_max_map = {('C', 'C'): 5.0, ('C', 'H'): 4.0, ('H', 'H'): 3.0}
    r_min_map = {('C', 'C'): 1.5, ('C', 'H'): 1.5, ('H', 'H'): 1.5}
    resolution_map = {('C', 'C'): 7, ('C', 'H'): 5, ('H', 'H'): 3}
    bspline_config = bspline.BSplineBasis(chemical_system=chemical_system,
                                          r_max_map=r_max_map,
                                          r_min_map=r_min_map,
                                          resolution_map=resolution_map
                                          )
    r_uq_obj = R_UQ(atoms=atoms,
                    trained_traj=[atoms],
                    bspline_config=bspline_config,
                    uq_tolerance=1.0,
                    )
    r_uq_obj.update_trained_rs()
    # type: (atom numbers): (distances) (voxel index)
    # C-C: (0, 3): 3.0 4, (0, 6): 3.0 4, (3, 6): 4.243 1
    assert np.all(np.equal(r_uq_obj.data_voxels[2][(6, 6)], 
                  np.array([0, 1, 0, 0, 1, 0, 0, 0], dtype=bool)))
    # H-C: (0, 1): 2.0 4, (0, 2): 2.828 2, (0, 4): 2.062 3, (0, 5): 3.606 0,
    #      (1, 3): 3.606 0, (2, 3): 2.236 3, (2, 6): 2.236 3, (3, 5): 2.0 4,
    #      (4, 6): 1.118 5 (this last one doesn't have to be included)
    assert np.all(np.equal(r_uq_obj.data_voxels[2][(1, 6)],
                  np.array([1, 0, 1, 1, 1, 1], dtype=bool)))
    # H-H: (1, 5): 3 0, (2, 4): 2.5 1
    assert np.all(np.equal(r_uq_obj.data_voxels[2][(1, 1)],
                  np.array([1, 1, 0, 0], dtype=bool)))

    # Test 3-body
    chemical_system = composition.ChemicalSystem(element_list=['C', 'H'],
                                                 degree=3)
    r_max_map = {('C', 'C'): 5.0, ('C', 'H'): 4.0, ('H', 'H'): 3.0, 
                 ('C', 'C', 'C'): [2.0, 2.0, 4.0], ('C', 'H', 'C'): [3.5, 3.5, 7.0],
                 ('C', 'H', 'H'): [2.5, 2.5, 5.0], ('H', 'H', 'H'): [1.0, 1.0, 2.0],
                 ('H', 'H', 'C'): [1.0, 1.0, 2.0], ('H', 'C', 'C'): [1.0, 1.0, 2.0]}
    r_min_map = {('C', 'C'): 0.0, ('C', 'H'): 0.0, ('H', 'H'): 0.0,
                 ('C', 'C', 'C'): [0.0, 0.0, 0.0], ('C', 'H', 'C'): [0.0, 0.0, 0.0],
                 ('C', 'H', 'H'): [0.0, 0.0, 0.0], ('H', 'H', 'H'): [0.0, 0.0, 0.0],
                 ('H', 'H', 'C'): [0.0, 0.0, 0.0], ('H', 'C', 'C'): [0.0, 0.0, 0.0]}
    resolution_map = {('C', 'C'): 16, ('C', 'H'): 16, ('H', 'H'): 16,
                      ('C', 'C', 'C'): [3, 3, 5], ('C', 'H', 'C'): [5, 5, 7],
                      ('C', 'H', 'H'): [4, 4, 8], ('H', 'H', 'H'): [1, 1, 2],
                      ('H', 'H', 'C'): [1, 1, 2], ('H', 'C', 'C'): [1, 1, 2]}
    bspline_config = bspline.BSplineBasis(chemical_system=chemical_system,
                                          r_max_map=r_max_map,
                                          r_min_map=r_min_map,
                                          resolution_map=resolution_map
                                          )
    r_uq_obj = R_UQ(atoms=atoms,
                    trained_traj=[atoms],
                    bspline_config=bspline_config,
                    uq_tolerance=1.0,
                    )
    r_uq_obj.update_trained_rs()
    assert np.all(~r_uq_obj.data_voxels[3][(6, 6, 6)])
    assert np.all(~r_uq_obj.data_voxels[3][(1, 6, 6)])
    assert np.all(~r_uq_obj.data_voxels[3][(1, 1, 6)])
    assert np.all(~r_uq_obj.data_voxels[3][(1, 1, 1)])
    assert r_uq_obj.data_voxels[3][(6, 1, 1)].shape == (5, 5, 9)
    assert r_uq_obj.data_voxels[3][(6, 1, 6)].shape == (6, 6, 8)
    # C-H-H: (0, 1, 4): (2.0, 2.062, 4.031) (0, 0, 1), (3, 2, 5): (2.236, 2, 4.123) (0, 0, 1),
    #        (6, 2, 4): (2.236, 1.118, 2.5) (0, 2, 4)
    assert np.all(np.equal(r_uq_obj.data_voxels[3][(6, 1, 1)],
                           np.array([[[0, 1, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                     [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                     [[0, 0, 0, 0, 1, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                     [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                     [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                      ], dtype=bool)))
    # C-H-C: (0, 1, 3): (2, 3, 3.606) (2, 0, 3), (0, 1, 6): (2, 3, 5.0) (2, 0, 2),
    #        (0, 2, 3): (2.828, 3, 2.236) (0, 0, 4), (0, 2, 6): (2.828, 3, 2.236) (0, 0, 4),
    #        (0, 4, 3): (2.062, 3, 4.031) (2, 0, 2), (0, 4, 6): (2.062, 3, 1.118) (2, 0, 5),
    #        (3, 2, 0): (2.236, 3, 2.828) (1, 0, 4), (3, 5, 0): (2, 3, 3.606) (2, 0, 3),
    #        (6, 2, 0): (2.236, 3, 2.828) (1, 0, 4), (6, 4, 0): (1.118, 3, 2.062) (3, 0, 4)
    assert np.all(np.equal(r_uq_obj.data_voxels[3][(6, 1, 6)],
                           np.array([[[0, 0, 0, 0, 1, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0]],
                                     [[0, 0, 0, 0, 1, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0]],
                                     [[0, 0, 1, 1, 0, 1, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0]],
                                     [[0, 0, 0, 0, 1, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0]],
                                     [[0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0]],
                                     [[0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0]],
                           ])))
