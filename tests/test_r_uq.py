from new_r_uq import R_UQ_NeighborList
from ase.atoms import Atoms
import numpy as np

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
                         algo=2
                         )

    which_d_2 = [(0, 1), (2, 0), (3, 0), (0, 4), (0, 5), (2, 1), (3, 1), (4, 1),
                 (8, 6)]
    type_2 = [(6, 8), (1, 6), (1, 6), (6, 6), (6, 8), (1, 8), (1, 8), (6, 8),
              (1, 8)]
    which_d_3 = [[(0, 2), (0, 1), (2, 1)], [(0, 3), (0, 1), (3, 1)],[ (0, 4), (0, 1), (4, 1)],
                 [(0, 1), (0, 5), (1, 5)], [(0, 2), (0, 3), (2, 3)], [(0, 2), (0, 4), (2, 4)],
                 [(0, 2), (0, 5), (2, 5)], [(0, 3), (0, 4), (3, 4)], [(0, 3), (0, 5), (3, 5)],
                 [(0, 4), (0, 5), (4, 5)], [(1, 2), (1, 0), (2, 0)], [(1, 3), (1, 0), (3, 0)],
                 [(1, 0), (1, 4), (0, 4)], [(1, 2), (1, 3), (2, 3)], [(1, 2), (1, 4), (2, 4)],
                 [(1, 3), (1, 4), (3, 4)], [(2, 0), (2, 1), (0, 1)], [(3, 0), (3, 1), (0, 1)],
                 [(4, 0), (4, 1), (0, 1)]]
    type_3 = [(6, 1, 8), (6, 1, 8), (6, 6, 8),
              (6, 8, 8), (6, 1, 1), (6, 1, 6),
              (6, 1, 8), (6, 1, 6), (6, 1, 8),
              (6, 6, 8), (8, 1, 6), (8, 1, 6),
              (8, 6, 6), (8, 1, 1), (8, 1, 6),
              (8, 1, 6), (1, 6, 8), (1, 6, 8),
              (6, 6, 8)]
    which_d_4 = [[(0, 2), (0, 3), (0, 1), (2, 3), (2, 1), (3, 1)], [(0, 2), (0, 4), (0, 1), (2, 4), (2, 1), (4, 1)],
                 [(0, 2), (0, 1), (0, 5), (2, 1), (2, 5), (1, 5)], [(0, 3), (0, 4), (0, 1), (3, 4), (3, 1), (4, 1)],
                 [(0, 3), (0, 1), (0, 5), (3, 1), (3, 5), (1, 5)], [(0, 4), (0, 1), (0, 5), (4, 1), (4, 5), (1, 5)],
                 [(0, 2), (0, 3), (0, 4), (2, 3), (2, 4), (3, 4)], [(0, 2), (0, 3), (0, 5), (2, 3), (2, 5), (3, 5)],
                 [(0, 2), (0, 4), (0, 5), (2, 4), (2, 5), (4, 5)], [(0, 3), (0, 4), (0, 5), (3, 4), (3, 5), (4, 5)],
                 [(1, 2), (1, 3), (1, 0), (2, 3), (2, 0), (3, 0)], [(1, 2), (1, 0), (1, 4), (2, 0), (2, 4), (0, 4)],
                 [(1, 3), (1, 0), (1, 4), (3, 0), (3, 4), (0, 4)], [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]]
    type_4 = [(6, 1, 1, 8), (6, 1, 6, 8),
              (6, 1, 8, 8), (6, 1, 6, 8),
              (6, 1, 8, 8), (6, 6, 8, 8),
              (6, 1, 1, 6), (6, 1, 1, 8),
              (6, 1, 6, 8), (6, 1, 6, 8),
              (8, 1, 1, 6), (8, 1, 6, 6),
              (8, 1, 6, 6), (8, 1, 1, 6)]
    
    assert len(nl.ntuplets['which_d'][2]) == len(which_d_2)
    assert len(nl.ntuplets['type'][2]) == len(type_2)
    assert len(nl.ntuplets['which_d'][3]) == len(which_d_3)
    assert len(nl.ntuplets['type'][3]) == len(type_3)
    assert len(nl.ntuplets['which_d'][4]) == len(which_d_4)
    assert len(nl.ntuplets['type'][4]) == len(type_4)

    true_zip = list(zip(which_d_2, type_2))
    test_zip = list(zip(nl.ntuplets['which_d'][2], nl.ntuplets['type'][2]))
    for true in true_zip:
        assert true in test_zip
    true_zip = list(zip(which_d_3, type_3))
    test_zip = list(zip(nl.ntuplets['which_d'][3], nl.ntuplets['type'][3]))
    for true in true_zip:
        assert true in test_zip
    true_zip = list(zip(which_d_4, type_4))
    test_zip = list(zip(nl.ntuplets['which_d'][4], nl.ntuplets['type'][4]))
    for true in true_zip:
        assert true in test_zip
    