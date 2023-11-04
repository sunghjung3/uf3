from new_r_uq import R_UQ_NeighborList, R_UQ
from ase.atoms import Atoms
from uf3.data import composition
from uf3.representation import bspline
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
 