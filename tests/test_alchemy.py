import numpy as np
import pytest
import scipy.sparse as sp

from uf3.data import composition
from uf3.representation import bspline
from uf3.alchemy import alchemy


@pytest.fixture(scope="module")
def small_config():
    chemical_system = composition.ChemicalSystem(
        element_list=["W", "Mo", "V"], degree=3)
    r_min_map = {p: 0.0 for p in chemical_system.interactions_map[2]}
    r_min_map.update({t: [0.0] * 3 for t in chemical_system.interactions_map[3]})
    r_max_map = {p: 6.0 for p in chemical_system.interactions_map[2]}
    r_max_map.update({t: [5.0, 5.0, 10.0]
                      for t in chemical_system.interactions_map[3]})
    resolution_map = {p: 6 for p in chemical_system.interactions_map[2]}
    resolution_map.update({t: [3, 3, 6]
                           for t in chemical_system.interactions_map[3]})
    config = bspline.BSplineBasis(chemical_system, r_min_map=r_min_map,
                                  r_max_map=r_max_map,
                                  resolution_map=resolution_map,
                                  leading_trim=0, trailing_trim=3)
    for trio in config.interactions_map.get(3, []):
        config.symmetry[trio] = 1
    config.update_basis_functions()
    return config


N_PSEUDO = {2: 2, 3: 3}


def make_model(small_config):
    return alchemy.AlchemicalModel(small_config, N_PSEUDO)


@pytest.fixture(scope="module")
def synth_data(small_config, tmp_path_factory):
    """Synthetic sparse preprocessed tables + metadata/coverage files."""
    tmp = tmp_path_factory.mktemp("synth")
    model = make_model(small_config)
    rng = np.random.default_rng(0)
    files = {"preprocessed": str(tmp / "preprocessed.h5"),
             "metadata": str(tmp / "metadata.npz"),
             "coverage": str(tmp / "coverage.npz")}
    n_e_total, n_f_total = 0, 0
    for table in ("t0", "t1", "t2"):
        n_e, n_f = 8, 120
        x_es = {1: rng.integers(1, 5, (n_e, model.n_elements)).astype(float)}
        x_fs = {1: np.zeros((n_f, model.n_elements))}
        for k in (2, 3):
            width = model.n_ituples[k] * model.n_basis[k]
            x_es[k] = (rng.random((n_e, width)) *
                       (rng.random((n_e, width)) < 0.1))
            x_fs[k] = (rng.standard_normal((n_f, width)) *
                       (rng.random((n_f, width)) < 0.1))
        y_e = rng.standard_normal(n_e)
        y_f = rng.standard_normal(n_f)
        alchemy.save_preprocessed_db(x_es, y_e, x_fs, y_f,
                                     files["preprocessed"], degree=3,
                                     batch_name=table)
        n_e_total += n_e
        n_f_total += n_f
    np.savez(files["metadata"], w_e=0.05, w_f=0.002,
             n_e=n_e_total, n_f=n_f_total)
    model.data_coverage = np.ones(model.n_feats, dtype=bool)
    model.frozen_data_coverage = {1: np.ones(model.n_elements, dtype=bool)}
    model.frozen_data_coverage.update(
        {k: np.ones(model.n_basis[k], dtype=bool) for k in (2, 3)})
    model.save_data_coverage(files["coverage"])
    model.initialize_parameters(None)
    model.save_alchemical_params(str(tmp / "init.npz"))
    files["init"] = str(tmp / "init.npz")
    return files


def load_tables(files):
    import tables
    with tables.open_file(files["preprocessed"]) as f:
        names = [g._v_name for g in f.list_nodes("/")]
    return [alchemy.load_preprocessed_db(files["preprocessed"], n,
                                         load_sparse=True) for n in names]


def test_krp_sum_equals_matmul():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((7, 3))
    B = rng.standard_normal((11, 7, 5))
    ref = alchemy.broad_row_krp_sum(A, B)
    alt = np.matmul(A.T, B).reshape(11, 15)
    np.testing.assert_allclose(ref, alt, rtol=1e-13)


def test_feature_matrixC_sparse_equals_dense(small_config, synth_data):
    model = make_model(small_config)
    model.load_alchemical_params(synth_data["init"])
    for x_es, y_e, x_fs, y_f in load_tables(synth_data):
        for xs in (x_es, x_fs):
            dense = {a: v.toarray() for a, v in xs.items()}
            sparse = {1: dense[1], 2: xs[2], 3: xs[3]}
            ref = model.feature_matrixC(dense, model.pseudo_weights)
            alt = model.feature_matrixC(sparse, model.pseudo_weights)
            np.testing.assert_allclose(alt, ref, atol=1e-12)


def test_feature_matrixW_sparse_equals_dense(small_config, synth_data):
    model = make_model(small_config)
    model.load_alchemical_params(synth_data["init"])
    for x_es, y_e, x_fs, y_f in load_tables(synth_data):
        dense = {a: v.toarray() for a, v in x_es.items()}
        sparse = {1: dense[1], 2: x_es[2], 3: x_es[3]}
        ref, ref_1b = model.feature_matrixW(dense, model.coeff, return_1b=True)
        alt, alt_1b = model.feature_matrixW(sparse, model.coeff, return_1b=True)
        np.testing.assert_allclose(alt, ref, atol=1e-12)
        np.testing.assert_allclose(alt_1b, ref_1b, atol=1e-12)


def run_fit(small_config, synth_data, workdir, method, **kwargs):
    import os
    cwd = os.getcwd()
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)
    try:
        model = make_model(small_config)
        C_regularizers = alchemy.get_C_regularizers(
            small_config, ridge_1b=1e-8, ridge_2b=1e-8, ridge_3b=1e-6,
            curvature_2b=1e-6, curvature_3b=1e-8)
        init = np.load(synth_data["init"])
        common = dict(init_params=init, C_regularizers=C_regularizers,
                      max_iter=3, checkpoint=3)
        if method == "gram":
            alchemy.accumulate_gram(synth_data["preprocessed"], "gram",
                                    progress="none")
            model.fit_from_gram("gram", metadata_file=synth_data["metadata"],
                                coverage_file=synth_data["coverage"], **common,
                                **kwargs)
        else:
            model.fit_from_file(preprocessed_file=synth_data["preprocessed"],
                                coverage_file=synth_data["coverage"],
                                metadata_file=synth_data["metadata"],
                                progress="none", **common, **kwargs)
        tracker = np.load("checkpoint/train_tracker.npz")
        return {k: tracker[k] for k in
                ("rmse_e", "rmse_f", "data_loss", "total_loss")}, model
    finally:
        os.chdir(cwd)


def test_cache_reuse_not_corrupted(small_config, synth_data, tmp_path):
    a, _ = run_fit(small_config, synth_data, tmp_path / "nc", "file",
                   sparse_tables=True, cache_tables=False)
    b, _ = run_fit(small_config, synth_data, tmp_path / "c", "file",
                   sparse_tables=True, cache_tables=True)
    for key in a:
        np.testing.assert_allclose(b[key], a[key], rtol=1e-12, err_msg=key)


