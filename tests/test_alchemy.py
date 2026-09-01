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


# --- exact vs diagonal-approximate regularization (C_reg_exact) ---

REG_KW = dict(ridge_1b=1e-3, ridge_2b=1e-2, ridge_3b=1e-2,
              curvature_2b=1e-1, curvature_3b=1e-1)


def reg_model(small_config, n_pseudo, seed=0):
    """Model with fully populated random C~ and W."""
    model = alchemy.AlchemicalModel(small_config, n_pseudo)
    rng = np.random.default_rng(seed)
    init = {"coeff_1b": rng.standard_normal(model.n_elements)}
    for k in (2, 3):
        init[f"coeff_{k}b"] = rng.standard_normal(
            (model.n_basis[k], model.n_pseudo[k]))
        init[f"pseudo_weights_{k}b"] = rng.standard_normal(
            (model.n_ituples[k], model.n_pseudo[k]))
    model.initialize_parameters(init)
    return model


def block_diag(*blocks):
    n = sum(b.shape[0] for b in blocks)
    out = np.zeros((n, n))
    i = 0
    for b in blocks:
        out[i:i + b.shape[0], i:i + b.shape[1]] = b
        i += b.shape[0]
    return out


@pytest.mark.parametrize("n_pseudo", [{2: 2, 3: 3}, {2: 1, 3: 1}])
def test_reg_gramC_exact_is_kronecker(small_config, n_pseudo):
    """C-step exact penalty is (W W^T) kron (R^T R) per order."""
    model = reg_model(small_config, n_pseudo)
    regs = alchemy.get_C_regularizers(small_config, **REG_KW)
    gram, _ = model.initialize_gramC_ordinateC()
    model.update_reg_gramC(gram, regs, C_reg_exact=True)
    expect = block_diag(regs[1].T @ regs[1],
                        *[np.kron(model.pseudo_weights[k].T @ model.pseudo_weights[k],
                                  regs[k].T @ regs[k]) for k in (2, 3)])
    np.testing.assert_allclose(gram, expect, rtol=0, atol=1e-12)


@pytest.mark.parametrize("n_pseudo", [{2: 2, 3: 3}, {2: 1, 3: 1}])
def test_reg_gramC_default_is_diagonal_approx(small_config, n_pseudo):
    """Default C-step keeps only the diagonal blocks (pre-change behaviour)."""
    model = reg_model(small_config, n_pseudo)
    regs = alchemy.get_C_regularizers(small_config, **REG_KW)
    gram, _ = model.initialize_gramC_ordinateC()
    model.update_reg_gramC(gram, regs)
    expect = block_diag(
        regs[1].T @ regs[1],
        *[np.kron(np.diag(np.diag(model.pseudo_weights[k].T
                                  @ model.pseudo_weights[k])),
                  regs[k].T @ regs[k]) for k in (2, 3)])
    np.testing.assert_allclose(gram, expect, rtol=0, atol=1e-12)


@pytest.mark.parametrize("n_pseudo", [{2: 2, 3: 3}, {2: 1, 3: 1}])
def test_reg_gramW_exact_and_default(small_config, n_pseudo):
    """W-step exact adds (R C~)^T (R C~) per ituple; default only its diagonal."""
    model = reg_model(small_config, n_pseudo)
    regs = alchemy.get_C_regularizers(small_config, **REG_KW)
    exact, _ = model.initialize_gramW_ordinateW()
    approx, _ = model.initialize_gramW_ordinateW()
    model.update_reg_gramW(exact, 0.0, 1e-12, regs, C_reg_exact=True)
    model.update_reg_gramW(approx, 0.0, 1e-12, regs)
    blocks_exact, blocks_approx = [], []
    for k in (2, 3):
        RC = regs[k] @ model.coeff[k]
        blocks_exact += [RC.T @ RC] * model.n_ituples[k]
        blocks_approx += [np.diag(np.sum(RC ** 2, axis=0))] * model.n_ituples[k]
    np.testing.assert_allclose(exact, block_diag(*blocks_exact), rtol=0, atol=1e-12)
    np.testing.assert_allclose(approx, block_diag(*blocks_approx), rtol=0, atol=1e-12)


def test_reg_exact_quadratic_equals_penalty(small_config):
    """Both exact half-step forms evaluate to ||R C~ W||_F^2 at the current
    parameters; the diagonal approximation does not."""
    model = reg_model(small_config, {2: 2, 3: 3})
    regs = alchemy.get_C_regularizers(small_config, **REG_KW)
    penalty = sum(np.sum((regs[k] @ model.coeff[k]
                          @ model.pseudo_weights[k].T) ** 2) for k in (2, 3))
    one_body = model.coeff[1] @ (regs[1].T @ regs[1]) @ model.coeff[1]

    gram, _ = model.initialize_gramC_ordinateC()
    model.update_reg_gramC(gram, regs, C_reg_exact=True)
    v = np.concatenate([model.coeff[1]]
                       + [model.coeff[k].T.flatten() for k in (2, 3)])
    np.testing.assert_allclose(v @ gram @ v - one_body, penalty, rtol=1e-12)

    gram, _ = model.initialize_gramW_ordinateW()
    model.update_reg_gramW(gram, 0.0, 1e-12, regs, C_reg_exact=True)
    w = model.flattened_W()
    np.testing.assert_allclose(w @ gram @ w, penalty, rtol=1e-12)

    gram, _ = model.initialize_gramC_ordinateC()
    model.update_reg_gramC(gram, regs)
    assert abs(v @ gram @ v - one_body - penalty) > 1e-6 * penalty


def test_reg_exact_equals_default_at_rank_one(small_config):
    """With one pseudo-interaction per order the approximation is exact."""
    model = reg_model(small_config, {2: 1, 3: 1})
    regs = alchemy.get_C_regularizers(small_config, **REG_KW)
    a, _ = model.initialize_gramC_ordinateC()
    b, _ = model.initialize_gramC_ordinateC()
    model.update_reg_gramC(a, regs, C_reg_exact=True)
    model.update_reg_gramC(b, regs)
    # ||w||^2 via norm() vs the dot product differ only in rounding
    np.testing.assert_allclose(a, b, rtol=1e-14, atol=0)
    a, _ = model.initialize_gramW_ordinateW()
    b, _ = model.initialize_gramW_ordinateW()
    model.update_reg_gramW(a, 0.0, 1e-12, regs, C_reg_exact=True)
    model.update_reg_gramW(b, 0.0, 1e-12, regs)
    np.testing.assert_allclose(a, b, rtol=1e-14, atol=0)


def test_reg_exact_ignored_when_free(small_config):
    """C_reg_free takes precedence; C_reg_exact must not change that path."""
    model = reg_model(small_config, {2: 2, 3: 3})
    regs = alchemy.get_C_regularizers(small_config, **REG_KW)
    a, _ = model.initialize_gramC_ordinateC()
    b, _ = model.initialize_gramC_ordinateC()
    model.update_reg_gramC(a, regs, C_reg_free=True, C_reg_exact=True)
    model.update_reg_gramC(b, regs, C_reg_free=True)
    np.testing.assert_array_equal(a, b)

