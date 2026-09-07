"""Torch-side tests for the alchemical model. Kept apart from test_alchemy.py so
that hosts without torch still run every numpy test instead of skipping the
whole module (a module-level importorskip skips everything after collection)."""
import os
import numpy as np
import pytest

from uf3.alchemy import alchemy
from tests.test_alchemy import (  # noqa: F401  (fixtures are discovered by name)
    N_PSEUDO,
    make_model,
    reg_model,
    small_config,
    stacked_design,
    synth_data)

torch = pytest.importorskip("torch")


def torch_reg_model(small_config, n_pseudo, seed=0):
    """AlchemicalModelTorch holding the same parameters as reg_model."""
    ref = reg_model(small_config, n_pseudo, seed)
    model = alchemy.AlchemicalModelTorch(small_config, n_pseudo)
    model.initialize_parameters(
        {"coeff_1b": ref.coeff[1],
         **{f"coeff_{k}b": ref.coeff[k] for k in (2, 3)},
         **{f"pseudo_weights_{k}b": ref.pseudo_weights[k] for k in (2, 3)}})
    model.convert2torch()
    return ref, model


def numpy_reg_penalty(model, regs, C_reg_free=False):
    """The diagonal-approximate penalty, written out directly from its definition."""
    total = np.sum((regs[1] @ model.coeff[1]) ** 2) if 1 in regs else 0.0
    for k in (2, 3):
        if k not in regs:
            continue
        part = regs[k] @ model.coeff[k]
        if not C_reg_free:
            part = part * np.linalg.norm(model.pseudo_weights[k], axis=0)
        total += np.sum(part ** 2)
    return total


@pytest.mark.parametrize("n_pseudo", [{2: 2, 3: 3}, {2: 1, 3: 1}])
@pytest.mark.parametrize("C_reg_free", [False, True])
def test_torch_regularization_loss_matches_definition(small_config, n_pseudo,
                                                      C_reg_free):
    ref, model = torch_reg_model(small_config, n_pseudo)
    regs = alchemy.get_C_regularizers(small_config, ridge_1b=1e-3, ridge_2b=2e-3,
                                      ridge_3b=3e-3, curvature_2b=4e-3,
                                      curvature_3b=5e-3)
    got = model.regularization_loss(regs, 0.0, C_reg_free).detach().numpy()
    np.testing.assert_allclose(got, numpy_reg_penalty(ref, regs, C_reg_free),
                               rtol=1e-12)


def test_torch_regularization_loss_no_regularizers(small_config):
    _, model = torch_reg_model(small_config, {2: 2, 3: 2})
    assert model.regularization_loss(None, 0.0) == 0.0


def test_torch_regularization_loss_adds_sparsity_term(small_config):
    ref, model = torch_reg_model(small_config, {2: 2, 3: 2})
    regs = alchemy.get_C_regularizers(small_config, ridge_2b=1e-3, ridge_3b=1e-3)
    base = model.regularization_loss(regs, 0.0).detach().numpy()
    with_l1 = model.regularization_loss(regs, 0.25).detach().numpy()
    expected = 0.25 * sum(np.abs(ref.pseudo_weights[k]).sum() for k in (2, 3))
    np.testing.assert_allclose(with_l1 - base, expected, rtol=1e-12)


def test_torch_regularization_loss_is_differentiable(small_config):
    """The dead block referenced an attribute that never existed; the live term
    must still carry gradients back to both factors."""
    _, model = torch_reg_model(small_config, {2: 2, 3: 2})
    regs = alchemy.get_C_regularizers(small_config, ridge_2b=1e-3, ridge_3b=1e-3)
    model.regularization_loss(regs, 0.1).backward()
    for k in (2, 3):
        assert model.coeff[k].grad is not None
        assert np.isfinite(model.coeff[k].grad.numpy()).all()
        assert model.pseudo_weights[k].grad is not None


@pytest.mark.parametrize("n_pseudo", [{2: 2, 3: 3}, {2: 1, 3: 1}])
def test_torch_flattened_W_matches_numpy(small_config, n_pseudo):
    """The torch override must reproduce the parent's ordering and values exactly."""
    ref, model = torch_reg_model(small_config, n_pseudo)
    np.testing.assert_array_equal(model.flattened_W().detach().numpy(),
                                  ref.flattened_W())


@pytest.mark.parametrize("n_pseudo", [{2: 2, 3: 3}, {2: 1, 3: 1}])
def test_torch_save_params_matches_numpy(small_config, n_pseudo, tmp_path):
    """Checkpoints written from the torch path must be byte-identical to the
    numpy path's for the same parameters."""
    ref, model = torch_reg_model(small_config, n_pseudo)
    ref.save_alchemical_params(tmp_path / "ref.npz")
    model.save_alchemical_params(tmp_path / "torch.npz")
    a, b = np.load(tmp_path / "ref.npz"), np.load(tmp_path / "torch.npz")
    assert sorted(a.files) == sorted(b.files)
    for k in a.files:
        np.testing.assert_array_equal(a[k], b[k])


def test_torch_save_params_leaves_graph_intact(small_config, tmp_path):
    """Saving must not detach the live parameters the optimizer holds."""
    _, model = torch_reg_model(small_config, {2: 2, 3: 2})
    model.save_alchemical_params(tmp_path / "c.npz")
    for k in (2, 3):
        assert model.coeff[k].requires_grad
        assert model.pseudo_weights[k].requires_grad


# --- torch training loop: parameter identity, loss, and gradients ---

class _CaptureOptimizer:
    """Optimizer stand-in recording what train_from_file() hands it and the
    gradients it produces. Leaves the parameters untouched."""

    def __init__(self, params, store):
        self.params = list(params)
        store["params"] = self.params
        self.store = store

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.store.setdefault("grads", []).append(
            [None if p.grad is None else p.grad.detach().cpu().clone()
             for p in self.params])


TORCH_REGS = dict(ridge_1b=1e-8, ridge_2b=1e-8, ridge_3b=1e-6,
                  curvature_2b=1e-6, curvature_3b=1e-8)


def run_torch_train(small_config, synth_data, workdir, **kwargs):
    """train_from_file() in its own directory, returning the model and tracker."""
    import os
    os.makedirs(workdir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(workdir)
    model = alchemy.AlchemicalModelTorch(small_config, N_PSEUDO)
    regs = alchemy.get_C_regularizers(small_config, **TORCH_REGS)
    try:
        model.train_from_file(preprocessed_file=synth_data["preprocessed"],
                              coverage_file=synth_data["coverage"],
                              metadata_file=synth_data["metadata"],
                              init_params=np.load(synth_data["init"]),
                              C_regularizers=regs, progress="none",
                              shuffle=False, **kwargs)
        tracker = dict(np.load("train_tracker.npz"))
    finally:
        os.chdir(cwd)
    return model, regs, tracker


def torch_loss_parts(model, synth_data, regs):
    """Pack/unpack and the loss train_from_file() minimizes, in numpy."""
    from types import SimpleNamespace
    X, y = stacked_design(model, synth_data, 1.0, 1.0)
    meta = np.load(synth_data["metadata"])
    w = {t: float(meta[f"w_{t}"]) for t in "ef"}

    def loss_of(parts):
        c1, C2, C3, W2, W3 = parts
        full = np.concatenate([c1, (C2 @ W2.T).flatten(order="F"),
                               (C3 @ W3.T).flatten(order="F")])
        data = sum(w[t]**2 * np.sum((X[t] @ full - y[t])**2) for t in "ef")
        shim = SimpleNamespace(coeff={1: c1, 2: C2, 3: C3},
                               pseudo_weights={2: W2, 3: W3})
        return data, data + numpy_reg_penalty(shim, regs)

    return loss_of


def test_torch_training_loss_matches_definition(small_config, synth_data,
                                                tmp_path):
    """The data and total loss recorded in the first epoch equal the weighted
    least-squares objective written out directly from its definition."""
    store = {}
    model, regs, tracker = run_torch_train(
        small_config, synth_data, tmp_path / "loss", max_epochs=1,
        checkpoint=1, optimizer_class=_CaptureOptimizer,
        optimizer_kwargs={"store": store})
    parts = [p.detach().cpu().numpy().copy() for p in store["params"]]
    data, total = torch_loss_parts(model, synth_data, regs)(parts)
    np.testing.assert_allclose(tracker["data_loss"][0], data, rtol=1e-10)
    np.testing.assert_allclose(tracker["total_loss"][0], total, rtol=1e-10)


def test_torch_training_gradient_matches_finite_differences(small_config,
                                                            synth_data,
                                                            tmp_path):
    """Autograd gradients from the training loop equal central differences of
    that same loss. Without this an optimizer comparison measures nothing."""
    store = {}
    model, regs, _ = run_torch_train(
        small_config, synth_data, tmp_path / "grad", max_epochs=1,
        checkpoint=1, optimizer_class=_CaptureOptimizer,
        optimizer_kwargs={"store": store})
    parts = [p.detach().cpu().numpy().copy() for p in store["params"]]
    assert all(g is not None for g in store["grads"][0]), \
        "every parameter must receive a gradient"
    loss_of = torch_loss_parts(model, synth_data, regs)

    shapes = [p.shape for p in parts]
    sizes = [p.size for p in parts]
    cuts = np.cumsum([0] + sizes)
    flat = np.concatenate([p.ravel() for p in parts])
    g_auto = np.concatenate([g.numpy().ravel() for g in store["grads"][0]])

    def unpack(v):
        return [v[cuts[i]:cuts[i + 1]].reshape(shapes[i])
                for i in range(len(parts))]

    f0 = loss_of(parts)[1]
    rng = np.random.default_rng(5)
    # sample within every parameter block, not just the largest
    idx = np.concatenate([rng.choice(np.arange(cuts[i], cuts[i + 1]),
                                     min(6, sizes[i]), replace=False)
                          for i in range(len(parts))])
    h = 1e-6
    for i in idx:
        vp, vm = flat.copy(), flat.copy()
        vp[i] += h
        vm[i] -= h
        fd = (loss_of(unpack(vp))[1] - loss_of(unpack(vm))[1]) / (2 * h)
        np.testing.assert_allclose(g_auto[i], fd, rtol=2e-5,
                                   atol=1e-6 * max(1.0, abs(f0)))


def test_torch_checkpoint_does_not_rebind_parameters(small_config, synth_data,
                                                     tmp_path):
    """The optimizer is handed its parameters once, so checkpointing must not
    replace the tensors the forward pass reads."""
    store = {}
    model, _, _ = run_torch_train(
        small_config, synth_data, tmp_path / "rebind", max_epochs=4,
        checkpoint=2, optimizer_class=_CaptureOptimizer,
        optimizer_kwargs={"store": store})
    live = list(model.coeff.values()) + list(model.pseudo_weights.values())
    assert len(live) == len(store["params"])
    for held, current in zip(store["params"], live):
        assert held is current
        assert current.requires_grad


@pytest.mark.parametrize("checkpoint", [1, 2, 3])
def test_torch_training_independent_of_checkpoint_frequency(small_config,
                                                            synth_data,
                                                            tmp_path,
                                                            checkpoint):
    """Checkpointing is bookkeeping: the loss trajectory must not depend on how
    often it happens. Rebinding the tensors froze training at the first one."""
    ref = run_torch_train(small_config, synth_data, tmp_path / "ref",
                          max_epochs=6, checkpoint=100,
                          optimizer_class=torch.optim.Adam,
                          optimizer_kwargs={"lr": 0.05})[2]
    got = run_torch_train(small_config, synth_data, tmp_path / f"ck{checkpoint}",
                          max_epochs=6, checkpoint=checkpoint,
                          optimizer_class=torch.optim.Adam,
                          optimizer_kwargs={"lr": 0.05})[2]
    for key in ("data_loss", "total_loss", "rmse_f"):
        a, b = ref[key], got[key]
        np.testing.assert_allclose(b[~np.isnan(b)], a[~np.isnan(a)], rtol=1e-12)
    assert got["total_loss"][~np.isnan(got["total_loss"])][-1] < \
        got["total_loss"][0]


def test_torch_decompress_matches_numpy_model(small_config, synth_data,
                                              tmp_path):
    """Restoring the tensors after decompression leaves self.coefficients
    exactly as the numpy implementation computes it."""
    model, _, _ = run_torch_train(small_config, synth_data,
                                  tmp_path / "decomp", max_epochs=2,
                                  checkpoint=1,
                                  optimizer_class=torch.optim.Adam,
                                  optimizer_kwargs={"lr": 0.05})
    ref = make_model(small_config)
    ref.load_data_coverage(synth_data["coverage"])
    ref.coeff = {k: v.detach().cpu().numpy() for k, v in model.coeff.items()}
    ref.pseudo_weights = {k: v.detach().cpu().numpy()
                          for k, v in model.pseudo_weights.items()}
    ref.decompress_alchemical_parameters()
    model.decompress_alchemical_parameters()
    np.testing.assert_array_equal(model.coefficients, ref.coefficients)
    assert isinstance(model.coeff[2], torch.Tensor)
