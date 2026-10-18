import numpy as np

from whatif.mc import RunConfig, convergence, first_converged, run_batch, run_local, t_quantile_975
from whatif.sim import Compiled
from whatif.sobol import indices


def test_t_quantile():
    # reference values of Student t 0.975
    for df, ref, tol in ((5, 2.5706, 3e-3), (9, 2.2622, 5e-4), (29, 2.0452, 1e-4), (100, 1.9840, 1e-4)):
        assert abs(t_quantile_975(df) - ref) < tol


def test_batches_are_pure_functions_of_index(labor):
    comp = Compiled(labor)
    cfg = RunConfig(scenario="find_rate_down_20", batch_size=64)
    a = run_batch(comp, cfg, 7)
    run_batch(comp, cfg, 3)                 # unrelated work in between
    b = run_batch(Compiled(labor), cfg, 7)
    assert a.base.tobytes() == b.base.tobytes() and a.scen.tobytes() == b.scen.tobytes()
    assert run_batch(comp, cfg, 8).base.tobytes() != a.base.tobytes()


def test_local_run_is_reproducible(labor):
    cfg = RunConfig(scenario="find_rate_down_20", batch_size=64, tol=5e-4)
    assert run_local(labor, cfg)["sha256"] == run_local(labor, cfg)["sha256"]


def test_stopping_uses_smallest_converged_prefix(labor):
    comp = Compiled(labor)
    cfg = RunConfig(scenario="find_rate_down_20", batch_size=64, tol=2e-4, min_batches=5)
    stats = [run_batch(comp, cfg, i).stats for i in range(40)]
    m = first_converged(stats, cfg)
    assert m is not None
    assert convergence(stats[:m], cfg.tol)["converged"]
    assert all(not convergence(stats[:k], cfg.tol)["converged"] for k in range(cfg.min_batches, m))
    # run_local stops at the same place
    assert run_local(labor, cfg)["batches"] == m


def test_effect_is_zero_for_identity_scenario(labor_json):
    from whatif.model import ModelSpec
    labor_json["scenarios"][0]["changes"][0]["multiply"] = 1.0
    spec = ModelSpec.model_validate(labor_json)
    res = run_local(spec, RunConfig(scenario="find_rate_down_20", batch_size=64, tol=1.0))
    # Common random numbers: zero up to rounding. Not bit-zero, because the
    # scenario run is split into segments at from_t=3 and so takes different
    # solver steps than the baseline. The bound is the ODE error, not sampling
    # noise (measured: mean 4.5e-14, p95 1.1e-12; a real effect is ~1e-2).
    assert abs(res["tracked"]["mean"]) < 1e-9
    assert abs(res["tracked"]["p95"]) < 1e-9


def test_batch_means_agree_with_iid_se(labor):
    """Samples are i.i.d., so the batch-means CI should be close to the naive one."""
    cfg = RunConfig(scenario="find_rate_down_20", batch_size=128, tol=0.0, max_batches=60)
    c = run_local(labor, cfg)["convergence"]["mean"]
    assert 0.6 < c["halfwidth"] / c["naive_halfwidth"] < 1.5


def test_sobol_ishigami():
    a, b = 7.0, 0.1
    f = lambda X: np.sin(X[:, 0]) + a * np.sin(X[:, 1]) ** 2 + b * X[:, 2] ** 4 * np.sin(X[:, 0])
    r = indices(lambda g, n: g.uniform(-np.pi, np.pi, (n, 3)), f, 3, 20000, n_boot=50)
    for got, want in zip(r["S1"], [0.3139, 0.4424, 0.0]):
        assert abs(got - want) < 0.03
    for got, want in zip(r["ST"], [0.5576, 0.4424, 0.2437]):
        assert abs(got - want) < 0.03
    assert r["evaluations"] == 20000 * 5
