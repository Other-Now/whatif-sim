import numpy as np
import pytest

from whatif.model import ModelSpec
from whatif.sim import Compiled
from whatif.solver import SolverError, solve


def decay(k):
    return lambda t, y: -k * y


def test_exponential_decay_matches_exact():
    ys, st = solve(decay(0.7), 0.0, np.array([[1.0]]), np.linspace(0, 10, 11), rtol=1e-9, atol=1e-12)
    exact = np.exp(-0.7 * np.linspace(0, 10, 11))
    assert np.max(np.abs(ys[0, :, 0] - exact)) < 1e-9
    assert st.rejected < st.steps


def test_oscillator_energy_and_phase():
    w = 2.0
    f = lambda t, y: np.stack([y[:, 1], -w * w * y[:, 0]], axis=1)
    T = 20.0
    ys, _ = solve(f, 0.0, np.array([[1.0, 0.0]]), np.array([T]), rtol=1e-10, atol=1e-13)
    assert abs(ys[0, 0, 0] - np.cos(w * T)) < 1e-8
    assert abs(ys[0, 0, 1] + w * np.sin(w * T)) < 1e-7


def test_error_tracks_tolerance():
    """Tighter tolerance -> smaller error, roughly proportionally."""
    r, K = 0.9, 10.0
    f = lambda t, y: r * y * (1 - y / K)
    exact = K / (1 + (K / 0.1 - 1) * np.exp(-r * 15))
    errs = []
    for tol in (1e-4, 1e-6, 1e-8):
        ys, _ = solve(f, 0.0, np.array([[0.1]]), np.array([15.0]), rtol=tol, atol=tol * 1e-3)
        errs.append(abs(ys[0, -1, 0] - exact))
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 1e-6


def test_fixed_step_order_is_five():
    T = 10.0
    errs = []
    for n in (40, 80, 160):
        ys, _ = solve(decay(0.7), 0.0, np.array([[1.0]]), np.array([T]), h_fixed=T / n)
        errs.append(abs(ys[0, -1, 0] - np.exp(-0.7 * T)))
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all((orders > 4.6) & (orders < 5.4)), orders


def test_batch_equals_one_at_a_time():
    """A shared step size across the batch must not change any sample's answer
    beyond the tolerance."""
    ks = np.array([0.1, 0.5, 2.0, 5.0])
    f = lambda t, y: -ks[:, None] * y
    ys, _ = solve(f, 0.0, np.ones((4, 1)), np.array([3.0]), rtol=1e-8, atol=1e-11)
    for i, k in enumerate(ks):
        alone, _ = solve(decay(k), 0.0, np.array([[1.0]]), np.array([3.0]), rtol=1e-8, atol=1e-11)
        assert abs(ys[i, 0, 0] - alone[0, 0, 0]) < 1e-8
        assert abs(ys[i, 0, 0] - np.exp(-3 * k)) < 1e-8


def test_step_underflow_raises():
    with pytest.raises(SolverError):
        solve(lambda t, y: y ** 3, 0.0, np.array([[10.0]]), np.array([1.0]))


def _two_state(sep, find, horizon=60.0, changes=()):
    return ModelSpec.model_validate({
        "name": "m", "time_unit": "month", "horizon": horizon, "dt_out": 1,
        "stocks": [{"name": "E", "unit": "kp", "initial": "E0"}, {"name": "U", "unit": "kp", "initial": "U0"}],
        "flows": [{"name": "h", "from": "U", "to": "E", "rate": "f * U"},
                  {"name": "s", "from": "E", "to": "U", "rate": "sep * E"}],
        "aux": [{"name": "u", "expr": "U / (E + U)", "unit": "1"}],
        "params": [{"name": "f", "unit": "1/month", "dist": {"kind": "fixed", "value": find}},
                   {"name": "sep", "unit": "1/month", "dist": {"kind": "fixed", "value": sep}},
                   {"name": "E0", "unit": "kp", "dist": {"kind": "fixed", "value": 95.0}},
                   {"name": "U0", "unit": "kp", "dist": {"kind": "fixed", "value": 5.0}}],
        "outputs": ["u"],
        "scenarios": [{"name": "sc", "changes": list(changes)}],
    })


def test_two_state_model_matches_closed_form():
    """Without feedback, u(t) = u* + (u0 - u*) exp(-(s+f) t), u* = s/(s+f)."""
    s, f = 0.01, 0.25
    spec = _two_state(s, f)
    comp = Compiled(spec)
    sim = comp.simulate(comp.center_params(1), None, rtol=1e-10, atol=1e-12)
    t = sim.t
    ustar = s / (s + f)
    exact = ustar + (0.05 - ustar) * np.exp(-(s + f) * t)
    assert np.max(np.abs(sim.outputs[0, :, 0] - exact)) < 1e-9


def test_scenario_breakpoint_is_exact():
    """find rate x0.8 from t=3: piecewise closed form, so the solver must land
    on t=3 exactly rather than step across the discontinuity."""
    s, f = 0.01, 0.25
    spec = _two_state(s, f, horizon=24.0, changes=[{"param": "f", "multiply": 0.8, "from_t": 3}])
    comp = Compiled(spec)
    sim = comp.simulate(comp.center_params(1), spec.scenarios[0], rtol=1e-10, atol=1e-12)
    u1 = s / (s + f)
    u3 = u1 + (0.05 - u1) * np.exp(-(s + f) * 3)
    f2 = 0.8 * f
    u2 = s / (s + f2)
    t = sim.t
    exact = np.where(t <= 3, u1 + (0.05 - u1) * np.exp(-(s + f) * t),
                     u2 + (u3 - u2) * np.exp(-(s + f2) * (t - 3)))
    assert np.max(np.abs(sim.outputs[0, :, 0] - exact)) < 1e-9
