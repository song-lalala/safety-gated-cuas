"""Journal Phase 3 (M1) — capturability-based I and abort-dynamics A_abort."""
import math
from types import SimpleNamespace

import pytest

from cuas_sim.types import State2D
from cuas_sim.config import Config
from cuas_sim.estimation import TargetEstimate
from cuas_sim.prediction import PredictedState, TrajectoryPrediction
from cuas_sim.evaluators import InterceptabilityEvaluator, AbortFeasibilityEvaluator
from cuas_sim.runner import MonteCarloConfig, MonteCarloRunner


def _est(tx, ty, tvx, tvy):
    return TargetEstimate(
        step_index=1,
        estimated_state=State2D(tx, ty, tvx, tvy),
        detected=True, is_valid=True,
        tracking_confidence=0.9, identification_confidence=0.9,
    )


def _scen_I(ix=0.0, iy=0.0, metadata=None):
    return SimpleNamespace(interceptor=SimpleNamespace(state=State2D(ix, iy, 0.0, 0.0)),
                           metadata=metadata or {})


# ---------- capturability I ----------

def test_capturability_faster_approaching_is_interceptable():
    ev = InterceptabilityEvaluator(i_mode="capturability", engagement_horizon_time=10.0, default_v_interceptor=2.0)
    # target at (10,0) approaching interceptor at origin, v_t=1 < V_I=2
    a = ev.evaluate(_scen_I(), _est(10.0, 0.0, -1.0, 0.0))
    assert a.interceptability_score > 0.0
    assert a.metadata["capture_time"] is not None and a.metadata["capture_time"] > 0


def test_capturability_faster_receding_target_not_interceptable():
    ev = InterceptabilityEvaluator(i_mode="capturability", default_v_interceptor=1.5)
    # target receding at v_t=2 > V_I=1.5 → no capture
    a = ev.evaluate(_scen_I(), _est(10.0, 0.0, 2.0, 0.0))
    assert a.interceptability_score == 0.0


def test_capturability_closer_target_scores_higher():
    ev = InterceptabilityEvaluator(i_mode="capturability", default_v_interceptor=2.0)
    near = ev.evaluate(_scen_I(), _est(5.0, 0.0, -1.0, 0.0)).interceptability_score
    far = ev.evaluate(_scen_I(), _est(15.0, 0.0, -1.0, 0.0)).interceptability_score
    assert near > far > 0.0


def test_capturability_invalid_mode_raises():
    ev = InterceptabilityEvaluator(i_mode="bogus")
    with pytest.raises(ValueError):
        ev.evaluate(_scen_I(), _est(10.0, 0.0, -1.0, 0.0))


# ---------- abort dynamics ----------

def _scen_zone(dt=0.1):
    return SimpleNamespace(
        config=Config(time_step=dt),
        metadata={"risk_zone_center": (5.0, 0.0), "risk_zone_radius": 1.0},
    )


def _pred_enters_at(step, dt=0.1):
    # build predictions; the `step`-th enters the unsafe radius (1.0 + 0.5 = 1.5)
    preds = []
    for m in range(1, 6):
        # place predicted point inside unsafe radius from m == step onward
        x = 5.0 if m >= step else 0.0
        preds.append(PredictedState(m, State2D(x, 0.0, 1.0, 0.0), 0.1, None))
    return TrajectoryPrediction(0, 5, preds, True)


def test_abort_dynamics_uses_physical_required_time():
    ev = AbortFeasibilityEvaluator(
        a_mode="dynamics", interceptor_max_speed=2.0, interceptor_max_accel=4.0,
        interceptor_reaction_delay=0.2, feasibility_threshold=0.5,
    )
    scen = _scen_zone(dt=0.1)
    a = ev.evaluate(scen, _pred_enters_at(3))   # enters at offset 3 → t_avail = 0.3
    # t_required = 0.2 + 2/4 = 0.7 → score = 0.3/0.7 ≈ 0.4286
    assert a.metadata["t_required"] == pytest.approx(0.7)
    assert a.metadata["t_avail"] == pytest.approx(0.3)
    assert a.abort_feasibility_score == pytest.approx(0.3 / 0.7, abs=1e-6)


def test_abort_dynamics_differs_from_time_proxy():
    scen = _scen_zone()
    pred = _pred_enters_at(3)
    proxy = AbortFeasibilityEvaluator(a_mode="time_proxy").evaluate(scen, pred).abort_feasibility_score
    dyn = AbortFeasibilityEvaluator(a_mode="dynamics").evaluate(scen, pred).abort_feasibility_score
    assert proxy != dyn


def test_abort_dynamics_no_entry_is_fully_feasible():
    ev = AbortFeasibilityEvaluator(a_mode="dynamics")
    # all predictions far from zone → no entry → score 1.0
    preds = [PredictedState(m, State2D(-10.0, 0.0, 1.0, 0.0), 0.1, None) for m in range(1, 6)]
    a = ev.evaluate(_scen_zone(), TrajectoryPrediction(0, 5, preds, True))
    assert a.abort_feasibility_score == 1.0


def test_abort_invalid_mode_raises():
    ev = AbortFeasibilityEvaluator(a_mode="bogus")
    with pytest.raises(ValueError):
        ev.evaluate(_scen_zone(), _pred_enters_at(3))


# ---------- defaults / end-to-end ----------

def test_m1_defaults_are_baseline():
    cfg = MonteCarloConfig(seeds=[1])
    assert cfg.i_mode == "heuristic"
    assert cfg.a_mode == "time_proxy"


def test_runner_m1_end_to_end():
    cfg = MonteCarloConfig(
        seeds=list(range(1, 11)),
        observation_noise_sigma=0.2,
        distance_engage_threshold=12.0,
        estimator_mode="kalman",
        i_mode="capturability",
        a_mode="dynamics",
        safety_tracking_confidence_threshold=0.05,
        safety_tracking_abort_threshold=0.01,
    )
    result = MonteCarloRunner(cfg).run()
    assert len(result.records) > 0
    assert any(s.policy_name == "SafetyGatePolicy" for s in result.summaries)
