"""Journal Phase 1 — tests for the Kalman estimator and NIS-based C_track."""
import math
import random

import pytest

from cuas_sim.types import State2D
from cuas_sim.observation import TargetObservation, ObservationModel
from cuas_sim.estimation import (
    TargetStateEstimator,
    KalmanTargetEstimator,
    ctrack_threshold_from_nis,
    ctrack_threshold_for_false_rate,
)
from cuas_sim.scenario import get_scenario_factory, get_all_scenario_types, ScenarioType
from cuas_sim.simulator import Simulator
from cuas_sim.runner import MonteCarloConfig, MonteCarloRunner


def _obs(step, x, y, vx=1.0, vy=0.0, detected=True):
    return TargetObservation(
        step_index=step,
        detected=detected,
        observed_state=State2D(x=x, y=y, vx=vx, vy=vy) if detected else None,
        tracking_confidence=0.9,
        identification_confidence=0.9,
        metadata={},
    )


# ---------- helper math ----------

def test_threshold_helpers():
    assert ctrack_threshold_from_nis(0.0) == pytest.approx(1.0)
    # NIS 9.21 (chi2_2 99%) -> ~0.01
    assert ctrack_threshold_from_nis(9.21) == pytest.approx(0.01, abs=1e-3)
    # Uniform property: threshold == false rate
    assert ctrack_threshold_for_false_rate(0.05) == 0.05


# ---------- first step / structure ----------

def test_first_step_confidence_one_and_covariance_present():
    est = KalmanTargetEstimator(time_step=0.1, measurement_sigma=0.2)
    e0 = est.estimate(_obs(0, 0.0, 0.0))
    assert e0.is_valid
    assert e0.tracking_confidence == pytest.approx(1.0)
    assert e0.metadata["nis"] == pytest.approx(0.0)
    assert e0.covariance is not None
    assert set(e0.covariance.keys()) == {"x", "y"}


def test_ctrack_equals_exp_minus_half_nis():
    est = KalmanTargetEstimator(time_step=0.1, measurement_sigma=0.2)
    est.estimate(_obs(0, 0.0, 0.0))
    e1 = est.estimate(_obs(1, 0.5, 0.3))  # deliberately off-track to make nis>0
    nis = e1.metadata["nis"]
    assert nis > 0
    assert e1.tracking_confidence == pytest.approx(math.exp(-0.5 * nis))


# ---------- backward compatibility ----------

def test_alpha_beta_has_no_covariance():
    est = TargetStateEstimator(time_step=0.1)
    e = est.estimate(_obs(0, 0.0, 0.0))
    assert e.covariance is None


def test_default_estimator_mode_is_alpha_beta():
    cfg = MonteCarloConfig(seeds=[1])
    assert cfg.estimator_mode == "alpha_beta"


# ---------- perfect-sensor behaviour ----------

def test_perfect_cv_observations_keep_confidence_high():
    """With noiseless CV observations the filter should not be 'surprised':
    innovations stay tiny, C_track near 1."""
    est = KalmanTargetEstimator(time_step=0.1, measurement_sigma=0.0)  # floored internally
    confs = []
    x = 0.0
    for k in range(15):
        e = est.estimate(_obs(k, x, 0.0, vx=1.0, vy=0.0))
        confs.append(e.tracking_confidence)
        x += 1.0 * 0.1  # true CV motion
    # after warmup, confidence should stay high (low NIS)
    assert min(confs[3:]) > 0.6


# ---------- dropout ----------

def test_dropout_forwards_tracking_confidence_and_marks_invalid():
    est = KalmanTargetEstimator(time_step=0.1, measurement_sigma=0.2)
    est.estimate(_obs(0, 0.0, 0.0))
    drop = TargetObservation(
        step_index=1, detected=False, observed_state=None,
        tracking_confidence=0.15, identification_confidence=0.9, metadata={},
    )
    e = est.estimate(drop)
    assert e.is_valid is False
    assert e.detected is False
    assert e.tracking_confidence == pytest.approx(0.15)


# ---------- NIS consistency (the Phase 1 deliverable) ----------

def _mean_nis(scenario_type, q=0.1, sigma=0.2, n_seeds=60, steps=20):
    factory = get_scenario_factory(scenario_type)
    sidx = get_all_scenario_types().index(scenario_type)
    vals = []
    for seed in range(1, n_seeds + 1):
        scenario = factory(seed=seed)
        rng = random.Random(seed * 1000 + sidx)
        obs_model = ObservationModel(noise_sigma=sigma, rng=rng)
        est = KalmanTargetEstimator(time_step=scenario.config.time_step, process_noise=q, measurement_sigma=sigma)
        sim = Simulator(config=scenario.config, target=scenario.target,
                        interceptor=scenario.interceptor, step_hook=scenario.step_hook)
        for k in range(steps):
            e = est.estimate(obs_model.observe(scenario, k))
            if e.is_valid and k > 0:
                vals.append(e.metadata["nis"])
            sim.step()
    return sum(vals) / len(vals)


def test_nis_consistency_cv_scenario_near_two():
    """Consistent filter on a CV target ⇒ mean NIS ≈ 2 (chi-square dof)."""
    mean_nis = _mean_nis(ScenarioType.S1_STRAIGHT_INTRUSION)
    assert 1.4 <= mean_nis <= 2.6, f"S1 mean NIS={mean_nis}"


def test_maneuver_raises_nis_above_cv():
    """S2 maneuver should make the filter 'surprised' ⇒ higher NIS than CV S1."""
    s1 = _mean_nis(ScenarioType.S1_STRAIGHT_INTRUSION)
    s2 = _mean_nis(ScenarioType.S2_EVASIVE_MANEUVER)
    assert s2 > s1


# ---------- end-to-end pipeline ----------

def test_runner_with_kalman_runs_and_is_nondegenerate():
    cfg = MonteCarloConfig(
        seeds=list(range(1, 11)),
        observation_noise_sigma=0.2,
        distance_engage_threshold=12.0,
        estimator_mode="kalman",
        # chi-square(2)-recalibrated thresholds (uniform p-value scale)
        safety_tracking_confidence_threshold=ctrack_threshold_for_false_rate(0.05),
        safety_tracking_abort_threshold=ctrack_threshold_for_false_rate(0.01),
    )
    result = MonteCarloRunner(cfg).run()
    assert len(result.records) > 0
    summaries = {s.policy_name: s for s in result.summaries}
    assert "SafetyGatePolicy" in summaries


def test_invalid_estimator_mode_raises():
    cfg = MonteCarloConfig(seeds=[1], estimator_mode="bogus")
    with pytest.raises(ValueError):
        MonteCarloRunner(cfg).run()
