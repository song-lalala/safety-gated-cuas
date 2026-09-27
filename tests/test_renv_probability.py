"""Journal Phase 2 — tests for covariance propagation and intrusion-probability R_env."""
import math

import pytest

from cuas_sim.types import State2D
from cuas_sim.observation import TargetObservation
from cuas_sim.estimation import KalmanTargetEstimator, TargetStateEstimator
from cuas_sim.prediction import ConstantVelocityPredictor
from cuas_sim.evaluators import gaussian_disk_probability, EnvironmentalRiskEvaluator
from cuas_sim.scenario import get_scenario_factory, ScenarioType
from cuas_sim.runner import MonteCarloConfig, MonteCarloRunner


def _obs(step, x, y, vx=1.0, vy=0.0):
    return TargetObservation(step, True, State2D(x, y, vx, vy), 0.9, 0.9, {})


# ---------- gaussian_disk_probability ----------

def test_disk_prob_centered_matches_rayleigh_closed_form():
    # N(0, σ²I) over disk radius R centered at 0:  P = 1 - exp(-R²/(2σ²))
    for sigma, R in [(1.0, 2.0), (0.5, 1.0), (2.0, 1.0)]:
        expected = 1.0 - math.exp(-(R * R) / (2.0 * sigma * sigma))
        got = gaussian_disk_probability(0.0, 0.0, sigma, sigma, 0.0, 0.0, R)
        assert got == pytest.approx(expected, abs=2e-3), (sigma, R, got, expected)


def test_disk_prob_far_mean_is_zero_and_inside_large_R_is_one():
    assert gaussian_disk_probability(20.0, 0.0, 0.3, 0.3, 0.0, 0.0, 1.0) == pytest.approx(0.0, abs=1e-6)
    assert gaussian_disk_probability(0.0, 0.0, 0.2, 0.2, 0.0, 0.0, 10.0) == pytest.approx(1.0, abs=1e-3)


def test_disk_prob_increases_with_sigma_when_mean_outside():
    # mean outside the disk → more spread leaks probability mass in
    p_small = gaussian_disk_probability(3.0, 0.0, 0.5, 0.5, 0.0, 0.0, 1.0)
    p_large = gaussian_disk_probability(3.0, 0.0, 2.0, 2.0, 0.0, 0.0, 1.0)
    assert p_large > p_small


# ---------- covariance propagation in the predictor ----------

def _kalman_estimate_after(steps, sigma=0.2):
    est = KalmanTargetEstimator(time_step=0.1, measurement_sigma=sigma)
    x = 0.0
    e = None
    for k in range(steps):
        e = est.estimate(_obs(k, x, 0.0))
        x += 0.1
    return e


def test_prediction_has_covariance_in_kalman_mode_and_grows():
    est = _kalman_estimate_after(6)
    pred = ConstantVelocityPredictor(horizon_steps=5, process_noise=0.1).predict(est, 0.1)
    cov_list = [p.position_covariance for p in pred.predictions]
    assert all(c is not None for c in cov_list)
    var_x = [c[0][0] for c in cov_list]
    # predicted position variance must grow with horizon
    assert all(var_x[i] < var_x[i + 1] for i in range(len(var_x) - 1))


def test_prediction_no_covariance_in_alpha_beta_mode():
    est = TargetStateEstimator(time_step=0.1)
    e = est.estimate(_obs(0, 0.0, 0.0))
    e = est.estimate(_obs(1, 0.1, 0.0))
    pred = ConstantVelocityPredictor(horizon_steps=5).predict(e, 0.1)
    assert all(p.position_covariance is None for p in pred.predictions)
    # legacy heuristic uncertainty still present
    assert pred.predictions[0].uncertainty == pytest.approx((1.0 - e.tracking_confidence) + 0.05)


# ---------- R_env probability behaviour ----------

class _StubScenario:
    """Minimal scenario exposing only the risk-zone metadata the evaluator reads."""
    def __init__(self, center, radius):
        self.metadata = {"risk_zone_center": center, "risk_zone_radius": radius}


def _build_pred_outside_zone(sigma_scale):
    """Trajectory that passes the zone at (5,0) r=1 with closest approach
    distance 2 (i.e. OUTSIDE the zone). Isotropic σ via the α-β fallback
    scalar uncertainty. With the mean outside, larger σ ⇒ more mass intrudes."""
    from cuas_sim.prediction import PredictedState, TrajectoryPrediction
    preds = []
    for m in range(1, 6):
        x = 1.0 + 0.4 * m  # m=5 → x=3.0, dist from center (5,0) = 2.0 (> r=1)
        preds.append(PredictedState(
            step_offset=m,
            state=State2D(x=x, y=0.0, vx=0.4, vy=0.0),
            uncertainty=sigma_scale,
            position_covariance=None,  # α-β fallback → isotropic σ = uncertainty
        ))
    return TrajectoryPrediction(0, 5, preds, True)


def test_renv_probability_increases_with_uncertainty():
    scn = _StubScenario(center=(5.0, 0.0), radius=1.0)
    ev = EnvironmentalRiskEvaluator(r_env_mode="probability")
    low = ev.evaluate(scn, _build_pred_outside_zone(0.3)).risk_score
    high = ev.evaluate(scn, _build_pred_outside_zone(1.5)).risk_score
    # uncertainty-aware: larger σ ⇒ more probability mass intrudes ⇒ higher R_env
    assert high > low > 0.0


def test_renv_probability_in_unit_interval():
    scn = _StubScenario(center=(5.0, 0.0), radius=1.0)
    ev = EnvironmentalRiskEvaluator(r_env_mode="probability")
    r = ev.evaluate(scn, _build_pred_outside_zone(0.8)).risk_score
    assert 0.0 <= r <= 1.0


def test_renv_invalid_mode_raises():
    scn = _StubScenario(center=(5.0, 0.0), radius=1.0)
    ev = EnvironmentalRiskEvaluator(r_env_mode="bogus")
    with pytest.raises(ValueError):
        ev.evaluate(scn, _build_pred_outside_zone(0.5))


# ---------- defaults / end-to-end ----------

def test_default_r_env_mode_is_linear():
    assert MonteCarloConfig(seeds=[1]).r_env_mode == "linear"


def test_runner_kalman_probability_end_to_end():
    cfg = MonteCarloConfig(
        seeds=list(range(1, 11)),
        observation_noise_sigma=0.2,
        distance_engage_threshold=12.0,
        estimator_mode="kalman",
        r_env_mode="probability",
        safety_tracking_confidence_threshold=0.05,
        safety_tracking_abort_threshold=0.01,
    )
    result = MonteCarloRunner(cfg).run()
    assert len(result.records) > 0
