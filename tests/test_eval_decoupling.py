"""Journal Phase 2 (eval decoupling) — evaluation 'unsafe' threshold is separate
from the policy decision threshold; probability R_env targets the keep-out disk."""
import pytest

from cuas_sim.types import State2D
from cuas_sim.prediction import PredictedState, TrajectoryPrediction
from cuas_sim.evaluators import EnvironmentalRiskEvaluator
from cuas_sim.runner import MonteCarloConfig, MonteCarloRunner


class _StubScenario:
    def __init__(self, center, radius):
        self.metadata = {"risk_zone_center": center, "risk_zone_radius": radius}


def _pred_grazing(sigma=0.4):
    # closest approach distance 2.0 from center (5,0)
    preds = [PredictedState(m, State2D(1.0 + 0.4 * m, 0.0, 0.4, 0.0), sigma, None) for m in range(1, 6)]
    return TrajectoryPrediction(0, 5, preds, True)


# ---------- keep-out buffer ----------

def test_keepout_buffer_increases_intrusion_probability():
    scn = _StubScenario(center=(5.0, 0.0), radius=1.0)
    p_no_buf = EnvironmentalRiskEvaluator(r_env_mode="probability", keepout_buffer=0.0).evaluate(scn, _pred_grazing()).risk_score
    p_buf = EnvironmentalRiskEvaluator(r_env_mode="probability", keepout_buffer=0.6).evaluate(scn, _pred_grazing()).risk_score
    # larger keep-out region ⇒ at least as much intrusion probability
    assert p_buf >= p_no_buf
    assert p_buf > 0.0


# ---------- eval threshold decoupling ----------

def test_default_eval_threshold_is_none_and_falls_back():
    cfg = MonteCarloConfig(seeds=[1])
    assert cfg.eval_risk_score_threshold is None


def test_eval_threshold_decoupled_from_policy_threshold():
    """With eval_risk_score_threshold fixed, the metrics logger uses the EVAL
    threshold, not the (different) policy decision threshold."""
    cfg = MonteCarloConfig(
        seeds=list(range(1, 4)),
        observation_noise_sigma=0.2,
        distance_engage_threshold=12.0,
        safety_risk_score_threshold=0.10,        # policy decision threshold
        eval_risk_score_threshold=0.70,          # evaluation yardstick (fixed)
    )
    result = MonteCarloRunner(cfg).run()
    assert result.logger_kwargs["risk_score_threshold"] == 0.70


def test_eval_threshold_none_uses_policy_threshold():
    cfg = MonteCarloConfig(
        seeds=list(range(1, 4)),
        safety_risk_score_threshold=0.70,
        eval_risk_score_threshold=None,
    )
    result = MonteCarloRunner(cfg).run()
    assert result.logger_kwargs["risk_score_threshold"] == 0.70
