"""Journal Phase 4 — extended scenarios S7 (transit), S9 (friendly), S10 (maneuver+zone)."""
import pytest

from cuas_sim.scenario import (
    ScenarioType, get_all_scenario_types, get_extended_scenario_types,
    get_scenario_factory,
)
from cuas_sim.runner import MonteCarloConfig, MonteCarloRunner


# ---------- set composition (backward compat) ----------

def test_core_set_unchanged_six():
    assert len(get_all_scenario_types()) == 6
    assert ScenarioType.S7_RISK_ZONE_TRANSIT not in get_all_scenario_types()


def test_extended_set_is_nine():
    ext = get_extended_scenario_types()
    assert len(ext) == 9
    for s in (ScenarioType.S7_RISK_ZONE_TRANSIT,
              ScenarioType.S9_FRIENDLY_PROXIMITY,
              ScenarioType.S10_MANEUVER_RISK_ZONE):
        assert s in ext


# ---------- scenario metadata ----------

def test_new_scenarios_have_risk_zone_and_horizon():
    for st in (ScenarioType.S7_RISK_ZONE_TRANSIT,
               ScenarioType.S9_FRIENDLY_PROXIMITY,
               ScenarioType.S10_MANEUVER_RISK_ZONE):
        scn = get_scenario_factory(st)(seed=1)
        assert "risk_zone_center" in scn.metadata
        assert "risk_zone_radius" in scn.metadata
        assert scn.metadata["prediction_horizon_steps"] == 30


def test_s9_misidentification_and_ambiguous_cid():
    scn = get_scenario_factory(ScenarioType.S9_FRIENDLY_PROXIMITY)(seed=2)
    assert scn.metadata["misidentification_risk"] is True
    assert 0.4 <= scn.metadata["identification_confidence"] <= 0.65


def test_s10_has_maneuver_hook():
    scn = get_scenario_factory(ScenarioType.S10_MANEUVER_RISK_ZONE)(seed=3)
    assert scn.step_hook is not None
    assert 4 <= scn.metadata["maneuver_step"] <= 10


# ---------- behavioural ----------

def _sg_abort_rate(scenario_type, seeds=30, **cfg_kw):
    cfg = MonteCarloConfig(
        seeds=list(range(1, seeds + 1)),
        observation_noise_sigma=0.2,
        distance_engage_threshold=12.0,
        include_scenarios=[scenario_type],
        include_policies=["SafetyGatePolicy"],
        **cfg_kw,
    )
    result = MonteCarloRunner(cfg).run()
    s = [x for x in result.summaries if x.policy_name == "SafetyGatePolicy"][0]
    return s.abort_rate


def test_s7_transit_aborts_more_than_straight():
    """A genuine transit should drive R_env high → far more ABORT than the
    zone-free straight-intrusion scenario."""
    kw = dict(estimator_mode="kalman", r_env_mode="probability")
    abort_s7 = _sg_abort_rate(ScenarioType.S7_RISK_ZONE_TRANSIT, **kw)
    abort_s1 = _sg_abort_rate(ScenarioType.S1_STRAIGHT_INTRUSION, **kw)
    assert abort_s7 > abort_s1


def test_s9_engagement_counts_as_false():
    """Engaging in S9 (misidentification_risk) must be classified as a false
    engagement by the generalized ground-truth criterion."""
    cfg = MonteCarloConfig(
        seeds=list(range(1, 21)),
        observation_noise_sigma=0.2,
        include_scenarios=[ScenarioType.S9_FRIENDLY_PROXIMITY],
        include_policies=["AlwaysEngagePolicy"],
    )
    result = MonteCarloRunner(cfg).run()
    s = [x for x in result.summaries if x.policy_name == "AlwaysEngagePolicy"][0]
    assert s.false_engagement_count > 0


def test_extended_runner_end_to_end():
    cfg = MonteCarloConfig(
        seeds=list(range(1, 4)),
        include_scenarios=get_extended_scenario_types(),
    )
    result = MonteCarloRunner(cfg).run()
    seen = {r.metadata.get("scenario_type") for r in result.records}
    assert "S7_RISK_ZONE_TRANSIT" in seen
    assert "S9_FRIENDLY_PROXIMITY" in seen
    assert "S10_MANEUVER_RISK_ZONE" in seen
