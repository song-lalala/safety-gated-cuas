"""Phase 4 verification — baseline equivalence + extended-set ablation (A' vs D').

Run:  python experiments/verify_phase4.py
Prints: (1) core baseline RWS vs known conference reference, (2) extended-set
SafetyGate overall + per-new-scenario (S7/S9/S10) for A' (baseline toggles) and
D' (KF + probability R_env + capturability I + abort dynamics).
"""
import sys
from cuas_sim.runner import MonteCarloConfig, MonteCarloRunner
from cuas_sim.scenario import get_extended_scenario_types

SEEDS = list(range(1, 1001))
NEW = ["S7_RISK_ZONE_TRANSIT", "S9_FRIENDLY_PROXIMITY", "S10_MANEUVER_RISK_ZONE"]

# Known conference reference (σ=0.2, N=1000, distance=12) — see phase0 doc.
REF = {
    "AlwaysEngagePolicy": 0.6615722532763235,
    "DistanceOnlyPolicy": 0.1595861055103054,
    "InterceptabilityOnlyPolicy": 0.3186681862829738,
    "SafetyGatePolicy": 0.7810114670163717,
}


def sg(result):
    return [s for s in result.summaries if s.policy_name == "SafetyGatePolicy"][0]


def grouped(result, scen):
    for d in result.grouped_summaries_to_dicts():
        if d.get("scenario_type") == scen and d["policy_name"] == "SafetyGatePolicy":
            return d
    return None


print("=" * 64)
print("1) CORE BASELINE equivalence (σ=0.2, N=1000, distance=12, defaults)")
core = MonteCarloRunner(MonteCarloConfig(
    seeds=SEEDS, observation_noise_sigma=0.2, distance_engage_threshold=12.0,
)).run()
ok = True
for s in core.summaries:
    exp = REF.get(s.policy_name)
    got = s.risk_weighted_score
    match = abs(got - exp) < 1e-9
    ok = ok and match
    print(f"   {s.policy_name:26s} RWS={got:.10f}  {'OK' if match else 'DIFF exp=%.10f'%exp}")
print(f"   => baseline reproduced: {'YES' if ok else 'NO'}")

common = dict(observation_noise_sigma=0.2, distance_engage_threshold=12.0,
              include_scenarios=get_extended_scenario_types())

print("=" * 64)
print("2) EXTENDED-SET ablation — SafetyGate")
configs = {
    "A' baseline": MonteCarloConfig(seeds=SEEDS, **common),
    "D' full": MonteCarloConfig(
        seeds=SEEDS, eval_risk_score_threshold=0.7,
        estimator_mode="kalman", r_env_mode="probability", risk_keepout_buffer=0.6,
        i_mode="capturability", a_mode="dynamics",
        safety_tracking_confidence_threshold=0.05, safety_tracking_abort_threshold=0.01,
        safety_risk_score_threshold=0.05, **common,
    ),
}
results = {}
for name, cfg in configs.items():
    results[name] = MonteCarloRunner(cfg).run()

print(f"\n   overall:  {'cell':12s} {'RWS':>9} {'unsafe/att':>11} {'abort_succ':>11} {'engage':>8} {'track':>8}")
for name, res in results.items():
    s = sg(res)
    print(f"            {name:12s} {s.risk_weighted_score:9.4f} {s.unsafe_engagement_per_attempt_rate:11.4f} "
          f"{s.abort_success_rate:11.4f} {s.engagement_attempt_rate:8.4f} {s.track_rate:8.4f}")

for scen in NEW:
    print(f"\n   {scen}:  {'cell':12s} {'abort':>8} {'engage':>8} {'unsafe/a':>9} {'false/a':>9} {'scen_RWS':>9}")
    for name, res in results.items():
        g = grouped(res, scen)
        print(f"            {name:12s} {float(g['abort_rate']):8.4f} {float(g['engagement_attempt_rate']):8.4f} "
              f"{float(g['unsafe_engagement_per_attempt_rate']):9.4f} {float(g['false_engagement_per_attempt_rate']):9.4f} "
              f"{float(g['risk_weighted_score']):9.4f}")

print("=" * 64)
print("DONE")
sys.exit(0 if ok else 1)
