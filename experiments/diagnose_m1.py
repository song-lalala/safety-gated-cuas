"""Phase 4 — diagnose why M1 (capturability I + abort dynamics) lowers RWS.

Decomposes M1 into its two parts and breaks the effect down by scenario, at
σ=0.2 (SafetyGate, core 6, N=1000). All share KF + prob R_env + recalibrated
C_track thresholds; only i_mode / a_mode vary.

    C    : i=heuristic,     a=time_proxy   (M1 off  — reference)
    capt : i=capturability, a=time_proxy   (capturability only)
    dyn  : i=heuristic,     a=dynamics     (abort dynamics only)
    D    : i=capturability, a=dynamics     (both = full M1)
"""
from cuas_sim.runner import MonteCarloConfig, MonteCarloRunner

N = 1000
SEEDS = list(range(1, N + 1))
SCENS = ["S1_STRAIGHT_INTRUSION", "S2_EVASIVE_MANEUVER", "S3_LOW_IDENTIFICATION_CONFIDENCE",
         "S4_RISK_ZONE_PROXIMITY", "S5_TRACKING_LOSS", "S6_NON_INTERCEPTABLE_TARGET"]


def cfg(i_mode, a_mode):
    return MonteCarloConfig(
        seeds=SEEDS, observation_noise_sigma=0.2, distance_engage_threshold=12.0,
        include_policies=["SafetyGatePolicy"], eval_risk_score_threshold=0.7,
        estimator_mode="kalman", r_env_mode="probability", risk_keepout_buffer=0.6,
        i_mode=i_mode, a_mode=a_mode,
        safety_tracking_confidence_threshold=0.05, safety_tracking_abort_threshold=0.01,
        safety_risk_score_threshold=0.05,
    )


CONFIGS = {
    "C(off)":  cfg("heuristic", "time_proxy"),
    "capt":    cfg("capturability", "time_proxy"),
    "dyn":     cfg("heuristic", "dynamics"),
    "D(both)": cfg("capturability", "dynamics"),
}


def run(c):
    res = MonteCarloRunner(c).run()
    overall = [s for s in res.summaries if s.policy_name == "SafetyGatePolicy"][0]
    grp = {}
    for d in res.grouped_summaries_to_dicts():
        if d["policy_name"] == "SafetyGatePolicy":
            grp[d["scenario_type"]] = d
    return overall, grp


results = {name: run(c) for name, c in CONFIGS.items()}

print("=" * 80)
print("OVERALL SafetyGate RWS (σ=0.2):")
for name, (ov, _) in results.items():
    print(f"   {name:9s} RWS={ov.risk_weighted_score:7.4f}  unsafe/att={ov.unsafe_engagement_per_attempt_rate:.4f}  "
          f"engage={ov.engagement_attempt_rate:.4f}  safe_cap={ov.safe_opportunity_capture_rate:.4f}")
print("   ΔRWS vs C(off):  " + "  ".join(
    f"{name}={results[name][0].risk_weighted_score - results['C(off)'][0].risk_weighted_score:+.4f}"
    for name in ["capt", "dyn", "D(both)"]))

print("=" * 80)
print("PER-SCENARIO engage_rate / safe_capture / scen_RWS:")
hdr = "scenario".ljust(34) + "".join(f"{n:>11}" for n in CONFIGS)
for metric, key in [("engage", "engagement_attempt_rate"),
                    ("safe_cap", "safe_opportunity_capture_rate"),
                    ("scen_RWS", "risk_weighted_score")]:
    print(f"\n[{metric}]")
    print(hdr)
    for s in SCENS:
        row = s.ljust(34)
        for name in CONFIGS:
            g = results[name][1].get(s)
            row += f"{float(g[key]):>11.4f}" if g else f"{'-':>11}"
        print(row)
print("=" * 80)
print("DONE")
