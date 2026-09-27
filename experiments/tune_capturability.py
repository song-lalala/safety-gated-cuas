"""Phase 4 — tune capturability engagement horizon T_engage.

Diagnosis showed the M1 RWS loss is entirely from capturability being too
conservative in S5 (longer-range target → capture time ~5s vs T_engage=10 →
I≈0.5 at threshold). T_engage is the physical engagement window; sweep it and
check S5 recovers without making S6 (non-interceptable) spuriously engageable
in a way that costs RWS. σ=0.2, SafetyGate, core 6, N=1000, a_mode=time_proxy
(abort dynamics is RWS-neutral per the decomposition).

Reference: C(heuristic I) overall RWS = 1.2736.
"""
from cuas_sim.runner import MonteCarloConfig, MonteCarloRunner

N = 1000
SEEDS = list(range(1, N + 1))
T_SWEEP = [10.0, 15.0, 20.0, 30.0, 50.0]


def cfg(T):
    return MonteCarloConfig(
        seeds=SEEDS, observation_noise_sigma=0.2, distance_engage_threshold=12.0,
        include_policies=["SafetyGatePolicy"], eval_risk_score_threshold=0.7,
        estimator_mode="kalman", r_env_mode="probability", risk_keepout_buffer=0.6,
        i_mode="capturability", a_mode="time_proxy",
        interceptor_engagement_horizon_time=T,
        safety_tracking_confidence_threshold=0.05, safety_tracking_abort_threshold=0.01,
        safety_risk_score_threshold=0.05,
    )


def run(c):
    res = MonteCarloRunner(c).run()
    ov = [s for s in res.summaries if s.policy_name == "SafetyGatePolicy"][0]
    grp = {d["scenario_type"]: d for d in res.grouped_summaries_to_dicts()
           if d["policy_name"] == "SafetyGatePolicy"}
    return ov, grp


print("Reference: C heuristic-I overall RWS = 1.2736 (S5 scen_RWS 1.4434, S6 engage 0.0)")
print("=" * 78)
print(f"{'T_engage':>9} | {'overall RWS':>11} | {'S5 engage':>10} {'S5 RWS':>8} | {'S6 engage':>10} {'S6 RWS':>8}")
print("-" * 78)
for T in T_SWEEP:
    ov, grp = run(cfg(T))
    s5 = grp["S5_TRACKING_LOSS"]
    s6 = grp["S6_NON_INTERCEPTABLE_TARGET"]
    print(f"{T:>9.0f} | {ov.risk_weighted_score:>11.4f} | "
          f"{float(s5['engagement_attempt_rate']):>10.4f} {float(s5['risk_weighted_score']):>8.4f} | "
          f"{float(s6['engagement_attempt_rate']):>10.4f} {float(s6['risk_weighted_score']):>8.4f}")
print("=" * 78)
print("DONE")
