import argparse
import sys
import os
import datetime
from typing import List, Optional, Dict, Any

# Assuming cuas_sim is installed or in PYTHONPATH
from cuas_sim.scenario import ScenarioType, get_extended_scenario_types
from cuas_sim.runner import MonteCarloConfig, MonteCarloRunner
from cuas_sim.results import export_monte_carlo_result, write_json

def is_contiguous_seed_range(seeds: List[int]) -> bool:
    if not seeds:
        return False
    if len(seeds) == 1:
        return True
    
    # Check if elements are sorted and contiguous
    for i in range(1, len(seeds)):
        if seeds[i] != seeds[i-1] + 1:
            return False
            
    return True

def serialize_seed_config(seeds: List[int]) -> Dict[str, Any]:
    if not seeds:
        return {
            "seed_mode": "empty",
            "num_seeds": 0,
            "seeds": []
        }
        
    if is_contiguous_seed_range(seeds):
        return {
            "seed_mode": "range",
            "seed_start": seeds[0],
            "seed_end": seeds[-1],
            "seed_step": 1,
            "num_seeds": len(seeds)
        }
        
    return {
        "seed_mode": "explicit",
        "num_seeds": len(seeds),
        "seeds": seeds
    }

def parse_scenario_names(value: Optional[str]) -> Optional[List[ScenarioType]]:
    if not value:
        return None
        
    scenario_names = [v.strip() for v in value.split(",") if v.strip()]
    if not scenario_names:
        return None
        
    scenarios = []
    for name in scenario_names:
        try:
            scenarios.append(ScenarioType(name))
        except ValueError:
            raise ValueError(f"Invalid ScenarioType: {name}")
            
    return scenarios

def parse_policy_names(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
        
    policies = [v.strip() for v in value.split(",") if v.strip()]
    if not policies:
        return None
        
    return policies

def build_config_from_args(args: argparse.Namespace) -> MonteCarloConfig:
    seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
    
    include_scenarios = parse_scenario_names(args.scenarios)
    if include_scenarios is None and getattr(args, "scenario_set", "core") == "extended":
        include_scenarios = get_extended_scenario_types()
    include_policies = parse_policy_names(args.policies)
    
    return MonteCarloConfig(
        seeds=seeds,
        steps_per_trial=args.steps_per_trial,
        prediction_horizon_steps=args.prediction_horizon_steps,
        s4_prediction_horizon_steps=args.s4_prediction_horizon_steps,
        distance_engage_threshold=args.distance_engage_threshold,
        safety_tracking_abort_threshold=args.safety_tracking_abort_threshold,
        safety_tracking_confidence_threshold=args.safety_tracking_confidence_threshold,
        safety_identification_confidence_threshold=args.safety_identification_confidence_threshold,
        safety_risk_score_threshold=args.safety_risk_score_threshold,
        safety_interceptability_threshold=args.safety_interceptability_threshold,
        safety_abort_feasibility_threshold=args.safety_abort_feasibility_threshold,
        abort_horizon_steps=args.abort_horizon_steps,
        abort_risk_zone_safety_buffer=args.abort_risk_zone_safety_buffer,
        risk_k_sigma=args.risk_k_sigma,
        risk_abort_margin=args.risk_abort_margin,
        risk_clearance_falloff=args.risk_clearance_falloff,
        observation_noise_sigma=args.observation_noise_sigma,
        observation_velocity_noise_sigma=args.observation_velocity_noise_sigma,
        # Journal Phase 1 additions are optional: fall back to baseline defaults
        # so callers constructing a Namespace by hand (e.g. older tests) keep working.
        estimator_mode=getattr(args, "estimator_mode", "alpha_beta"),
        kalman_process_noise=getattr(args, "kalman_process_noise", 0.1),
        kalman_measurement_sigma=getattr(args, "kalman_measurement_sigma", None),
        r_env_mode=getattr(args, "r_env_mode", "linear"),
        eval_risk_score_threshold=getattr(args, "eval_risk_score_threshold", None),
        risk_keepout_buffer=getattr(args, "risk_keepout_buffer", 0.6),
        i_mode=getattr(args, "i_mode", "heuristic"),
        a_mode=getattr(args, "a_mode", "time_proxy"),
        interceptor_max_speed=getattr(args, "interceptor_max_speed", 2.0),
        interceptor_max_accel=getattr(args, "interceptor_max_accel", 4.0),
        interceptor_reaction_delay=getattr(args, "interceptor_reaction_delay", 0.2),
        interceptor_engagement_horizon_time=getattr(args, "interceptor_engagement_horizon_time", 20.0),
        late_abort_lag_steps=args.late_abort_lag_steps,
        weight_safe_capture=args.weight_safe_capture,
        weight_abort_success=args.weight_abort_success,
        weight_unsafe=args.weight_unsafe,
        weight_false=args.weight_false,
        weight_late_abort=args.weight_late_abort,
        ctrack_window=getattr(args, "ctrack_window", 1),
        nis_gate_threshold=getattr(args, "nis_gate_threshold", None),
        abort_latching=getattr(args, "abort_latching", False),
        include_scenarios=include_scenarios,
        include_policies=include_policies
    )

def config_to_dict(config: MonteCarloConfig) -> Dict[str, Any]:
    scenarios = None
    if config.include_scenarios is not None:
        scenarios = [s.value if hasattr(s, "value") else str(s) for s in config.include_scenarios]
        
    seed_config = serialize_seed_config(config.seeds)
    
    result = {
        **seed_config,
        "steps_per_trial": config.steps_per_trial,
        "prediction_horizon_steps": config.prediction_horizon_steps,
        "s4_prediction_horizon_steps": config.s4_prediction_horizon_steps,
        "distance_engage_threshold": config.distance_engage_threshold,
        "safety_tracking_abort_threshold": config.safety_tracking_abort_threshold,
        "safety_tracking_confidence_threshold": config.safety_tracking_confidence_threshold,
        "safety_identification_confidence_threshold": config.safety_identification_confidence_threshold,
        "safety_risk_score_threshold": config.safety_risk_score_threshold,
        "safety_interceptability_threshold": config.safety_interceptability_threshold,
        "safety_abort_feasibility_threshold": config.safety_abort_feasibility_threshold,
        "abort_horizon_steps": config.abort_horizon_steps,
        "abort_risk_zone_safety_buffer": config.abort_risk_zone_safety_buffer,
        "risk_k_sigma": config.risk_k_sigma,
        "risk_abort_margin": config.risk_abort_margin,
        "risk_clearance_falloff": config.risk_clearance_falloff,
        "observation_noise_sigma": config.observation_noise_sigma,
        "observation_velocity_noise_sigma": config.observation_velocity_noise_sigma,
        "estimator_mode": config.estimator_mode,
        "kalman_process_noise": config.kalman_process_noise,
        "kalman_measurement_sigma": config.kalman_measurement_sigma,
        "r_env_mode": config.r_env_mode,
        "eval_risk_score_threshold": config.eval_risk_score_threshold,
        "risk_keepout_buffer": config.risk_keepout_buffer,
        "i_mode": config.i_mode,
        "a_mode": config.a_mode,
        "interceptor_max_speed": config.interceptor_max_speed,
        "interceptor_max_accel": config.interceptor_max_accel,
        "interceptor_reaction_delay": config.interceptor_reaction_delay,
        "interceptor_engagement_horizon_time": config.interceptor_engagement_horizon_time,
        "late_abort_lag_steps": config.late_abort_lag_steps,
        "weight_safe_capture": config.weight_safe_capture,
        "weight_abort_success": config.weight_abort_success,
        "weight_unsafe": config.weight_unsafe,
        "weight_false": config.weight_false,
        "weight_late_abort": config.weight_late_abort,
        "include_scenarios": scenarios,
        "include_policies": config.include_policies
    }
    return result

def make_run_output_dir(base_output_dir: str, run_name: Optional[str]) -> str:
    if run_name is None:
        run_name = f"monte_carlo_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return os.path.join(base_output_dir, run_name)

def run_experiment(args: argparse.Namespace) -> Dict[str, str]:
    config = build_config_from_args(args)
    
    print(f"Starting Monte Carlo runner with {len(config.seeds)} seeds...")
    runner = MonteCarloRunner(config)
    result = runner.run()
    
    output_dir = make_run_output_dir(args.output_dir, args.run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    run_config_path = os.path.join(output_dir, "run_config.json")
    write_json(config_to_dict(config), run_config_path)
    
    print(f"Exporting results to {output_dir}...")
    paths = export_monte_carlo_result(result, output_dir)
    paths["run_config_json"] = run_config_path
    
    return paths

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run CUAS Monte Carlo Simulation")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Base output directory")
    parser.add_argument("--run-name", type=str, default=None, help="Specific run name (timestamp will be used if None)")
    parser.add_argument("--num-seeds", type=int, default=10, help="Number of seeds to run")
    parser.add_argument("--start-seed", type=int, default=1, help="Starting seed value")
    parser.add_argument("--steps-per-trial", type=int, default=20, help="Steps per simulation trial")
    parser.add_argument("--prediction-horizon-steps", type=int, default=5, help="Normal prediction horizon steps")
    parser.add_argument("--s4-prediction-horizon-steps", type=int, default=30, help="Prediction horizon steps for S4 scenario")
    parser.add_argument("--distance-engage-threshold", type=float, default=5.0, help="DistanceOnlyPolicy engage threshold")
    parser.add_argument("--safety-tracking-abort-threshold", type=float, default=0.3, help="SafetyGate tracking abort threshold")
    parser.add_argument("--safety-tracking-confidence-threshold", type=float, default=0.6, help="SafetyGate tracking confidence threshold")
    parser.add_argument("--safety-identification-confidence-threshold", type=float, default=0.6, help="SafetyGate identification confidence threshold")
    parser.add_argument("--safety-risk-score-threshold", type=float, default=0.7, help="SafetyGate risk score threshold")
    parser.add_argument("--safety-interceptability-threshold", type=float, default=0.5, help="SafetyGate interceptability threshold")
    parser.add_argument("--safety-abort-feasibility-threshold", type=float, default=0.5, help="SafetyGate abort-feasibility threshold (A_abort < this → ABORT)")
    parser.add_argument("--abort-horizon-steps", type=int, default=10, help="Number of prediction steps used as the abort decision horizon")
    parser.add_argument("--abort-risk-zone-safety-buffer", type=float, default=0.5, help="Extra buffer (normalized units) added to risk zone radius for abort feasibility evaluation")
    parser.add_argument("--risk-k-sigma", type=float, default=1.0, help="M3: weight on prediction uncertainty in effective unsafe radius (R_eff = R_zone + k_sigma*sigma_pred + abort_margin). 0 disables.")
    parser.add_argument("--risk-abort-margin", type=float, default=0.0, help="M3: optional kinematic margin added to unsafe radius")
    parser.add_argument("--risk-clearance-falloff", type=float, default=2.0, help="M3: distance over which risk_score linearly decays from 1.0 to 0.0")
    parser.add_argument("--observation-noise-sigma", type=float, default=0.0, help="Gaussian sigma for position observation noise (per axis). 0 = perfect sensor.")
    parser.add_argument("--observation-velocity-noise-sigma", type=float, default=0.0, help="Gaussian sigma for velocity observation noise (per axis). 0 = perfect velocity.")
    parser.add_argument("--estimator-mode", type=str, default="alpha_beta", choices=["alpha_beta", "kalman"], help="Estimator: alpha_beta (baseline) or kalman (NIS-based C_track).")
    parser.add_argument("--kalman-process-noise", type=float, default=0.1, help="Kalman DWNA acceleration PSD q (tuned for NIS consistency).")
    parser.add_argument("--kalman-measurement-sigma", type=float, default=None, help="Kalman per-axis measurement std. Default None → use observation-noise-sigma.")
    parser.add_argument("--r-env-mode", type=str, default="linear", choices=["linear", "probability"], help="R_env: linear effective-radius (baseline) or covariance-based intrusion probability.")
    parser.add_argument("--eval-risk-score-threshold", type=float, default=None, help="Ground-truth 'unsafe' threshold for EVALUATION, decoupled from the policy decision threshold. Default None → use safety-risk-score-threshold (baseline). Fix at 0.7 when sweeping policy threshold.")
    parser.add_argument("--risk-keepout-buffer", type=float, default=0.6, help="Physical keep-out margin b: probability R_env integrates over R_zone+b (0.6 matches eval unsafe region at eval θ=0.7). Probability mode only.")
    parser.add_argument("--i-mode", type=str, default="heuristic", choices=["heuristic", "capturability"], help="Interceptability: heuristic product (baseline) or pursuit-evasion capture-time (M1).")
    parser.add_argument("--a-mode", type=str, default="time_proxy", choices=["time_proxy", "dynamics"], help="A_abort: time proxy N_remain/N_norm (baseline) or abort dynamics t_avail/t_required (M1).")
    parser.add_argument("--interceptor-max-speed", type=float, default=2.0, help="M1: interceptor max speed V_I.")
    parser.add_argument("--interceptor-max-accel", type=float, default=4.0, help="M1: interceptor max accel/decel a_max (abort).")
    parser.add_argument("--interceptor-reaction-delay", type=float, default=0.2, help="M1: interceptor reaction delay τ (s).")
    parser.add_argument("--interceptor-engagement-horizon-time", type=float, default=20.0, help="M1: engagement horizon T_engage (s) for capturability.")
    parser.add_argument("--ctrack-window", type=int, default=1, help="W4: time-averaged NIS window W (1 = single-sample, the submitted form). Sum of last W NIS is chi-square(2W), so C_track stays a p-value.")
    parser.add_argument("--nis-gate-threshold", type=float, default=None, help="W4: chi-square measurement gate; a step whose 2-D NIS exceeds this discards the measurement (9.21 = chi2(2) 99%%). None = off.")
    parser.add_argument("--abort-latching", action="store_true", help="W4: make ABORT absorbing within a trial (a fielded system would not un-break-off).")
    parser.add_argument("--late-abort-lag-steps", type=int, default=2, help="Steps after first-required-abort before policy abort counts as late")
    parser.add_argument("--weight-safe-capture", type=float, default=1.0, help="Risk-weighted score weight on safe opportunity capture rate")
    parser.add_argument("--weight-abort-success", type=float, default=0.5, help="Risk-weighted score weight on abort success rate")
    parser.add_argument("--weight-unsafe", type=float, default=2.0, help="Risk-weighted score penalty weight on unsafe-engagement-per-attempt rate")
    parser.add_argument("--weight-false", type=float, default=2.0, help="Risk-weighted score penalty weight on false-engagement-per-attempt rate")
    parser.add_argument("--weight-late-abort", type=float, default=1.0, help="Risk-weighted score penalty weight on late abort rate")
    parser.add_argument("--scenario-set", type=str, default="core", choices=["core", "extended"], help="core = conference 6 (default, baseline-preserving); extended = +S7/S9/S10 (journal). Ignored if --scenarios given.")
    parser.add_argument("--scenarios", type=str, default=None, help="Comma-separated ScenarioType values to include (overrides --scenario-set)")
    parser.add_argument("--policies", type=str, default=None, help="Comma-separated Policy names to include")

    try:
        args = parser.parse_args(argv)
        paths = run_experiment(args)
        
        print("\nExperiment completed successfully.")
        print("Generated files:")
        for k, v in paths.items():
            print(f"  {k}: {v}")
            
        return 0
    except Exception as e:
        print(f"Error running experiment: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
