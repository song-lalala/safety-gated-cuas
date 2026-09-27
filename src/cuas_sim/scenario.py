import enum
import random
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict

from .config import Config
from .agents import Target, Interceptor
from .types import State2D

class ScenarioType(enum.Enum):
    """Enumeration of supported scenario types."""
    S1_STRAIGHT_INTRUSION = "S1_STRAIGHT_INTRUSION"
    S2_EVASIVE_MANEUVER = "S2_EVASIVE_MANEUVER"
    S3_LOW_IDENTIFICATION_CONFIDENCE = "S3_LOW_IDENTIFICATION_CONFIDENCE"
    S4_RISK_ZONE_PROXIMITY = "S4_RISK_ZONE_PROXIMITY"
    S5_TRACKING_LOSS = "S5_TRACKING_LOSS"
    S6_NON_INTERCEPTABLE_TARGET = "S6_NON_INTERCEPTABLE_TARGET"
    # Journal Phase 4 (extended set; not in the core conference 6).
    S7_RISK_ZONE_TRANSIT = "S7_RISK_ZONE_TRANSIT"
    S9_FRIENDLY_PROXIMITY = "S9_FRIENDLY_PROXIMITY"
    S10_MANEUVER_RISK_ZONE = "S10_MANEUVER_RISK_ZONE"
    # Review response (W8, Reviewer 1.3): a grazing intrusion whose true
    # clearance straddles the keep-out margin, so the two risk models can
    # actually be told apart.
    S11_MARGINAL_GRAZING = "S11_MARGINAL_GRAZING"

@dataclass
class Scenario:
    """Container for a simulation scenario.

    Attributes
    ----------
    config: Config
        Simulation configuration.
    target: Target
        Target agent.
    interceptor: Interceptor
        Interceptor agent.
    scenario_type: ScenarioType
        Type of the scenario.
    step_hook: Optional[Callable[[int], None]]
        Optional hook called each simulation step with the current step index.
        It can modify agents (e.g., change target velocity).
    metadata: Dict[str, object]
        Arbitrary metadata for the scenario.
    """
    config: Config
    target: Target
    interceptor: Interceptor
    scenario_type: ScenarioType
    step_hook: Optional[Callable[[int], None]] = None
    metadata: Dict[str, object] = field(default_factory=dict)

def _perturb_state(base_state: State2D, rng: random.Random) -> State2D:
    """Apply small random perturbations to a base State2D.

    The perturbation range is [-0.1, 0.1] for position and [-0.05, 0.05] for velocity.
    """
    return State2D(
        x=base_state.x + rng.uniform(-0.1, 0.1),
        y=base_state.y + rng.uniform(-0.1, 0.1),
        vx=base_state.vx + rng.uniform(-0.05, 0.05),
        vy=base_state.vy + rng.uniform(-0.05, 0.05),
    )

def get_s1_scenario(seed: int | None = None) -> Scenario:
    cfg = Config(time_step=0.1, random_seed=seed)
    rng = random.Random(seed) if seed is not None else random.Random()
    target_base = State2D(x=-10.0, y=0.0, vx=1.0, vy=0.0)
    interceptor_base = State2D(x=0.0, y=0.0, vx=0.0, vy=0.0)
    target_state = _perturb_state(target_base, rng)
    interceptor_state = _perturb_state(interceptor_base, rng)
    target = Target(state=target_state, config=cfg)
    interceptor = Interceptor(state=interceptor_state, config=cfg)
    metadata = {"scenario_name": "S1 Straight Intrusion", "description": "Target moves straight towards interceptor."}
    return Scenario(config=cfg, target=target, interceptor=interceptor, scenario_type=ScenarioType.S1_STRAIGHT_INTRUSION, metadata=metadata)

def get_s2_scenario(seed: int | None = None) -> Scenario:
    cfg = Config(time_step=0.1, random_seed=seed)
    rng = random.Random(seed) if seed is not None else random.Random()
    target_base = State2D(x=-10.0, y=0.0, vx=1.0, vy=0.0)
    interceptor_base = State2D(x=0.0, y=0.0, vx=0.0, vy=0.0)
    target_state = _perturb_state(target_base, rng)
    interceptor_state = _perturb_state(interceptor_base, rng)
    target = Target(state=target_state, config=cfg)
    interceptor = Interceptor(state=interceptor_state, config=cfg)
    maneuver_step = rng.randint(5, 15)
    dvx = -2.0 * target.state.vx + rng.uniform(-0.1, 0.1)
    dvy = -2.0 * target.state.vy + rng.uniform(-0.1, 0.1)
    def hook(step: int) -> None:
        if step == maneuver_step:
            target.state.vx += dvx
            target.state.vy += dvy
    metadata = {"scenario_name": "S2 Evasive Maneuver", "description": "Target performs an evasive maneuver at a specific step.", "maneuver_step": maneuver_step, "dvx": dvx, "dvy": dvy}
    return Scenario(config=cfg, target=target, interceptor=interceptor, scenario_type=ScenarioType.S2_EVASIVE_MANEUVER, step_hook=hook, metadata=metadata)
def get_s3_scenario(seed: int | None = None) -> Scenario:
    cfg = Config(time_step=0.1, random_seed=seed)
    rng = random.Random(seed) if seed is not None else random.Random()
    # Simple target moving towards a protected zone (e.g., positive x direction)
    target_base = State2D(x=-10.0, y=0.0, vx=1.0, vy=0.0)
    interceptor_base = State2D(x=0.0, y=0.0, vx=0.0, vy=0.0)
    target_state = _perturb_state(target_base, rng)
    interceptor_state = _perturb_state(interceptor_base, rng)
    target = Target(state=target_state, config=cfg)
    interceptor = Interceptor(state=interceptor_state, config=cfg)
    # Identification confidence in low range 0.2‑0.5
    identification_confidence = rng.uniform(0.2, 0.5)
    metadata = {
        "scenario_name": "S3 Low Identification Confidence",
        "description": "Target is detected with low identification confidence.",
        "identification_confidence": identification_confidence,
        "confidence_range": [0.2, 0.5],
    }
    return Scenario(
        config=cfg,
        target=target,
        interceptor=interceptor,
        scenario_type=ScenarioType.S3_LOW_IDENTIFICATION_CONFIDENCE,
        metadata=metadata,
    )

def get_s4_scenario(seed: int | None = None) -> Scenario:
    cfg = Config(time_step=0.1, random_seed=seed)
    rng = random.Random(seed) if seed is not None else random.Random()
    # Target moves horizontally near risk zone
    target_base = State2D(x=-10.0, y=5.0, vx=1.0, vy=0.0)
    interceptor_base = State2D(x=0.0, y=0.0, vx=0.0, vy=0.0)
    target_state = _perturb_state(target_base, rng)
    interceptor_state = _perturb_state(interceptor_base, rng)
    target = Target(state=target_state, config=cfg)
    interceptor = Interceptor(state=interceptor_state, config=cfg)
    # Risk zone parameters
    risk_zone_radius = 3.0
    # clearance between 0.1 and 0.5
    risk_clearance = rng.uniform(0.1, 0.5)
    
    proximity_step = 20
    risk_zone_center_x = target_state.x + target_state.vx * cfg.time_step * proximity_step
    risk_zone_center_y = target_state.y + risk_zone_radius + risk_clearance
    risk_zone_center = (risk_zone_center_x, risk_zone_center_y)
    expected_min_distance_to_risk_zone = risk_zone_radius + risk_clearance
    metadata = {
        "scenario_name": "S4 Risk Zone Proximity",
        "description": "Target passes near a defined risk zone.",
        "risk_zone_center": risk_zone_center,
        "risk_zone_radius": risk_zone_radius,
        "risk_proximity_level": "high",
        "expected_min_distance_to_risk_zone": expected_min_distance_to_risk_zone,
        "risk_clearance": risk_clearance,
        "proximity_step": proximity_step,
    }
    return Scenario(
        config=cfg,
        target=target,
        interceptor=interceptor,
        scenario_type=ScenarioType.S4_RISK_ZONE_PROXIMITY,
        metadata=metadata,
    )

def get_s5_scenario(seed: int | None = None) -> Scenario:
    cfg = Config(time_step=0.1, random_seed=seed)
    rng = random.Random(seed) if seed is not None else random.Random()
    
    # Target moves in a straight line
    target_base = State2D(x=-15.0, y=0.0, vx=1.0, vy=0.0)
    interceptor_base = State2D(x=0.0, y=0.0, vx=0.0, vy=0.0)
    
    target_state = _perturb_state(target_base, rng)
    interceptor_state = _perturb_state(interceptor_base, rng)
    
    target = Target(state=target_state, config=cfg)
    interceptor = Interceptor(state=interceptor_state, config=cfg)
    
    # Tracking loss metadata
    tracking_loss_start_step = rng.randint(10, 20)
    tracking_loss_duration_steps = rng.randint(5, 15)
    tracking_loss_end_step = tracking_loss_start_step + tracking_loss_duration_steps
    nominal_tracking_confidence = rng.uniform(0.7, 1.0)
    lost_tracking_confidence = rng.uniform(0.0, 0.3)
    
    metadata = {
        "scenario_name": "S5 Tracking Loss",
        "description": "Target tracking is temporarily lost due to observation dropout.",
        "tracking_loss_start_step": tracking_loss_start_step,
        "tracking_loss_duration_steps": tracking_loss_duration_steps,
        "tracking_loss_end_step": tracking_loss_end_step,
        "nominal_tracking_confidence": nominal_tracking_confidence,
        "lost_tracking_confidence": lost_tracking_confidence,
        "tracking_loss_reason": "simulated_observation_dropout",
    }
    
    return Scenario(
        config=cfg,
        target=target,
        interceptor=interceptor,
        scenario_type=ScenarioType.S5_TRACKING_LOSS,
        metadata=metadata,
    )

def get_s6_scenario(seed: int | None = None) -> Scenario:
    import math
    cfg = Config(time_step=0.1, random_seed=seed)
    rng = random.Random(seed) if seed is not None else random.Random()
    
    # Target is far away and moving fast
    target_base = State2D(x=-50.0, y=50.0, vx=10.0, vy=-5.0)
    interceptor_base = State2D(x=0.0, y=0.0, vx=0.0, vy=0.0)
    
    target_state = _perturb_state(target_base, rng)
    interceptor_state = _perturb_state(interceptor_base, rng)
    
    target = Target(state=target_state, config=cfg)
    interceptor = Interceptor(state=interceptor_state, config=cfg)
    
    target_speed = math.hypot(target_state.vx, target_state.vy)
    interceptor_nominal_max_speed = rng.uniform(2.0, 5.0)
    speed_ratio = interceptor_nominal_max_speed / target_speed
    
    initial_distance = math.hypot(target_state.x - interceptor_state.x, target_state.y - interceptor_state.y)
    engagement_horizon_steps = rng.randint(100, 200)
    
    metadata = {
        "scenario_name": "S6 Non-Interceptable Target",
        "description": "Target is too fast or far to be intercepted within the engagement horizon.",
        "target_speed": target_speed,
        "interceptor_nominal_max_speed": interceptor_nominal_max_speed,
        "speed_ratio": speed_ratio,
        "initial_distance": initial_distance,
        "engagement_horizon_steps": engagement_horizon_steps,
        "interceptability_label": "non_interceptable",
        "non_interceptable_reason": "target_speed_exceeds_interceptor_capability",
    }
    
    return Scenario(
        config=cfg,
        target=target,
        interceptor=interceptor,
        scenario_type=ScenarioType.S6_NON_INTERCEPTABLE_TARGET,
        metadata=metadata,
    )

def get_s7_scenario(seed: int | None = None) -> Scenario:
    """S7 — Risk-Zone Transit. Target passes THROUGH a risk zone (centred on its
    path), unlike S4 which only grazes. Exercises R_env intrusion probability
    against a genuine intrusion. Long prediction horizon so the transit is seen."""
    cfg = Config(time_step=0.1, random_seed=seed)
    rng = random.Random(seed) if seed is not None else random.Random()
    target_state = _perturb_state(State2D(x=-10.0, y=0.0, vx=1.0, vy=0.0), rng)
    interceptor_state = _perturb_state(State2D(x=0.0, y=0.0, vx=0.0, vy=0.0), rng)
    target = Target(state=target_state, config=cfg)
    interceptor = Interceptor(state=interceptor_state, config=cfg)
    risk_zone_radius = 2.0
    # Zone centred on the target's straight-line path, ~2.5 units ahead.
    zone_x = target_state.x + target_state.vx * cfg.time_step * 25
    zone_y = target_state.y + target_state.vy * cfg.time_step * 25
    metadata = {
        "scenario_name": "S7 Risk Zone Transit",
        "description": "Target passes through (not just near) a risk zone.",
        "risk_zone_center": (zone_x, zone_y),
        "risk_zone_radius": risk_zone_radius,
        "prediction_horizon_steps": 30,
    }
    return Scenario(config=cfg, target=target, interceptor=interceptor,
                    scenario_type=ScenarioType.S7_RISK_ZONE_TRANSIT, metadata=metadata)


def get_s9_scenario(seed: int | None = None) -> Scenario:
    """S9 — Friendly/Civilian Proximity. A protected entity (risk zone) lies just
    off the target's path AND identification is ambiguous (C_id straddles θ_id).
    Engaging here is both potentially unsafe (collateral) and a misidentification
    risk — a combined-evidence test that crisp independent thresholds can miss."""
    cfg = Config(time_step=0.1, random_seed=seed)
    rng = random.Random(seed) if seed is not None else random.Random()
    target_state = _perturb_state(State2D(x=-10.0, y=0.0, vx=1.0, vy=0.0), rng)
    interceptor_state = _perturb_state(State2D(x=0.0, y=0.0, vx=0.0, vy=0.0), rng)
    target = Target(state=target_state, config=cfg)
    interceptor = Interceptor(state=interceptor_state, config=cfg)
    risk_zone_radius = 1.5
    zone_x = target_state.x + target_state.vx * cfg.time_step * 25
    # graze just outside the friendly's bubble (borderline environmental risk)
    zone_y = target_state.y + (risk_zone_radius + rng.uniform(0.1, 0.5))
    identification_confidence = rng.uniform(0.4, 0.65)  # ambiguous; straddles θ_id=0.6
    metadata = {
        "scenario_name": "S9 Friendly Proximity",
        "description": "Protected friendly/civilian near the target's path under identification ambiguity.",
        "risk_zone_center": (zone_x, zone_y),
        "risk_zone_radius": risk_zone_radius,
        "identification_confidence": identification_confidence,
        "misidentification_risk": True,
        "prediction_horizon_steps": 30,
    }
    return Scenario(config=cfg, target=target, interceptor=interceptor,
                    scenario_type=ScenarioType.S9_FRIENDLY_PROXIMITY, metadata=metadata)


def get_s10_scenario(seed: int | None = None) -> Scenario:
    """S10 — Maneuver into Risk Zone. Target flies straight then maneuvers (like
    S2) toward a risk zone mid-flight. Exercises Kalman tracking (NIS spike at the
    maneuver) AND R_env (predicting the post-maneuver path into the zone) together."""
    cfg = Config(time_step=0.1, random_seed=seed)
    rng = random.Random(seed) if seed is not None else random.Random()
    target_state = _perturb_state(State2D(x=-10.0, y=0.0, vx=1.0, vy=0.0), rng)
    interceptor_state = _perturb_state(State2D(x=0.0, y=0.0, vx=0.0, vy=0.0), rng)
    target = Target(state=target_state, config=cfg)
    interceptor = Interceptor(state=interceptor_state, config=cfg)
    risk_zone_radius = 2.0
    zone_x = target_state.x + target_state.vx * cfg.time_step * 25
    zone_y = target_state.y - 2.0  # zone below initial path (reachable in-trial)
    maneuver_step = rng.randint(3, 7)
    # steep downward turn so the TRUE target reaches the zone within the trial
    # (≈0.2/step descent → ~10 steps to the zone), not just in prediction
    dvy = -2.0 + rng.uniform(-0.1, 0.1)

    def hook(step: int) -> None:
        if step == maneuver_step:
            target.state.vy += dvy

    metadata = {
        "scenario_name": "S10 Maneuver into Risk Zone",
        "description": "Target maneuvers toward a risk zone mid-flight.",
        "risk_zone_center": (zone_x, zone_y),
        "risk_zone_radius": risk_zone_radius,
        "maneuver_step": maneuver_step,
        "dvy": dvy,
        "prediction_horizon_steps": 30,
    }
    return Scenario(config=cfg, target=target, interceptor=interceptor,
                    scenario_type=ScenarioType.S10_MANEUVER_RISK_ZONE, step_hook=hook, metadata=metadata)


def get_s11_scenario(seed: int | None = None) -> Scenario:
    """S11 - Marginal Grazing. The risk zone sits beside the target's path at a
    clearance drawn so that it straddles the keep-out margin b = 0.6.

    S4 and S9 always clear the zone and S7 always transits it, so on the
    submitted benchmark the linear and the probabilistic R_env saturate to the
    same answer and cannot be compared (Reviewer 1.3, and Section V-F of the
    manuscript, which deferred exactly this scenario). Here roughly half the
    trials are genuinely unsafe by the ground-truth criterion and half are not,
    and the true clearance is the same order as the prediction uncertainty --
    the regime where a probability and a linear falloff should disagree.
    """
    cfg = Config(time_step=0.1, random_seed=seed)
    rng = random.Random(seed) if seed is not None else random.Random()
    target_state = _perturb_state(State2D(x=-10.0, y=0.0, vx=1.0, vy=0.0), rng)
    interceptor_state = _perturb_state(State2D(x=0.0, y=0.0, vx=0.0, vy=0.0), rng)
    target = Target(state=target_state, config=cfg)
    interceptor = Interceptor(state=interceptor_state, config=cfg)
    risk_zone_radius = 2.0
    zone_x = target_state.x + target_state.vx * cfg.time_step * 25
    # Clearance rho* = |p_T - c| - R_zone drawn over [-0.2, 1.4]; the evaluation
    # calls a contact unsafe when rho* <= b = 0.6, so the label splits ~50/50
    # instead of being decided by construction.
    clearance = rng.uniform(-0.2, 1.4)
    zone_y = target_state.y + (risk_zone_radius + clearance)
    metadata = {
        "scenario_name": "S11 Marginal Grazing",
        "description": "Zone clearance straddles the keep-out margin; separates the risk models.",
        "risk_zone_center": (zone_x, zone_y),
        "risk_zone_radius": risk_zone_radius,
        "nominal_clearance": clearance,
        "prediction_horizon_steps": 30,
    }
    return Scenario(config=cfg, target=target, interceptor=interceptor,
                    scenario_type=ScenarioType.S11_MARGINAL_GRAZING, metadata=metadata)


SCENARIO_FACTORIES: Dict[ScenarioType, Callable[..., Scenario]] = {
    ScenarioType.S1_STRAIGHT_INTRUSION: get_s1_scenario,
    ScenarioType.S2_EVASIVE_MANEUVER: get_s2_scenario,
    ScenarioType.S3_LOW_IDENTIFICATION_CONFIDENCE: get_s3_scenario,
    ScenarioType.S4_RISK_ZONE_PROXIMITY: get_s4_scenario,
    ScenarioType.S5_TRACKING_LOSS: get_s5_scenario,
    ScenarioType.S6_NON_INTERCEPTABLE_TARGET: get_s6_scenario,
    ScenarioType.S7_RISK_ZONE_TRANSIT: get_s7_scenario,
    ScenarioType.S9_FRIENDLY_PROXIMITY: get_s9_scenario,
    ScenarioType.S10_MANEUVER_RISK_ZONE: get_s10_scenario,
    ScenarioType.S11_MARGINAL_GRAZING: get_s11_scenario,
}

def get_all_scenario_types() -> list[ScenarioType]:
    """The CORE conference set (6). Kept unchanged so default runs and the
    baseline byte-identical guarantee are preserved. Journal/extended scenarios
    are in get_extended_scenario_types()."""
    return [
        ScenarioType.S1_STRAIGHT_INTRUSION,
        ScenarioType.S2_EVASIVE_MANEUVER,
        ScenarioType.S3_LOW_IDENTIFICATION_CONFIDENCE,
        ScenarioType.S4_RISK_ZONE_PROXIMITY,
        ScenarioType.S5_TRACKING_LOSS,
        ScenarioType.S6_NON_INTERCEPTABLE_TARGET,
    ]

def get_extended_scenario_types() -> list[ScenarioType]:
    """Journal Phase 4 extended set: core 6 + S7 transit, S9 friendly proximity,
    S10 maneuver-into-zone. Exercises R_env intrusion, combined evidence, and
    tracking+risk coupling that the benign core set under-demonstrates."""
    return get_all_scenario_types() + [
        ScenarioType.S7_RISK_ZONE_TRANSIT,
        ScenarioType.S9_FRIENDLY_PROXIMITY,
        ScenarioType.S10_MANEUVER_RISK_ZONE,
    ]

def get_scenario_factory(scenario_type: ScenarioType) -> Callable[..., Scenario]:
    return SCENARIO_FACTORIES[scenario_type]
