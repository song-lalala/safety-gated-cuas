"""C-UAS Simulation Package.

Expose primary classes for external use.
"""

from .config import Config
from .types import State2D
from .agents import Agent, Target, Interceptor
from .simulator import Simulator
from .observation import TargetObservation, ObservationModel
from .estimation import TargetEstimate, TargetStateEstimator
from .prediction import PredictedState, TrajectoryPrediction, ConstantVelocityPredictor
from .evaluators import EnvironmentalRiskAssessment, EnvironmentalRiskEvaluator, InterceptabilityAssessment, InterceptabilityEvaluator
from .policies import DecisionAction, DecisionContext, build_decision_context, AlwaysEngagePolicy, DistanceOnlyPolicy, InterceptabilityOnlyPolicy, SafetyGatePolicy
from .metrics import DecisionRecord, MetricsSummary, MetricsLogger
from .runner import MonteCarloConfig, MonteCarloResult, MonteCarloRunner, get_default_policy_factories
from .results import write_dict_rows_csv, write_json, export_monte_carlo_result
from .scenario import (
    Scenario,
    ScenarioType,
    get_s1_scenario,
    get_s2_scenario,
    get_s3_scenario,
    get_s4_scenario,
    get_s5_scenario,
    get_s6_scenario,
    SCENARIO_FACTORIES,
    get_all_scenario_types,
    get_scenario_factory,
)

__all__ = [
    "Config",
    "State2D",
    "Agent",
    "Target",
    "Interceptor",
    "Simulator",
    "Scenario",
    "ScenarioType",
    "get_s1_scenario",
    "get_s2_scenario",
    "get_s3_scenario",
    "get_s4_scenario",
    "get_s5_scenario",
    "get_s6_scenario",
    "SCENARIO_FACTORIES",
    "get_all_scenario_types",
    "get_scenario_factory",
    "TargetObservation",
    "ObservationModel",
    "TargetEstimate",
    "TargetStateEstimator",
    "PredictedState",
    "TrajectoryPrediction",
    "ConstantVelocityPredictor",
    "EnvironmentalRiskAssessment",
    "EnvironmentalRiskEvaluator",
    "InterceptabilityAssessment",
    "InterceptabilityEvaluator",
    "DecisionAction",
    "DecisionContext",
    "build_decision_context",
    "AlwaysEngagePolicy",
    "DistanceOnlyPolicy",
    "InterceptabilityOnlyPolicy",
    "SafetyGatePolicy",
    "DecisionRecord",
    "MetricsSummary",
    "MetricsLogger",
    "MonteCarloConfig",
    "MonteCarloResult",
    "MonteCarloRunner",
    "get_default_policy_factories",
    "write_dict_rows_csv",
    "write_json",
    "export_monte_carlo_result",
]
