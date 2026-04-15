"""CCUS-Gym: Multi-disruption CCUS network coordination environment.

A PettingZoo-based multi-agent environment for studying coordinated
carbon capture, utilization, and storage under transport, supply, and
storage disruptions.

Architecture:
    - core/: physics, network, quality, and proxy-model utilities
    - sim/: environment, cases, scenarios, and disruption modelling
    - rl/: training specs and the minimal MAPPO baseline
    - cli/: command-line entry points for train/eval/batch experiments
"""

from ccus_gym.sim.env import CCUSEnv
from ccus_gym.core.physical import (
    PhysicalLayer,
    PhysicalOutcome,
    EmitterPhysics,
    PipelinePhysics,
    ShipPhysics,
    RailPhysics,
    TerminalBuffer,
    StoragePhysics,
    SHIP_TYPES,
    RAIL_DEFAULTS,
)
from ccus_gym.core.network import CCUSNetwork, Emitter, TransportMode, StorageSite
from ccus_gym.sim.disruptions import DisruptionGenerator, MechanismAxes
from ccus_gym.sim.configs import (
    MINIMAL_NETWORK_CONFIG,
    FULL_NETWORK_CONFIG,
    SCENARIO_CONFIGS,
    CALIBRATED_HUB_CONFIG,
    make_config,
)
from ccus_gym.sim.case_loader import load_case
from ccus_gym.core.quality import (
    COMPONENT_KEYS,
    CAPTURE_METHOD_LIBRARY,
    DEFAULT_STORAGE_QUALITY_LIMITS,
    blend_streams,
    compute_effective_stream,
    storage_quality_penalty,
)
from ccus_gym.core.storage_proxy import StorageProxyModel
from ccus_gym.rl.training import (
    DEFAULT_MAPPO_CONFIG,
    build_role_groups,
    describe_training_setup,
    make_env_and_training_spec,
)
from ccus_gym.rl.mappo import (
    RoleMAPPOPolicy,
    build_role_policies,
    evaluate_policies,
    load_checkpoint,
    plot_training_history,
    save_checkpoint,
    save_history_csv,
    save_history_jsonl,
    score_episode,
    train_mappo,
    write_tensorboard_history,
)
from ccus_gym.baselines.rule_based import (
    DEFAULT_RULE_BASED_CONFIG,
    EconomicRuleBasedController,
    evaluate_rule_based,
)
from ccus_gym.viz import (
    animate_episode_trace,
    rollout_episode_trace,
    save_episode_animation,
    save_episode_trace_json,
)

__version__ = "0.4.0"

__all__ = [
    "CCUSEnv",
    "PhysicalLayer",
    "PhysicalOutcome",
    "EmitterPhysics",
    "PipelinePhysics",
    "ShipPhysics",
    "RailPhysics",
    "TerminalBuffer",
    "StoragePhysics",
    "SHIP_TYPES",
    "RAIL_DEFAULTS",
    "CCUSNetwork",
    "Emitter",
    "TransportMode",
    "StorageSite",
    "DisruptionGenerator",
    "MechanismAxes",
    "MINIMAL_NETWORK_CONFIG",
    "FULL_NETWORK_CONFIG",
    "SCENARIO_CONFIGS",
    "CALIBRATED_HUB_CONFIG",
    "make_config",
    "load_case",
    "StorageProxyModel",
    "COMPONENT_KEYS",
    "CAPTURE_METHOD_LIBRARY",
    "DEFAULT_STORAGE_QUALITY_LIMITS",
    "blend_streams",
    "compute_effective_stream",
    "storage_quality_penalty",
    "DEFAULT_MAPPO_CONFIG",
    "build_role_groups",
    "describe_training_setup",
    "make_env_and_training_spec",
    "RoleMAPPOPolicy",
    "build_role_policies",
    "evaluate_policies",
    "load_checkpoint",
    "plot_training_history",
    "save_checkpoint",
    "save_history_csv",
    "save_history_jsonl",
    "score_episode",
    "train_mappo",
    "write_tensorboard_history",
    "DEFAULT_RULE_BASED_CONFIG",
    "EconomicRuleBasedController",
    "evaluate_rule_based",
    "animate_episode_trace",
    "rollout_episode_trace",
    "save_episode_animation",
    "save_episode_trace_json",
]
