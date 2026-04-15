"""CCUS-Gym: Multi-disruption CCUS network coordination environment.

A PettingZoo-based multi-agent environment for studying coordinated
carbon capture, utilization, and storage under transport, supply, and
storage disruptions.

Architecture:
    - Physical Layer (physical.py): Simulates CCUS network physics
    - Decision Layer (env.py): PettingZoo ParallelEnv wrapping the physical layer
    - Case Loader (case_loader.py): YAML case definition system
    - Storage Proxy (storage_proxy.py): ML proxy model wrapper for storage sites
"""

from ccus_gym.env import CCUSEnv
from ccus_gym.physical import (
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
from ccus_gym.network import CCUSNetwork, Emitter, TransportMode, StorageSite
from ccus_gym.disruptions import DisruptionGenerator, MechanismAxes
from ccus_gym.configs import (
    MINIMAL_NETWORK_CONFIG,
    FULL_NETWORK_CONFIG,
    SCENARIO_CONFIGS,
    CALIBRATED_HUB_CONFIG,
    make_config,
)
from ccus_gym.case_loader import load_case
from ccus_gym.quality import (
    COMPONENT_KEYS,
    CAPTURE_METHOD_LIBRARY,
    DEFAULT_STORAGE_QUALITY_LIMITS,
    blend_streams,
    compute_effective_stream,
    storage_quality_penalty,
)
from ccus_gym.storage_proxy import StorageProxyModel
from ccus_gym.training import (
    DEFAULT_MAPPO_CONFIG,
    build_role_groups,
    describe_training_setup,
    make_env_and_training_spec,
)
from ccus_gym.mappo import (
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
]
