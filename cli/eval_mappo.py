"""Evaluate a trained MAPPO checkpoint on CCUS-Gym."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a MAPPO checkpoint on CCUS-Gym.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base", default="minimal", choices=["minimal", "full", "calibrated"])
    parser.add_argument("--scenario", default="T", choices=["T", "S", "G", "TS", "TG", "SG", "TSG"])
    parser.add_argument("--severity", type=float, default=0.5)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    package_parent = str(Path(__file__).resolve().parents[2])
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)

    import ccus_gym
    from ccus_gym.rl.mappo import evaluate_policies, load_checkpoint

    config = ccus_gym.make_config(
        base=args.base,
        scenario_family=args.scenario,
        severity=args.severity,
        seed=args.seed,
    )
    env = ccus_gym.CCUSEnv(config)
    policies, metadata = load_checkpoint(
        args.checkpoint,
        env,
        device=args.device,
        load_optimizer=False,
    )
    result = evaluate_policies(
        env,
        policies,
        episodes=args.episodes,
        seed=args.seed,
        deterministic=True,
    )
    result["checkpoint_metadata"] = metadata
    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"evaluation_saved={args.output}")


if __name__ == "__main__":
    main()
