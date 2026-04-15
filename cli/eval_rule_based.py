"""Evaluate the economics-aware rule-based baseline on CCUS-Gym."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the economic rule-based baseline on CCUS-Gym.")
    parser.add_argument("--base", default="minimal", choices=["minimal", "full", "calibrated"])
    parser.add_argument("--scenario", default="T", choices=["T", "S", "G", "TS", "TG", "SG", "TSG"])
    parser.add_argument("--severity", type=float, default=0.5)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    package_parent = str(Path(__file__).resolve().parents[2])
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)

    import ccus_gym
    from ccus_gym.baselines.rule_based import EconomicRuleBasedController, evaluate_rule_based

    config = ccus_gym.make_config(
        base=args.base,
        scenario_family=args.scenario,
        severity=args.severity,
        seed=args.seed,
    )
    env = ccus_gym.CCUSEnv(config)
    controller = EconomicRuleBasedController()
    result = evaluate_rule_based(
        env,
        controller,
        episodes=args.episodes,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"evaluation_saved={args.output}")


if __name__ == "__main__":
    main()
