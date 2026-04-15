"""Generate a dynamic episode replay for CCUS-Gym."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an animated CCUS episode replay.")
    parser.add_argument("--base", default="minimal", choices=["minimal", "full", "calibrated"])
    parser.add_argument("--scenario", default="T", choices=["T", "S", "G", "TS", "TG", "SG", "TSG"])
    parser.add_argument("--severity", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default="", help="Optional MAPPO checkpoint for policy replay.")
    parser.add_argument("--random-policy", action="store_true", help="Force random actions even if a checkpoint is given.")
    parser.add_argument("--steps", type=int, default=24, help="Maximum number of steps to animate.")
    parser.add_argument("--interval-ms", type=int, default=1200)
    parser.add_argument("--output", default="runs/episode_replay.html")
    parser.add_argument("--trace-json", default="", help="Optional JSON trace output.")
    args = parser.parse_args()

    package_parent = str(Path(__file__).resolve().parents[2])
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)

    import ccus_gym
    from ccus_gym.rl.mappo import load_checkpoint
    from ccus_gym.viz import rollout_episode_trace, save_episode_animation, save_episode_trace_json

    config = ccus_gym.make_config(
        base=args.base,
        scenario_family=args.scenario,
        severity=args.severity,
        seed=args.seed,
    )
    env = ccus_gym.CCUSEnv(config)

    policies = None
    replay_title = f"CCUS Replay | {args.base} | {args.scenario} | severity={args.severity}"
    if args.checkpoint and not args.random_policy:
        policies, metadata = load_checkpoint(
            args.checkpoint,
            env,
            device=args.device,
            load_optimizer=False,
        )
        replay_title += f" | checkpoint seed={metadata.get('seed', 'n/a')}"
    elif args.random_policy:
        replay_title += " | random policy"

    trace = rollout_episode_trace(
        env,
        policies=policies,
        seed=args.seed,
        deterministic=True,
        max_steps=args.steps,
    )
    save_episode_animation(
        args.output,
        env,
        trace,
        title=replay_title,
        interval_ms=args.interval_ms,
    )
    print(f"animation_saved={args.output}")

    if args.trace_json:
        save_episode_trace_json(args.trace_json, trace)
        print(f"trace_saved={args.trace_json}")


if __name__ == "__main__":
    main()
