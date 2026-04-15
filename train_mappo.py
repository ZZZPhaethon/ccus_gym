"""Command-line entry point for the minimal MAPPO baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a minimal MAPPO baseline on CCUS-Gym.")
    parser.add_argument("--base", default="minimal", choices=["minimal", "full", "calibrated"])
    parser.add_argument("--scenario", default="T", choices=["T", "S", "G", "TS", "TG", "SG", "TSG"])
    parser.add_argument("--severity", type=float, default=0.5)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save", default="", help="Optional checkpoint output path.")
    parser.add_argument("--history", default="", help="Optional JSONL path for training history.")
    parser.add_argument("--history-csv", default="", help="Optional CSV path for training history.")
    parser.add_argument("--tensorboard-dir", default="", help="Optional TensorBoard log directory.")
    parser.add_argument("--plot", default="", help="Optional PNG path for training curves.")
    parser.add_argument("--best-save", default="", help="Optional best-checkpoint output path.")
    parser.add_argument("--latest-save", default="", help="Optional latest-checkpoint output path.")
    parser.add_argument("--resume", default="", help="Optional checkpoint to resume from.")
    parser.add_argument(
        "--best-metric",
        default="score",
        choices=["score", "total_stored", "total_vented"],
        help="Metric used to track the best checkpoint.",
    )
    args = parser.parse_args()

    repo_parent = os.path.dirname(os.path.abspath(os.getcwd()))
    if repo_parent not in sys.path:
        sys.path.insert(0, repo_parent)

    import ccus_gym
    from ccus_gym.mappo import (
        load_checkpoint,
        plot_training_history,
        save_checkpoint,
        save_history_csv,
        save_history_jsonl,
        train_mappo,
        write_tensorboard_history,
    )

    config = ccus_gym.make_config(
        base=args.base,
        scenario_family=args.scenario,
        severity=args.severity,
        seed=args.seed,
    )
    env = ccus_gym.CCUSEnv(config)
    policies = None
    resumed_metadata = {}
    if args.resume:
        policies, resumed_metadata = load_checkpoint(args.resume, env, device=args.device)
    result = train_mappo(
        env,
        train_config={"best_metric": args.best_metric},
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
        policies=policies,
        best_checkpoint_path=args.best_save,
        latest_checkpoint_path=args.latest_save,
    )
    if resumed_metadata:
        result["resumed_from"] = resumed_metadata

    summary = {
        "episodes": args.episodes,
        "last_episode": result["history"][-1] if result["history"] else {},
        "best_metric": result.get("best_metric", {}),
        "best_episode": result.get("best_episode", {}),
    }
    print(json.dumps(summary, indent=2))

    if args.history:
        save_history_jsonl(args.history, result["history"])
        print(f"history_saved={args.history}")

    if args.history_csv:
        save_history_csv(args.history_csv, result["history"])
        print(f"history_csv_saved={args.history_csv}")

    if args.tensorboard_dir:
        write_tensorboard_history(args.tensorboard_dir, result["history"])
        print(f"tensorboard_saved={args.tensorboard_dir}")

    if args.plot:
        plot_training_history(args.plot, result["history"])
        print(f"plot_saved={args.plot}")

    if args.best_save:
        print(f"best_checkpoint_saved={args.best_save}")
    if args.latest_save:
        print(f"latest_checkpoint_saved={args.latest_save}")

    if args.save:
        save_checkpoint(
            args.save,
            result["policies"],
            metadata={
                "base": args.base,
                "scenario": args.scenario,
                "severity": args.severity,
                "episodes": args.episodes,
                "seed": args.seed,
                "best_metric": args.best_metric,
            },
        )
        print(f"checkpoint_saved={args.save}")


if __name__ == "__main__":
    main()
