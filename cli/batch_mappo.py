"""Run MAPPO experiments over multiple random seeds and aggregate results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np


def _parse_seeds(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_aggregate_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [
        "seed",
        "best_metric_value",
        "best_metric_episode",
        "eval_mean_total_stored",
        "eval_mean_total_vented",
        "eval_mean_total_captured",
        "eval_mean_pressure_violations",
        "eval_mean_quality_violations",
        "eval_mean_transport_cost",
        "eval_mean_capture_cost",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed MAPPO experiments.")
    parser.add_argument("--base", default="minimal", choices=["minimal", "full", "calibrated"])
    parser.add_argument("--scenario", default="T", choices=["T", "S", "G", "TS", "TG", "SG", "TSG"])
    parser.add_argument("--severity", type=float, default=0.5)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--best-metric", default="score", choices=["score", "total_stored", "total_vented"])
    parser.add_argument("--output-dir", default="runs/mappo_batch")
    args = parser.parse_args()

    package_parent = str(Path(__file__).resolve().parents[2])
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)

    import ccus_gym
    from ccus_gym.rl.mappo import (
        evaluate_policies,
        load_checkpoint,
        plot_training_history,
        save_history_csv,
        save_history_jsonl,
        train_mappo,
        write_tensorboard_history,
    )

    seeds = _parse_seeds(args.seeds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_rows: List[Dict[str, Any]] = []

    for seed in seeds:
        run_dir = output_dir / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        config = ccus_gym.make_config(
            base=args.base,
            scenario_family=args.scenario,
            severity=args.severity,
            seed=seed,
        )
        env = ccus_gym.CCUSEnv(config)

        best_path = run_dir / "best.pt"
        latest_path = run_dir / "latest.pt"
        result = train_mappo(
            env,
            train_config={"best_metric": args.best_metric},
            episodes=args.episodes,
            seed=seed,
            device=args.device,
            best_checkpoint_path=str(best_path),
            latest_checkpoint_path=str(latest_path),
        )

        save_history_jsonl(str(run_dir / "history.jsonl"), result["history"])
        save_history_csv(str(run_dir / "history.csv"), result["history"])
        plot_training_history(str(run_dir / "training.png"), result["history"])
        write_tensorboard_history(str(run_dir / "tb"), result["history"])

        policies, checkpoint_meta = load_checkpoint(
            str(best_path),
            env,
            device=args.device,
            load_optimizer=False,
        )
        eval_result = evaluate_policies(
            env,
            policies,
            episodes=args.eval_episodes,
            seed=seed,
            deterministic=True,
        )
        eval_result["checkpoint_metadata"] = checkpoint_meta
        _write_json(run_dir / "eval.json", eval_result)

        aggregate_rows.append(
            {
                "seed": seed,
                "best_metric_value": result["best_metric"]["value"],
                "best_metric_episode": result["best_metric"]["episode"],
                "eval_mean_total_stored": eval_result["summary"]["mean_total_stored"],
                "eval_mean_total_vented": eval_result["summary"]["mean_total_vented"],
                "eval_mean_total_captured": eval_result["summary"]["mean_total_captured"],
                "eval_mean_pressure_violations": eval_result["summary"]["mean_pressure_violations"],
                "eval_mean_quality_violations": eval_result["summary"]["mean_quality_violations"],
                "eval_mean_transport_cost": eval_result["summary"]["mean_transport_cost"],
                "eval_mean_capture_cost": eval_result["summary"]["mean_capture_cost"],
            }
        )

    _write_aggregate_csv(output_dir / "aggregate.csv", aggregate_rows)

    summary = {
        "config": {
            "base": args.base,
            "scenario": args.scenario,
            "severity": args.severity,
            "episodes": args.episodes,
            "eval_episodes": args.eval_episodes,
            "best_metric": args.best_metric,
            "seeds": seeds,
        },
        "aggregate": {
            "mean_eval_total_stored": float(np.mean([r["eval_mean_total_stored"] for r in aggregate_rows])),
            "std_eval_total_stored": float(np.std([r["eval_mean_total_stored"] for r in aggregate_rows])),
            "mean_eval_total_vented": float(np.mean([r["eval_mean_total_vented"] for r in aggregate_rows])),
            "std_eval_total_vented": float(np.std([r["eval_mean_total_vented"] for r in aggregate_rows])),
            "mean_eval_quality_violations": float(np.mean([r["eval_mean_quality_violations"] for r in aggregate_rows])),
            "std_eval_quality_violations": float(np.std([r["eval_mean_quality_violations"] for r in aggregate_rows])),
        },
        "runs": aggregate_rows,
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
