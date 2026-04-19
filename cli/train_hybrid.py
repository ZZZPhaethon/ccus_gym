"""Train hybrid LLM-emitter + MAPPO transport/storage on CCUS-Gym.

Usage examples
--------------
# Local Qwen3 on GPU node (recommended — run via SLURM, see scripts/run_hybrid_slurm.sh)
python -m ccus_gym.cli.train_hybrid \\
    --llm-backend local --llm-model Qwen/Qwen3-7B \\
    --base minimal --scenario T --episodes 20

# Local with 4-bit quantization (less VRAM, e.g. Qwen3-14B on one GPU)
python -m ccus_gym.cli.train_hybrid \\
    --llm-backend local --llm-model Qwen/Qwen3-14B --load-in-4bit \\
    --base minimal --scenario TS --episodes 20

# Remote API: Qwen3 via DashScope
python -m ccus_gym.cli.train_hybrid \\
    --llm-backend api \\
    --llm-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \\
    --llm-model qwen-plus --llm-api-key YOUR_KEY \\
    --base minimal --scenario T --episodes 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid training: LLM controls emitters, MAPPO controls transport+storage."
    )
    # Environment
    parser.add_argument("--base", default="minimal", choices=["minimal", "full", "calibrated"])
    parser.add_argument("--scenario", default="T", choices=["T", "S", "G", "TS", "TG", "SG", "TSG"])
    parser.add_argument("--severity", type=float, default=0.5)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")

    # LLM backend selection
    parser.add_argument(
        "--llm-backend",
        default="local",
        choices=["local", "api"],
        help="'local' loads model via HuggingFace transformers (GPU node). "
             "'api' calls any OpenAI-compatible HTTP endpoint.",
    )
    parser.add_argument(
        "--llm-model",
        default="Qwen/Qwen3-7B",
        help="HuggingFace model ID (local) or model name passed to API.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="[local only] 4-bit quantization via bitsandbytes (saves VRAM).",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="[local only] 8-bit quantization via bitsandbytes.",
    )
    # API-only settings
    parser.add_argument(
        "--llm-base-url",
        default="http://localhost:11434/v1",
        help="[api only] Base URL of OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--llm-api-key",
        default="none",
        help="[api only] API key ('none' for local services without auth).",
    )
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=60,
        help="[api only] Request timeout in seconds.",
    )
    # Shared settings
    parser.add_argument(
        "--llm-call-interval",
        type=int,
        default=12,
        help="Months between LLM calls per emitter (default: 12 = once per year).",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.3,
        help="Sampling temperature.",
    )

    # Output paths (same schema as train_mappo.py for easy comparison)
    parser.add_argument("--best-save", default="", help="Best MAPPO checkpoint path.")
    parser.add_argument("--latest-save", default="", help="Latest MAPPO checkpoint path.")
    parser.add_argument("--history", default="", help="JSONL history output path.")
    parser.add_argument("--history-csv", default="", help="CSV history output path.")
    parser.add_argument("--tensorboard-dir", default="", help="TensorBoard log directory.")
    parser.add_argument("--plot", default="", help="PNG training curves output path.")
    parser.add_argument("--llm-log", default="", help="JSON file to save LLM reasoning log.")
    parser.add_argument(
        "--best-metric",
        default="score",
        choices=["score", "total_stored", "total_vented"],
    )
    args = parser.parse_args()

    package_parent = str(Path(__file__).resolve().parents[2])
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)

    import ccus_gym
    from ccus_gym.rl.hybrid_runner import train_hybrid
    from ccus_gym.rl.mappo import (
        plot_training_history,
        save_history_csv,
        save_history_jsonl,
        write_tensorboard_history,
    )

    config = ccus_gym.make_config(
        base=args.base,
        scenario_family=args.scenario,
        severity=args.severity,
        seed=args.seed,
    )
    env = ccus_gym.CCUSEnv(config)

    # Build LLM policies for emitter agents
    if args.llm_backend == "local":
        from ccus_gym.llm.local_policy import LocalLLMEmitterPolicy
        from ccus_gym.rl.training import build_role_groups as _brg
        import numpy as np

        groups = _brg(env)
        llm_policies = {}
        for agent in groups.get("emitter", []):
            eid = env._agent_comp_id[agent]
            n_routes = len(env.physical_layer.get_routes_for_emitter(eid))
            act_dim = int(np.prod(env.action_space(agent).shape))
            llm_policies[agent] = LocalLLMEmitterPolicy(
                agent_id=agent,
                n_routes=n_routes,
                action_dim=act_dim,
                model_name=args.llm_model,
                call_interval=args.llm_call_interval,
                temperature=args.llm_temperature,
                load_in_4bit=args.load_in_4bit,
                load_in_8bit=args.load_in_8bit,
            )
        print(f"LLM backend  : local (HuggingFace transformers)")
    else:
        from ccus_gym.rl.hybrid_runner import build_llm_emitter_policies
        llm_policies = build_llm_emitter_policies(
            env,
            base_url=args.llm_base_url,
            model=args.llm_model,
            api_key=args.llm_api_key,
            call_interval=args.llm_call_interval,
            temperature=args.llm_temperature,
            timeout=args.llm_timeout,
        )
        print(f"LLM backend  : api ({args.llm_base_url})")

    print(f"LLM model    : {args.llm_model}")
    print(f"Call interval: every {args.llm_call_interval} months")
    print(f"Emitter agents controlled by LLM: {list(llm_policies.keys())}")
    print()

    result = train_hybrid(
        env,
        llm_policies,
        train_config={"best_metric": args.best_metric},
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
        best_checkpoint_path=args.best_save,
        latest_checkpoint_path=args.latest_save,
    )

    summary = {
        "episodes": args.episodes,
        "llm_model": args.llm_model,
        "llm_call_interval": args.llm_call_interval,
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

    if args.llm_log:
        Path(args.llm_log).parent.mkdir(parents=True, exist_ok=True)
        llm_log = {
            agent: p.call_log
            for agent, p in result["llm_policies"].items()
        }
        with open(args.llm_log, "w", encoding="utf-8") as f:
            json.dump(llm_log, f, indent=2, ensure_ascii=False)
        print(f"llm_log_saved={args.llm_log}")

    if args.best_save:
        print(f"best_checkpoint_saved={args.best_save}")
    if args.latest_save:
        print(f"latest_checkpoint_saved={args.latest_save}")


if __name__ == "__main__":
    main()
