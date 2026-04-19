#!/bin/bash
#SBATCH --job-name=ccus_mappo
#SBATCH --partition=tide
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --qos=intermediate
#SBATCH --output=/scratch_tide/yc5224/logs/train_%j.out
#SBATCH --error=/scratch_tide/yc5224/logs/train_%j.err

# ── configurable parameters ────────────────────────────────────────────────
BASE=minimal          # minimal | full | calibrated
SCENARIO=T            # T | S | G | TS | TG | SG | TSG
SEVERITY=0.5
EPISODES=2000
SEED=42
# ──────────────────────────────────────────────────────────────────────────

ENV=/scratch_tide/yc5224/miniconda3/envs/ccus_gym
CODE=/scratch_tide/yc5224/CYW_code
OUT=/scratch_tide/yc5224/runs/${BASE}_${SCENARIO}_s${SEED}_${SLURM_JOB_ID}

mkdir -p /scratch_tide/yc5224/logs
mkdir -p "$OUT"

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $(hostname)"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Output dir : $OUT"
echo "Config     : base=$BASE scenario=$SCENARIO severity=$SEVERITY episodes=$EPISODES seed=$SEED"
echo "Start      : $(date)"

export PYTHONPATH="$CODE:$PYTHONPATH"

$ENV/bin/python -m ccus_gym.cli.train_mappo \
    --base        "$BASE"              \
    --scenario    "$SCENARIO"          \
    --severity    "$SEVERITY"          \
    --episodes    "$EPISODES"          \
    --seed        "$SEED"              \
    --device      cuda                 \
    --best-save   "$OUT/best.pt"       \
    --latest-save "$OUT/latest.pt"     \
    --history     "$OUT/history.jsonl" \
    --history-csv "$OUT/history.csv"   \
    --plot        "$OUT/curves.png"    \
    --tensorboard-dir "$OUT/tb"

echo "Done : $(date)"
