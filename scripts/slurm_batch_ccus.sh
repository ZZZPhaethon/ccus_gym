#!/bin/bash
#SBATCH --job-name=ccus_batch
#SBATCH --partition=tide
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch_tide/yc5224/logs/batch_%j.out
#SBATCH --error=/scratch_tide/yc5224/logs/batch_%j.err

# Multi-seed batch run — trains the same config with multiple seeds and
# aggregates results. Use this for statistically robust comparisons.

# ── configurable parameters ────────────────────────────────────────────────
BASE=minimal
SCENARIO=T
SEVERITY=0.5
EPISODES=2000
SEEDS="42,43,44,45,45"    # comma-separated
# ──────────────────────────────────────────────────────────────────────────

ENV=/scratch_tide/yc5224/miniconda3/envs/ccus_gym
CODE=/scratch_tide/yc5224/CYW_code
OUT=/scratch_tide/yc5224/runs/batch_${BASE}_${SCENARIO}_${SLURM_JOB_ID}

mkdir -p /scratch_tide/yc5224/logs
mkdir -p "$OUT"

echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $(hostname)"
echo "GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Output  : $OUT"
echo "Start   : $(date)"

export PYTHONPATH="$CODE:$PYTHONPATH"

$ENV/bin/python -m ccus_gym.cli.batch_mappo \
    --base       "$BASE"     \
    --scenario   "$SCENARIO" \
    --severity   "$SEVERITY" \
    --episodes   "$EPISODES" \
    --seeds      "$SEEDS"    \
    --device     cuda        \
    --out-dir    "$OUT"

echo "Done : $(date)"
