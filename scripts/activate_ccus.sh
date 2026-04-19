#!/bin/bash
# Activate the ccus_gym conda environment
export CONDA_PREFIX=/scratch_tide/yc5224/envs/ccus_gym
export PATH=/scratch_tide/yc5224/envs/ccus_gym/bin:$PATH
export PYTHONPATH=/scratch_tide/yc5224/CYW_code:$PYTHONPATH
echo "ccus_gym environment activated"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
