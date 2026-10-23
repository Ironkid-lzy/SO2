#!/usr/bin/env bash
set -euo pipefail
cd /home/lzy/Projects/SO2_t014
export LD_LIBRARY_PATH=/home/lzy/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu:/home/lzy/Projects/SO2/.venv/lib/python3.10/site-packages/torch/lib
export MUJOCO_GL=osmesa D4RL_SUPPRESS_IMPORT_ERROR=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec /home/lzy/Projects/SO2/.venv/bin/python "$@"
