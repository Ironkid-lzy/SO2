#!/usr/bin/env bash
set -euo pipefail
cd /home/lzy/Projects/SO2_t008
export LD_LIBRARY_PATH=/home/lzy/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/lib/x86_64-linux-gnu:/home/lzy/Projects/SO2/.venv/lib/python3.10/site-packages/torch/lib
export MUJOCO_GL=osmesa
export D4RL_SUPPRESS_IMPORT_ERROR=1
exec /home/lzy/Projects/SO2/.venv/bin/python scripts/repro/t008_queue.py
