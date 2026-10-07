#!/usr/bin/env bash
set -euo pipefail
export D4RL_SUPPRESS_IMPORT_ERROR=1
export LD_LIBRARY_PATH=/home/lzy/.mujoco/mujoco210/bin:/usr/lib/nvidia
export PYTHONPATH=/home/lzy/Projects/SO2_t006/SO2
cd /home/lzy/Projects/SO2_t006
mkdir -p _so2_work
exec >> _so2_work/t006_queue.log 2>&1
exec /home/lzy/Projects/SO2/.venv/bin/python scripts/repro/t006_queue.py
