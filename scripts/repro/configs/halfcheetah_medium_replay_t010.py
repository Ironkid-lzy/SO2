"""Frozen same-network seed0 online protocol; parent supplied by queue."""
from pathlib import Path
import runpy
_config = runpy.run_path(str(Path(__file__).with_name('halfcheetah_medium_replay_t005.py')))
main_config = _config['main_config']
create_config = _config['create_config']
main_config.policy.model.actor_log_sigma_bounds = (-20., 2.)
main_config.policy.model.actor_head_layer_num = 2  # Initial hidden layer + two head hidden layers.
