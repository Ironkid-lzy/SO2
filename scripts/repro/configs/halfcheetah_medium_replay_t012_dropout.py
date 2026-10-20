"""T012 structure only; frozen T010/T011 training protocol."""
from pathlib import Path
import runpy
_base=runpy.run_path(str(Path(__file__).with_name("halfcheetah_medium_replay_t010.py")))
main_config=_base["main_config"];create_config=_base["create_config"]
main_config.policy.model.critic_layer_norm=False
main_config.policy.model.critic_dropout_rate=0.005
main_config.policy.learn.actor_q_aggregation="mean"
