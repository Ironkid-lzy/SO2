from typing import Union, Optional, List, Any, Tuple
import os
import sys
import json
import time
import torch
import logging
import random
import numpy as np
from functools import partial
from tensorboardX import SummaryWriter

from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner, InteractionSerialEvaluator, BaseSerialCommander, create_buffer, \
    create_serial_collector
from ding.config import read_config, compile_config
from ding.policy import create_policy, PolicyFactory
from ding.utils import set_pkg_seed
from ding.utils.data import create_dataset

from torch.utils.data import DataLoader


def serial_pipeline_offline2online(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        max_iterations: Optional[int] = int(1e6),
        max_env_steps: Optional[int] = None,
        max_wall_clock_sec: Optional[float] = None,
        eval_env_step_interval: Optional[int] = None,
        eval_n_episode: Optional[int] = None,
        metrics_path: Optional[str] = None,
        policy_probe_path: Optional[str] = None,
) -> 'Policy':  # noqa
    """
    Overview:
        Serial pipeline entry.
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - env_setting (:obj:`Optional[List[Any]]`): A list with 3 elements: \
            ``BaseEnv`` subclass, collector env config, and evaluator env config.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - max_iterations (:obj:`Optional[torch.nn.Module]`): Learner's max iteration. Pipeline will stop \
            when reaching this iteration.
        - max_env_steps (:obj:`Optional[int]`): T001 explicit engineering stop budget. Total training \
            environment steps as reported by ``collector.envstep`` (which already includes \
            ``random_collect_size``). ``None`` keeps upstream behaviour (hard-coded stop at 150k policy \
            env-steps). This only bounds *when* the loop stops; it does not change the learning rule.
        - max_wall_clock_sec (:obj:`Optional[float]`): T001 explicit engineering wall-clock stop budget \
            for the online loop. ``None`` disables the wall-clock bound.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
    """
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = input_cfg
    create_cfg.policy.type = create_cfg.policy.type + '_command'
    env_fn = None if env_setting is None else env_setting[0]
    cfg = compile_config(cfg, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg, save_cfg=True)
    dataset = create_dataset(cfg)
    
    # Create main components: env, policy
    if env_setting is None:
        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    else:
        env_fn, collector_env_cfg, evaluator_env_cfg = env_setting
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=eval_env_step_interval is not None)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'collect', 'eval', 'command'])

    # Create worker components: learner, collector, evaluator, replay buffer, commander.
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = create_serial_collector(
        cfg.policy.collect.collector,
        env=collector_env,
        policy=policy.collect_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name
    )
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    replay_buffer = create_buffer(cfg.policy.other.replay_buffer, tb_logger=tb_logger, exp_name=cfg.exp_name)
    if cfg.policy.learn.get('offline_buffer_size', None):
        cfg.policy.other.replay_buffer.replay_buffer_size=cfg.policy.learn.offline_buffer_size
    else:
        cfg.policy.other.replay_buffer.replay_buffer_size=len(dataset)
    offline_buffer=create_buffer(cfg.policy.other.replay_buffer, tb_logger=tb_logger, exp_name=cfg.exp_name, instance_name='offline_buffer')
    for d in dataset.data:
        d['reward']=d['reward'].reshape(-1)
    import random
    offline_buffer.push(random.sample(dataset.data, int(len(dataset.data)*cfg.policy.learn.get('offline_data_ratio', 1))), cur_collector_envstep=collector.envstep)
    commander = BaseSerialCommander(
        cfg.policy.other.commander, learner, collector, evaluator, replay_buffer, policy.command_mode
    )
    # ==========
    # Main loop
    # ==========
    # T001 engineering stop budget: upstream had a hard-coded `sys.exit(0)` once the
    # collector reached 150k env-steps, which skips `after_run` and therefore the
    # final checkpoint save. Replace it with an explicit, logged, configurable stop.
    # Nothing below changes the loss, target computation, sampling or update counts.
    _budget_t0 = time.time()
    _online_steps = 0
    _timings = dict(collect=0.0, sample=0.0, learn=0.0, eval=0.0, save=0.0)
    _updates = dict(critic=0, actor=0, alpha=0)
    _evaluated_steps = set()
    _initial_offline_count = offline_buffer.count()
    _initial_online_count = replay_buffer.count()
    _offline_capacity = offline_buffer.replay_buffer_size
    _online_capacity = replay_buffer.replay_buffer_size

    def _cuda_sync():
        if cfg.policy.cuda:
            torch.cuda.synchronize()

    def _snapshot(event):
        record = {
            'event': event, 'env_steps': int(collector.envstep),
            'collected_samples': int(collector._total_train_sample_count),
            'online_iterations': _online_steps, 'learner_train_iter': int(learner.train_iter),
            'updates': dict(_updates), 'batch_size': 2560, 'online_batch': 256,
            'mixed_buffer_batch': 2304, 'auto_alpha': bool(policy._auto_alpha),
            'online_buffer': {'capacity': _online_capacity, 'count': replay_buffer.count(),
                              'push_count': replay_buffer.push_count,
                              'evictions': max(0, replay_buffer.push_count - _online_capacity)},
            'mixed_buffer': {'capacity': _offline_capacity, 'count': offline_buffer.count(),
                             'push_count': offline_buffer.push_count,
                             'evictions': max(0, offline_buffer.push_count - _offline_capacity)},
            'initial_buffer_counts': {'online': _initial_online_count, 'mixed': _initial_offline_count},
            'timings_sec': dict(_timings), 'wall_clock_sec': time.time() - _budget_t0,
        }
        print('[T002_STATE] ' + json.dumps(record), flush=True)
        if metrics_path:
            with open(metrics_path, 'a') as f:
                f.write(json.dumps(record) + '\n')

    def _eval_now():
        step = int(collector.envstep)
        if step in _evaluated_steps:
            return
        py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if cfg.policy.cuda else None
        start = time.perf_counter()
        try:
            stop_eval, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter,
                                               step, n_episode=eval_n_episode)
            returns = evaluator.last_episode_reward
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
        _timings['eval'] += time.perf_counter() - start
        assert len(returns) == eval_n_episode, (len(returns), eval_n_episode)
        import d4rl
        score = float(d4rl.get_normalized_score(cfg.env.env_id, float(reward)))
        record = {'event': 'evaluation', 'env_steps': step, 'learner_train_iter': int(learner.train_iter),
                  'episodes_requested': eval_n_episode, 'episodes_completed': len(returns),
                  'returns': returns, 'return_mean': float(reward),
                  'normalized_score': score, 'normalized_score_100': 100 * score,
                  'stop_value_reached': bool(stop_eval)}
        _evaluated_steps.add(step)
        print('[T002_EVAL] ' + json.dumps(record), flush=True)
        if metrics_path:
            with open(metrics_path, 'a') as f:
                f.write(json.dumps(record) + '\n')
        if policy_probe_path:
            obs = torch.linspace(-1, 1, int(cfg.policy.model.obs_shape))
            action = torch.as_tensor(policy._forward_eval({0: obs})[0]['action']).detach().cpu()
            with open(policy_probe_path, 'a') as f:
                f.write(json.dumps({'env_steps': step, 'observation': obs.tolist(),
                                    'live_action_before_save': action.tolist()}) + '\n')
        save_start = time.perf_counter()
        learner.save_checkpoint('envstep_%d.pth.tar' % step)
        _timings['save'] += time.perf_counter() - save_start
        _snapshot('evaluation_state')

    def _budget_reason():
        if max_wall_clock_sec is not None and (time.time() - _budget_t0) >= max_wall_clock_sec:
            return 'wall_clock'
        if max_env_steps is not None and collector.envstep >= max_env_steps:
            return 'env_steps'
        return None

    def _emit_budget(reason):
        rec = {
            'event': 'budget_stop',
            'reason': reason,
            # `collector.envstep` counts every env interaction handled by the collector,
            # including the random-collect phase (SampleSerialCollector increments
            # `_total_envstep_count` in its transition loop). It therefore IS the total
            # training env-step count; random_collect_size must not be added on top.
            'collector_envstep_total': int(collector.envstep),
            'random_collect_size': int(cfg.policy.random_collect_size),
            'online_iterations': int(_online_steps),
            'learner_train_iter': int(learner.train_iter),
            'wall_clock_sec': round(time.time() - _budget_t0, 3),
            'max_env_steps': max_env_steps,
            'max_wall_clock_sec': max_wall_clock_sec,
        }
        print('[T001_BUDGET] ' + json.dumps(rec), flush=True)

    # Learner's before_run hook.
    learner.call_hook('before_run')
    stop = False
    # Accumulate plenty of data at the beginning of training.
    for _ in range(learner.train_iter, cfg.policy.learn.offline_pretrain_iterations):
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter)
            if stop:
                break
        train_data = offline_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
        cfg.policy.learn.online = False
        # learner.train(train_data, collector.envstep)
        policy._cfg.learn.only_value = True
        learner.train(train_data, collector.envstep)
        policy._cfg.learn.only_value = False
    
    if eval_env_step_interval is not None:
        assert eval_env_step_interval > 0 and eval_n_episode is not None
        _eval_now()
    if cfg.policy.get('random_collect_size', 0) > 0:
        collector.reset_policy(policy.collect_mode)
        collect_kwargs = commander.step()
        remaining = int(cfg.policy.random_collect_size)
        while remaining > 0:
            if _budget_reason() is not None:
                break
            segment = min(remaining, eval_env_step_interval or remaining)
            start = time.perf_counter()
            new_data = collector.collect(n_sample=segment, policy_kwargs=collect_kwargs)
            replay_buffer.push(new_data, cur_collector_envstep=0)
            _timings['collect'] += time.perf_counter() - start
            remaining -= len(new_data)
            if eval_env_step_interval is not None and collector.envstep % eval_env_step_interval == 0:
                _eval_now()
    if _budget_reason() is not None:
        _emit_budget(_budget_reason())
        _snapshot('budget_stop')
        learner.call_hook('after_run')
        return policy, stop
    
    for _ in range(cfg.policy.learn.offline_pretrain_iterations, 
            cfg.policy.learn.get('online_pretrain_iterations', cfg.policy.learn.offline_pretrain_iterations)):
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter)
            if stop:
                break
        if cfg.policy.learn.get('concat_online_linear', False):
            concat_online_ratio = min(cfg.policy.learn.concat_online_ratio / cfg.policy.learn.concat_online_tmax * collector.envstep, cfg.policy.learn.concat_online_ratio)
        else:
            concat_online_ratio = cfg.policy.learn.concat_online_ratio
        batch_size = min(int(256/concat_online_ratio), 8192)
        online_bs = int(batch_size * concat_online_ratio)
        offline_bs = batch_size - online_bs
        train_data = replay_buffer.sample(online_bs, learner.train_iter)
        if cfg.policy.learn.get('pretrain_with_offline', False):
            # train_data = replay_buffer.sample(online_bs, learner.train_iter)
            offlinetrain_data = offline_buffer.sample(offline_bs, learner.train_iter)
            train_data.extend(offlinetrain_data)
        policy._cfg.learn.only_value = True
        learner.train(train_data, collector.envstep)
        policy._cfg.learn.only_value = False

    for _ in range(max_iterations):
        _reason = _budget_reason()
        if _reason is not None:
            _emit_budget(_reason)
            break
        if eval_env_step_interval is None and evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        collect_kwargs = commander.step()
        # Collect data by default config n_sample/n_episode
        if cfg.policy.learn.concat_online_ratio > 0:
            start = time.perf_counter()
            new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
            replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
            offline_buffer.push(new_data, cur_collector_envstep=collector.envstep)
            _timings['collect'] += time.perf_counter() - start
        # Learn policy from collected data

        for i in range(cfg.policy.learn.update_per_collect):
            # Learner will train ``update_per_collect`` times in one iteration.
            if cfg.policy.learn.concat_online_ratio > 0:
                cfg.policy.learn.online = True
                if cfg.policy.learn.get('concat_online_linear', False):
                    concat_online_ratio = min(cfg.policy.learn.concat_online_ratio / cfg.policy.learn.concat_online_tmax * collector.envstep, cfg.policy.learn.concat_online_ratio)
                else:
                    concat_online_ratio = cfg.policy.learn.concat_online_ratio
                batch_size = min(int(256/concat_online_ratio), 8192)
                online_bs = int(batch_size * concat_online_ratio)
                offline_bs = batch_size - online_bs
                sample_start = time.perf_counter()
                train_data = replay_buffer.sample(online_bs, learner.train_iter)
                offlinetrain_data = offline_buffer.sample(offline_bs, learner.train_iter)
                train_data.extend(offlinetrain_data)
                _timings['sample'] += time.perf_counter() - sample_start
                assert len(train_data) == batch_size
            else:
                if random.uniform(0, 1) < cfg.policy.learn.online_ratio:
                    train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
                    cfg.policy.learn.online = False # use min_q_weight in forward not init in cql 
                    if train_data is None:
                        # It is possible that replay buffer's data count is too few to train ``update_per_collect`` times
                        logging.warning(
                            "Replay buffer's data can only train for {} steps. ".format(i) +
                            "You can modify data collect config, e.g. increasing n_sample, n_episode."
                        )
                        break
                else:
                    train_data = offline_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
                    cfg.policy.learn.online = False
            _cuda_sync()
            learn_start = time.perf_counter()
            before_cnt = policy._forward_learn_cnt
            learner.train(train_data, collector.envstep)
            _cuda_sync()
            _timings['learn'] += time.perf_counter() - learn_start
            assert policy._forward_learn_cnt == before_cnt + 1
            _updates['critic'] += 1
            if cfg.policy.learn.online and not policy._cfg.learn.only_value and (
                    (before_cnt + 1) % policy._actor_update_freq == 0):
                _updates['actor'] += 1
            if policy._auto_alpha and not policy._cfg.learn.only_value:
                _updates['alpha'] += 1
            # if learner.policy.get_attribute('priority'):
                # replay_buffer.update(learner.priority_info)
        _online_steps += 1
        if eval_env_step_interval is not None and collector.envstep % eval_env_step_interval == 0:
            _eval_now()
        if _online_steps % 100 == 0:
            _emit_budget('periodic')

    # T001: if the loop ended because of `max_iterations` / evaluator stop, still
    # log the budget record so the smoke log is self-describing.
    _emit_budget('loop_end')
    _snapshot('final')

    # Learner's after_run hook.
    learner.call_hook('after_run')
    return policy, stop
