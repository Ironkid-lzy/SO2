"""Frozen T011 setup and read-only identity utilities for the A1 baseline."""
import so2_compat
import ast
import copy
import hashlib
import json
import runpy
import shutil
import subprocess
import time
import types
from pathlib import Path
import h5py
from t010_edac import *
from ding.config import compile_config
from ding.policy import create_policy
from t014_a1_mask import install

WORK = ROOT/'_so2_work/t014/a1'
PROTO = WORK/'protocol.json'
CONFIG = ROOT/'scripts/repro/configs/halfcheetah_medium_replay_t010.py'
PARENT = Path('/home/lzy/Projects/SO2_t010/_so2_work/runs/EXP-025/EXP-025-s0-attempt1/ckpt/update_3000000.pth')
MANIFEST = Path('/home/lzy/Projects/SO2_t010/_so2_work/runs/EXP-026/EXP-026-s0-attempt2/initialization_manifest.json')
RUN = ROOT/'_so2_work/runs/EXP-035/EXP-035-s0-attempt1'
BASE_SHA = '30757a74072340cc7ec099267e9a7e091cf00f05'
PARENT_SHA = '799d8c07aff11009790b00650276e4e39ec96cc6d63739bf8d80070c2e034267'


def setup_config(device, exp_name=None, online=None):
    fresh = runpy.run_path(str(CONFIG))
    cfg, create = fresh['main_config'], fresh['create_config']
    cfg.policy.cuda = device == 'cuda'
    if online is not None:
        cfg.policy.learn.online = online
    cfg.policy.learn.actor_q_aggregation = 'mean'
    if exp_name is not None:
        cfg.exp_name = exp_name
    return cfg, create


def make_policy(device='cpu', mode='independent'):
    install()
    cfg, create = setup_config(device, online=True)
    cfg = compile_config(cfg, seed=0, env=None, auto=True, create_cfg=create, save_cfg=False)
    cfg.policy.type += '_command'
    cfg.policy.import_names = ['ding.policy.edac','ding.policy.command_mode_policy_instance']
    policy = create_policy(cfg.policy, model=None, enable_field=['learn','collect','eval','command'])
    policy._a1_mode = mode
    return policy, cfg


def checked_weights():
    manifest = json.loads(MANIFEST.read_text())
    assert digest(PARENT) == manifest['parent_sha256'] == PARENT_SHA
    assert digest(DATASET) == DATA_SHA == manifest['dataset_sha256']
    saved = torch.load(PARENT, map_location='cpu', weights_only=False)
    assert saved['updates'] == 3000000 and saved['train_env_steps'] == 0
    assert saved['identity']['seed'] == 0
    weights = {k:saved[k] for k in ('model','target_model')}
    assert all(tensor_hash(weights[k]) == manifest['network_hashes'][k] for k in weights)
    assert digest(manifest['export_path']) == manifest['export_sha256']
    export = torch.load(manifest['export_path'], map_location='cpu', weights_only=True)
    assert all(tensor_hash(export[k]) == tensor_hash(weights[k]) for k in weights)
    return weights, manifest


def load(policy, weights):
    policy.learn_mode.load_state_dict(weights)
    assert tensor_hash(policy._model.state_dict()) == tensor_hash(weights['model'])
    assert tensor_hash(policy._target_model.state_dict()) == tensor_hash(weights['target_model'])


def fixed_batch(n):
    # First valid transitions of d4rl.qlearning_dataset, without an env instance.
    with h5py.File(DATASET,'r') as f:
        limit = n + 100
        timeouts = f['timeouts'][:limit]
        idx = np.flatnonzero(~timeouts.astype(bool))[:n]
        observations = f['observations'][:limit+1]
        actions = f['actions'][:limit+1]
        rewards = f['rewards'][:limit+1]
        terminals = f['terminals'][:limit+1]
    assert len(idx) == n and not terminals[idx].any()
    return [dict(obs=torch.tensor(observations[i],dtype=torch.float32),
                 next_obs=torch.tensor(observations[i+1],dtype=torch.float32),
                 action=torch.tensor(actions[i],dtype=torch.float32),
                 reward=torch.tensor(rewards[i],dtype=torch.float32),
                 done=torch.tensor(bool(terminals[i]))) for i in idx]


def original_critic_forward():
    import ding.model.template.ensemble_qac as module
    text = subprocess.check_output(['git','show',BASE_SHA+':SO2/ding/model/template/ensemble_qac.py'],cwd=ROOT,text=True)
    cls = next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='ParallelizedEnsembleFlattenMLP')
    fn = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    scope = dict(module.__dict__)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'frozen-original-critic','exec'),scope)
    return scope['forward']


def bind_original(policy):
    function = original_critic_forward()
    for model in (policy._model, policy._target_model):
        model.critic.forward = types.MethodType(function,model.critic)


def equivalent(left, right, path=''):
    if isinstance(left,torch.Tensor):
        assert torch.equal(left,right),path
    elif isinstance(left,dict):
        assert left.keys()==right.keys(),path
        for key in left: equivalent(left[key],right[key],path+'/'+str(key))
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right),path
        for i,(a,b) in enumerate(zip(left,right)): equivalent(a,b,path+'/'+str(i))
    elif isinstance(left,np.ndarray):
        assert np.array_equal(left,right),path
    else:
        assert left==right,path


def source_hashes():
    files=[ROOT/'SO2/ding/model/template/ensemble_qac.py',ROOT/'SO2/ding/policy/edac.py',
           ROOT/'SO2/ding/entry/serial_entry_offline2online.py',CONFIG,
           ROOT/'scripts/repro/configs/halfcheetah_medium_replay_t005.py',
           ROOT/'scripts/repro/t011_guards.py']+sorted((ROOT/'scripts/repro').glob('t014_a1_*.py'))
    return {str(p.relative_to(ROOT)):digest(p) for p in files}
