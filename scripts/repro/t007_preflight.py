"""Read-only T007 identity and fixed-batch equivalence gate."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path('/home/lzy/Projects/SO2_t007')
RESEARCH = Path('/home/lzy/Projects/rl_sample_efficiency_research_t007')
OUTPUT = ROOT / '_so2_work/validation/EXP-018/preflight.json'
OLD = RESEARCH / 'reports/so2/SO2-T005/validation/t005.json'
NEW = OUTPUT.parent / 't007_fixed_batch.json'
CONFIG = ROOT / 'scripts/repro/configs/halfcheetah_medium_replay_t005.py'
CHECKPOINT = Path('/home/lzy/Projects/SO2/_so2_work/assets/checkpoints/ckpt/halfcheetah-medium-replay-v2.ckpt')
DATASET = Path('/home/lzy/.d4rl/datasets/halfcheetah_medium_replay-v2.hdf5')

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def strip_timing(value):
    if isinstance(value, dict):
        return {key: strip_timing(item) for key, item in value.items()
                if key not in ('wall_sec', 'update_seconds')}
    if isinstance(value, list):
        return [strip_timing(item) for item in value]
    return value

old = json.loads(OLD.read_text())
new = json.loads(NEW.read_text())
same_batch = old['batch_sha256'] == new['batch_sha256'] and old['batch_indices'] == new['batch_indices']
same_d = strip_timing(old['groups']['D']) == strip_timing(new['groups']['D'])
original_runner = (ROOT / 'scripts/repro/t005_run.py').read_text()
new_runner = (ROOT / 'scripts/repro/t007_run.py').read_text()
expected_runner = original_runner.replace(
    'Frozen SO2 + DroQ-style critic single-seed runner.',
    'Frozen SO2 + DroQ-style critic T007 seed runner.').replace(
    'assert args.seed == 0', 'assert args.seed in (1, 2, 3, 4)')
assert 'rng_snapshot_dir' not in new_runner
runner_exact = new_runner == expected_runner
code_diff = subprocess.check_output([
    'git', 'diff', '--name-only', 'c1ed167618ce210dea7c974ffee28aba02b730cc',
    '9741aa16b7afab8bb9d1d15ef58c52091a7f974b', '--',
    'SO2/ding/model/template/ensemble_qac.py', 'SO2/ding/policy/edac.py',
    'SO2/ding/entry/serial_entry_offline2online.py',
    'scripts/repro/configs/halfcheetah_medium_replay_t005.py'],
    cwd=ROOT, text=True).splitlines()
only_optional_entry_change = code_diff == ['SO2/ding/entry/serial_entry_offline2online.py']
work = ROOT / '_so2_work/runs/EXP-018'
no_existing_runs = not work.exists() and not (ROOT / '_so2_work/EXP-018.queue.lock').exists()
no_existing_research_raw = not (RESEARCH / 'results/raw/EXP-018').exists()
checks = dict(config_sha256=digest(CONFIG) == 'b314b70c161db8359b5d24753244776ee5a065591a4246d03f5f9374129d552f',
              checkpoint_sha256=digest(CHECKPOINT) == '02d74b7860620f1f9863a2f217aff0abfaae37f4a69ba6d0e1a0ea10f98d94f7',
              dataset_sha256=digest(DATASET) == '48d494a4770c11f48260736dec090b78bdca8375647ed4d24bfa5b6610c7f683',
              original_fixed_batch=same_batch, D_equivalent_excluding_timing=same_d,
              runner_only_seed_gate_and_docstring_changed=runner_exact,
              no_rng_snapshot_argument=not ('rng_snapshot_dir' in new_runner),
              core_diff_only_optional_entry_change=only_optional_entry_change,
              no_existing_runs=no_existing_runs,
              no_existing_research_raw=no_existing_research_raw,
              fixed_batch_updates_within_budget=new['validation_updates'] <= 200)
result = dict(status='passed' if all(checks.values()) else 'failed', checks=checks,
              fixed_batch_updates=new['validation_updates'], online_environment_steps=0,
              code_diff=code_diff, config_sha256=digest(CONFIG),
              checkpoint_sha256=digest(CHECKPOINT), dataset_sha256=digest(DATASET),
              old_D=old['groups']['D'], new_D=new['groups']['D'],
              effective_config_diff={'seed': '1,2,3,4 instead of 0',
                                     'run_dir': 'EXP-018 instead of EXP-016',
                                     'max_env_steps': 'unchanged 100000',
                                     'max_wall_clock_sec': 'unchanged 28800'})
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps({'status': result['status'], 'checks': checks, 'output': str(OUTPUT)}))
assert result['status'] == 'passed'
