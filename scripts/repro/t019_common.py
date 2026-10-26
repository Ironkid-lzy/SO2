"""Frozen T019 identities, initialization and phase admission helpers."""
import os
import subprocess
import time
from pathlib import Path
from t010_edac import *
BASEWORK=ROOT/'_so2_work/t019'
SEED=int(os.environ['SO2_T019_SEED']);assert SEED in (1,2,3,4)
WORK=BASEWORK/f'seed{SEED}'
PROTO=BASEWORK/'protocol.json'
CONFIG=ROOT/'scripts/repro/configs/halfcheetah_medium_replay_t010.py'
RECIPE=ROOT/'scripts/repro/configs/t010_recipe.json'
OFFLINE=ROOT/f'_so2_work/runs/EXP-039/EXP-039-s{SEED}-attempt1'
ONLINE=ROOT/f'_so2_work/runs/EXP-040/EXP-040-s{SEED}-attempt1'
INITIAL=next(r for r in json.loads(PROTO.read_text())['initializations'] if r['seed']==SEED)
ARCHIVE=Path(INITIAL['archive_update0_path']);ARCHIVE_SHA=INITIAL['archive_update0_sha256']
VALIDATION_WORK=Path('/home/lzy/Projects/SO2_t018/_so2_work/t018')
VALIDATION_SHA='5fec0eff6e59f7c67380bdc76c3ee2f94a75f7fbd760bac5503184369f7fde1d'
TOTAL=151200


def read(p): return json.loads(Path(p).read_text())


def source_hashes():
    return read(BASEWORK/'formal-freeze.json')['source_sha256']


def frozen():
    r=read(BASEWORK/'formal-freeze.json')
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==r['code_sha']
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    assert all(digest(ROOT/p)==h for p,h in r['source_sha256'].items())
    assert digest(PROTO)==r['protocol_sha256']
    assert digest(VALIDATION_WORK/'semantic-validation-v1/validation.json')==VALIDATION_SHA
    assert read(VALIDATION_WORK/'validation-sealed.json')['status']=='completed'
    return r


def identity(seed=None):
    if seed is None:seed=SEED
    r=frozen();assert seed==SEED
    return dict(git_commit=r['code_sha'],recipe_sha256=digest(RECIPE),config_sha256=digest(CONFIG),
        reference_sha=REFERENCE_SHA,dataset_sha256=DATA_SHA,seed=SEED,actor_bounds=[-20,2],
        torch=torch.__version__,cuda=torch.version.cuda,python=sys.version,gpu=torch.cuda.get_device_name(0),
        command=sys.argv,cwd=str(ROOT),offline_ES_coefficient=0,protocol_sha256=digest(PROTO))


def initial_training_rng(trainer):
    assert digest(ARCHIVE)==ARCHIVE_SHA
    a=torch.load(ARCHIVE,map_location='cpu',weights_only=False)
    assert a['updates']==0 and a['train_env_steps']==0 and a['identity']['seed']==SEED
    for k,actual in [('model',trainer.model),('target_model',trainer.target)]:
        assert tensor_hash(actual.state_dict())==tensor_hash(a[k]),k
    assert torch.equal(trainer.log_alpha.detach().cpu(),a['log_alpha'])
    assert all(not a[k]['state'] for k in ('optimizer_policy','optimizer_q','optimizer_alpha'))
    # Fresh weights/optimizers; only archive0 training RNG is explicitly restored.
    restore_rng(a['rng'])
    return a


def historical_initial_returns():
    p=ARCHIVE.parents[1]/'metrics.jsonl'
    events=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return next(e['returns'] for e in events if e.get('event')=='evaluation' and e['gradient_updates']==0)


def phase_admission(phase,seconds):
    frozen();ledger=read(BASEWORK/'budget.json');job=next(j for j in ledger['jobs'] if j['job']==f's{SEED}-{phase}-v1')
    assert job['status']=='running' and job['pgid']==os.getpgrp()
    assert 0<seconds and seconds+15<=job['limit_seconds']
    assert ledger['used_seconds']<TOTAL
    return source_hashes()


def checked_new_parent():
    seal=read(WORK/'offline-sealed.json')
    assert seal['status']=='completed' and seal['quiescent']
    parent=OFFLINE/'ckpt/update_3000000.pth'
    assert seal['parent_sha256']==digest(parent)
    a=read(OFFLINE/'acceptance.json');assert a['accepted'] and a['parent_sha256']==seal['parent_sha256']
    s=torch.load(parent,map_location='cpu',weights_only=False)
    assert s['updates']==3000000 and s['train_env_steps']==0
    assert s['identity']['offline_ES_coefficient']==0 and s['identity']['seed']==SEED
    assert s['identity']['git_commit']==read(BASEWORK/'formal-freeze.json')['code_sha']
    assert s['identity']['dataset_sha256']==DATA_SHA
    return parent,s


def export_parent(out,parent=None,saved=None,final_required=True):
    from t010_loading import export_weights
    if final_required: parent,saved=checked_new_parent()
    assert saved is not None and parent is not None
    out.mkdir(parents=True,exist_ok=False)
    m=export_weights(saved,parent,out/'online_init.pth',identity())
    safe=torch.load(m['export_path'],map_location='cpu',weights_only=True)
    assert set(safe)=={'model','target_model'}
    assert all(tensor_hash(safe[k])==m['network_hashes'][k] for k in safe)
    atomic_json(out/'initialization_manifest.json',m)
    return safe,m
