"""CPU-only technical acceptance, invoked only after GPU process quiescence."""
import json,math
from pathlib import Path
import torch
from t018_control import digest,read,WORK

def verify(phase,run):
    if phase=='preflight':
        x=read(run/'result.json');assert x['passed'] and x['single_speed_session']
        return dict(result_sha256=digest(run/'result.json'),total_with30percent=x['total_with30percent'])
    meta=read(run/'run_meta.json');a=read(run/'acceptance.json')
    assert meta['status']=='completed' and a['accepted']
    rows=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines() if x.strip()]
    e=[x for x in rows if x['event']=='evaluation']
    assert len(e)==(31 if phase=='offline' else 41)
    assert all(len(x['returns'])==20 and all(math.isfinite(float(v)) for v in x['returns']) for x in e)
    if phase=='offline':
        assert [x['gradient_updates'] for x in e]==list(range(0,3000001,100000))
        assert a['updates']==dict(critic=3000000,actor=3000000,alpha=3000000)
        parent=run/'ckpt/update_3000000.pth';s=torch.load(parent,map_location='cpu',weights_only=False)
        assert s['updates']==3000000 and s['train_env_steps']==0
        assert s['identity']['offline_ES_coefficient']==0
        assert all(s[k]['state'] for k in ('optimizer_policy','optimizer_q','optimizer_alpha'))
        for k in ('optimizer_policy','optimizer_q','optimizer_alpha'):
            assert all(float(v['step'])==3000000 for v in s[k]['state'].values())
        assert torch.isfinite(s['log_alpha']).all()
        for k in ('model','target_model'):
            assert all(torch.isfinite(v).all() for v in s[k].values())
        manifest=read(run/'checkpoint_manifest.json');assert len(manifest)==31
        assert all(digest(run/x['path'])==x['sha256'] for x in manifest)
        return dict(parent_path=str(parent),parent_sha256=digest(parent),updates=3000000,evaluations=31,evaluation_interactions=a['evaluation_interactions_total'],peak_cuda_allocated_bytes=meta['peak_cuda_allocated_bytes'],peak_cuda_reserved_bytes=meta['peak_cuda_reserved_bytes'])
    assert [x['env_steps'] for x in e]==list(range(0,100001,2500))
    assert a['final_state']['updates']==dict(critic=950000,actor=95000,alpha=0)
    assert meta['paired_critic_updates']==950000
    assert read(run/'actual-parent-hook.json')['strict_tensor_hashes_match']
    assert read(run/'actual-initial-eval.json')['initial_returns_exact_new_parent']
    cp=run/'ckpt/envstep_100000.pth.tar';s=torch.load(cp,map_location='cpu',weights_only=False)
    for k in ('model','target_model'):assert all(torch.isfinite(v).all() for v in s[k].values())
    return dict(checkpoint_sha256=digest(cp),updates=a['final_state']['updates'],evaluations=41,evaluation_interactions=a['final_state']['evaluation_interactions_total'],peak_cuda_allocated_bytes=meta['peak_cuda_allocated_bytes'],peak_cuda_reserved_bytes=meta['peak_cuda_reserved_bytes'])
