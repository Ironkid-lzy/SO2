"""Frozen D04: read-only ES/action/parameter-gradient diagnostics, never update."""
import json
import signal
import time
from pathlib import Path
signal.alarm(840)
START = time.monotonic()
import so2_compat
import h5py
import numpy as np
import torch
from t010_edac import make_model, OfflineEDAC, q_values, digest, tensor_hash, seed_all, rng_state, restore_rng

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT/'_so2_work/t017'


def angle(a,b):
    na,nb=np.linalg.norm(a,axis=-1),np.linalg.norm(b,axis=-1)
    valid=(na>1e-12)&(nb>1e-12)
    values=np.full(na.shape,np.nan)
    values[valid]=np.sum(a[valid]*b[valid],axis=-1)/(na[valid]*nb[valid])
    return dict(defined_count=int(valid.sum()),undefined_count=int((~valid).sum()),
                mean=float(values[valid].mean()) if valid.any() else None)


def geom_metrics(g):
    g=g.astype(np.float64)
    norms=np.linalg.norm(g,axis=-1)
    v=g/(norms[...,None]+1e-10)
    m=v.mean(axis=0);mg=g.mean(axis=0)
    pair=2*np.sum(v[:,0]*v[:,1],axis=-1).mean()
    crossed=2*np.sum(m[0]*m[1],axis=-1).mean()
    return dict(pair_es_mean=float(pair),crossed_es_mean=float(crossed),
        resultant_norms_mean=np.linalg.norm(m,axis=-1).mean(axis=1).tolist(),
        normalized_mean_direction_cos=angle(m[0],m[1]),
        raw_mean_gradient_cos=angle(mg[0],mg[1]),
        raw_gradient_norms_mean=norms.mean(axis=(0,2)).tolist(),
        raw_head_average_gradient_norm_mean=float(np.linalg.norm(mg.mean(axis=0),axis=-1).mean()),
        smallest_sample_raw_gradient_norm=float(norms.min()))


def es_only(model,batch):
    ot=batch['observations'].unsqueeze(0).repeat(2,1,1)
    at=batch['actions'].detach().unsqueeze(0).repeat(2,1,1).requires_grad_(True)
    q=q_values(model,ot,at)
    g,=torch.autograd.grad(q.sum(),at,create_graph=True)
    v=g/(g.norm(dim=-1,keepdim=True)+1e-10)
    return (2*(v[0]*v[1]).sum(-1)).mean(),g


def parameter_vector(value,params):
    gs=torch.autograd.grad(value,params,allow_unused=True,retain_graph=True)
    unused=[i for i,g in enumerate(gs) if g is None]
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                      for p,g in zip(params,gs)]).cpu().numpy(),unused


def main():
    protocol=WORK/'diagnostic-protocol.json';p=json.loads(protocol.read_text())
    out=WORK/'d04-es-final-v1';assert not out.exists();out.mkdir()
    assert all(digest(ROOT/k)==v for k,v in p['source_sha256'].items())
    assert digest(p['dataset_path'])==p['dataset_sha256']
    assert all(digest(x['path'])==x['sha256'] for x in p['parents'])
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available();torch.cuda.reset_peak_memory_stats()
    with h5py.File(p['dataset_path'],'r') as f:
        valid=np.flatnonzero(~f['timeouts'][:-1].astype(bool))
        ix=np.random.default_rng(p['sample_seed']).choice(valid,p['batch_size'],replace=False)
        rows=np.sort(ix)
        data={k:f[k][rows][np.searchsorted(rows,ix)] for k in ['observations','actions','rewards','terminals']}
        data['next_observations']=f['observations'][rows+1][np.searchsorted(rows,ix)]
    assert not data['terminals'].any()
    arrays={'indices':ix,**{'data_'+k:v for k,v in data.items()}};records=[]
    for parent in p['parents']:
        seed=parent['seed'];seed_all(9000+seed,True)
        saved=torch.load(parent['path'],map_location='cpu',weights_only=False)
        assert saved['updates']==3000000 and saved['train_env_steps']==0 and saved['identity']['seed']==seed
        trainer=OfflineEDAC(make_model('cuda'))
        trainer.model.load_state_dict(saved['model'],strict=True)
        trainer.target.load_state_dict(saved['target_model'],strict=True)
        with torch.no_grad():trainer.log_alpha.copy_(saved['log_alpha'].to('cuda'))
        trainer.model.train();trainer.target.train()
        before=[tensor_hash(x.state_dict()) for x in [trainer.model,trainer.target]]
        assert before==[tensor_hash(saved[k]) for k in ['model','target_model']]
        batch={k:torch.tensor(v,device='cuda',dtype=torch.float32).reshape(p['batch_size'],-1) for k,v in data.items()}
        g_samples=[];seed_all(p['geometry_seed']+seed,True)
        for k in range(p['mask_samples']):
            ot=batch['observations'].unsqueeze(0).repeat(2,1,1)
            at=batch['actions'].detach().unsqueeze(0).repeat(2,1,1).requires_grad_(True)
            q=q_values(trainer.model,ot,at)
            g,=torch.autograd.grad(q.sum(),at)
            g_samples.append(g.cpu().numpy())
        g=np.stack(g_samples);arrays[f'action_grad_s{seed}']=g
        assert np.isfinite(g).all()
        trainer.model.eval()
        ot=batch['observations'].unsqueeze(0).repeat(2,1,1)
        at=batch['actions'].detach().unsqueeze(0).repeat(2,1,1).requires_grad_(True)
        q=q_values(trainer.model,ot,at);off,=torch.autograd.grad(q.sum(),at)
        arrays[f'off_action_grad_s{seed}']=off.cpu().numpy();trainer.model.train()
        named=list(trainer.model.critic.named_parameters());params=[v for _,v in named]
        tds=[];ess=[];loss_records=[];unused_records=[]
        for draw in range(p['parameter_draws']):
            seed_all(p['parameter_seed']+seed*100+draw,True)
            states=[]
            def hook(module,args):
                states.append(rng_state())
            handle=trainer.model.critic.register_forward_pre_hook(hook)
            losses=trainer.losses(batch);handle.remove()
            assert len(states)==3,'native current critic queries must be actor,TD,ES'
            assert all(torch.isfinite(v) for v in losses.values())
            td,unused_td=parameter_vector(losses['td'],params)
            es,unused_es=parameter_vector(losses['es'],params)
            restore_rng(states[-1]);reconstructed,gexact=es_only(trainer.model,batch)
            check,_=parameter_vector(reconstructed,params)
            np.testing.assert_allclose(float(reconstructed.detach()),float(losses['es'].detach()),rtol=2e-5,atol=2e-6)
            np.testing.assert_allclose(check,es,rtol=2e-5,atol=2e-6)
            assert np.isfinite(td).all() and np.isfinite(es).all()
            arrays[f'native_es_action_grad_s{seed}_d{draw}']=gexact.detach().cpu().numpy()
            tds.append(td);ess.append(es)
            loss_records.append({k:float(v.detach()) for k,v in losses.items()})
            unused_records.append(dict(td=[named[i][0] for i in unused_td],es=[named[i][0] for i in unused_es]))
        td,es=np.stack(tds),np.stack(ess)
        arrays[f'td_parameter_grad_s{seed}']=td;arrays[f'es_parameter_grad_s{seed}']=es
        td64,es64=td.astype(np.float64),es.astype(np.float64)
        tn,en=np.linalg.norm(td64,axis=1),np.linalg.norm(es64,axis=1)
        assert np.isfinite(tn).all() and np.isfinite(en).all() and (tn>1e-12).all() and (en>1e-12).all()
        dot=np.sum(td64*es64,axis=1)
        rec=dict(seed=seed,geometry=geom_metrics(g),
            mask_panels=[geom_metrics(x) for x in np.split(g,2,axis=0)],
            state_halves=[geom_metrics(x) for x in np.split(g,2,axis=2)],
            off_geometry=geom_metrics(arrays[f'off_action_grad_s{seed}'][None]),
            losses=loss_records,unused=unused_records,
            td_parameter_norms=tn.tolist(),es_parameter_norms=en.tolist(),
            parameter_norm_ratios=(en/tn).tolist(),parameter_cosines=(dot/(tn*en)).tolist(),
            source_reconstruction_checks=4,before_network_hashes=before,
            parameter_names=[k for k,_ in named],parameter_sizes=[x.numel() for x in params])
        assert [tensor_hash(x.state_dict()) for x in [trainer.model,trainer.target]]==before
        assert trainer.updates==0 and all(x.grad is None for x in trainer.model.parameters())
        assert not trainer.critic_optim.state and not trainer.actor_optim.state and not trainer.alpha_optim.state
        assert all(x.grad is None for x in trainer.target.parameters())
        records.append(rec)
        print(json.dumps(dict(seed=seed,geometry=rec['geometry'],parameter_norm_ratios=rec['parameter_norm_ratios'],parameter_cosines=rec['parameter_cosines'])),flush=True)
    np.savez_compressed(out/'gradients.npz',**arrays);torch.cuda.synchronize()
    result=dict(status='completed',records=records,protocol_sha256=digest(protocol),source_sha256=p['source_sha256'],
        raw_sha256=digest(out/'gradients.npz'),wall_core_seconds=time.monotonic()-START,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        environment_steps=0,training_optimizer_steps=0,learner_update_calls=0,
        torch_version=torch.__version__,device=torch.cuda.get_device_name())
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
