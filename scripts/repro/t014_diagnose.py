"""Preregistered conditional actor-gradient diagnostic, no optimizer or rollout."""
import json, math, random, statistics, subprocess, time
from pathlib import Path
import h5py
from torch.distributions import Independent
from t010_edac import DATASET, DATA_SHA, ROOT, atomic_json, digest, make_model, tensor_hash, torch, np

WORK = ROOT / '_so2_work/t014'
OUT = ROOT / '_so2_work/diagnostics/t014-action-mask-v1'
ARMS = ((1, 1), (1, 4), (4, 1))


def actor_loss(model, obs, eps, masks, mask_seed):
    """One shared logits forward; independent epsilon for each action branch."""
    mu, sigma = model(obs, mode='compute_actor')['logit']
    dist = Independent(torch.distributions.Normal(mu, sigma), 1)
    losses = []
    # Dropout consumes a private reproducible global stream inside fork_rng.
    with torch.random.fork_rng(devices=[0] if obs.is_cuda else []):
        if obs.is_cuda:
            torch.cuda.manual_seed(mask_seed)
        else:
            torch.manual_seed(mask_seed)
        for epsilon in eps:
            z = mu + sigma * epsilon
            action = z.tanh()
            logp = dist.log_prob(z) - torch.log(1 - action.square() + 1e-6).sum(-1)
            q = [model({'obs': obs, 'action': action}, mode='compute_critic')['q_value'].mean(0)
                 for _ in range(masks)]
            losses.append((.2 * logp - torch.stack(q).mean(0)).mean())
    return torch.stack(losses).mean()


def draw_gradient(model, obs, actions, masks, action_seed, mask_seed, fixed_eps=None):
    gen = torch.Generator(device=obs.device).manual_seed(action_seed)
    eps = torch.randn((actions, len(obs), 6), device=obs.device, generator=gen) if fixed_eps is None else fixed_eps
    loss = actor_loss(model, obs, eps, masks, mask_seed)
    # Original actor backward computes critic gradients too. Include their cost.
    all_params = tuple(model.parameters())
    grads = torch.autograd.grad(loss, all_params)
    by_id = {id(p): g for p, g in zip(all_params, grads)}
    result = torch.cat([by_id[id(p)].flatten() for p in model.actor.parameters()])
    return result


def timed_loss(model, obs, actions, masks, private_state):
    """Native first action/mask; only proposed extra branches incur RNG isolation."""
    mu, sigma = model(obs, mode='compute_actor')['logit']
    dist = Independent(torch.distributions.Normal(mu, sigma), 1)
    def branch():
        z = dist.rsample(); act = z.tanh()
        lp = dist.log_prob(z) - torch.log(1-act.square()+1e-6).sum(-1)
        q = model({'obs':obs,'action':act},mode='compute_critic')['q_value'].mean(0)
        return act, lp, q
    action, logp, first_q = branch()
    if actions == masks == 1:
        return (.2*logp-first_q).mean(), private_state
    losses = []
    with torch.random.fork_rng(devices=[0]):
        torch.cuda.set_rng_state(private_state)
        if actions == 4:
            losses = [(.2*logp-first_q).mean()]
            for _ in range(3):
                _, lp, q = branch(); losses.append((.2*lp-q).mean())
            loss = torch.stack(losses).mean()
        else:
            qs = [first_q]
            for _ in range(3):
                qs.append(model({'obs':obs,'action':action},mode='compute_critic')['q_value'].mean(0))
            loss = (.2*logp-torch.stack(qs).mean(0)).mean()
        private_state = torch.cuda.get_rng_state()
    return loss, private_state


def measure_costs(model, obs, ci, bi):
    # Outer diagnostic-only fork is outside every measured call.
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(140018+ci*4+bi);torch.cuda.manual_seed(140018+ci*4+bi)
        states = {i:torch.Generator(device='cuda').manual_seed(140019+i).get_state() for i in range(3)}
        def step(ai):
            a,k = ARMS[ai]
            model.zero_grad(set_to_none=True)
            loss,states[ai] = timed_loss(model,obs,a,k,states[ai])
            loss.backward()
        for ai in range(3):
            for _ in range(10):step(ai)
        order = list(range(3))*60;random.Random(140015+ci*4+bi).shuffle(order)
        timings = {f'A{a}K{k}': [] for a,k in ARMS};peaks = {key:0 for key in timings}
        for ai in order:
            a,k = ARMS[ai];key=f'A{a}K{k}'
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
            step(ai)
            torch.cuda.synchronize();elapsed=time.perf_counter()-t
            timings[key].append(elapsed);peaks[key]=max(peaks[key],torch.cuda.max_memory_allocated())
        model.zero_grad(set_to_none=True)
    return timings,peaks


def variance(x):
    return ((x - x.mean(0)).square().sum() / (len(x) - 1)).item()


def stats(x, ref):
    x = x.double(); ref = ref.double()
    v = variance(x)
    return {'trace_variance': v,
            'quarter_trace_variances': [variance(y) for y in x.chunk(4)],
            'mean_gradient_squared_norm': x.mean(0).square().sum().item(),
            'mean_squared_distance_to_finite_reference': (x - ref).square().sum(1).mean().item(),
            'squared_mean_difference_from_finite_reference': (x.mean(0) - ref).square().sum().item()}


def checkpoints():
    online = Path('/home/lzy/Projects/SO2_t011/_so2_work/runs/EXP-029/EXP-029-s0-attempt1')
    manifest = {r['path']: r for r in json.loads((WORK / 'frozen-checkpoint-manifest.json').read_text())}
    points = [('offline3M', Path('/home/lzy/Projects/SO2_t010/_so2_work/runs/EXP-025/EXP-025-s0-attempt1/ckpt/update_3000000.pth'),
               '799d8c07aff11009790b00650276e4e39ec96cc6d63739bf8d80070c2e034267')]
    for step in (25000, 100000):
        rel = f'ckpt/envstep_{step}.pth.tar'
        points.append((f'online{step}', online / rel, manifest[rel]['sha256']))
    return points


def toy_checks():
    # Exact finite law of total variance, including action-mask interaction.
    x = torch.tensor([[[b + 2*m + 3*b*m] for m in (-1., 1.)] for b in (-1., 1.)], dtype=torch.float64)
    inner = x.var(1, unbiased=False).sum(1).mean().item()
    outer = x.mean(1).var(0, unbiased=False).sum().item()
    assert inner == 13 and outer == 1
    return {'exact_action_component': outer, 'exact_mask_component': inner,
            'A1K1_population_variance': outer + inner,
            'A1K4_population_variance': outer + inner/4,
            'A4K1_population_variance': (outer + inner)/4}


def main():
    start = time.monotonic(); torch.set_num_threads(2)
    design = json.loads((WORK / 'diagnostic-design.json').read_text())
    assert design['batch_size'] == 2560 and design['direct_replicates_per_arm'] == 128
    OUT.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'code_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'design_sha256': digest(WORK / 'diagnostic-design.json'), 'dataset_sha256': digest(DATASET),
              'optimizer_updates': 0, 'environment_interactions': 0, 'toy_checks': toy_checks(), 'points': []}
    assert report['dataset_sha256'] == DATA_SHA
    atomic_json(OUT / 'report.json', report)
    with h5py.File(DATASET, 'r') as f:
        indices = [np.sort(np.random.default_rng(140014 + b).choice(len(f['observations']), 2560, replace=False)) for b in range(4)]
        batches = [torch.tensor(f['observations'][idx], device='cuda', dtype=torch.float32) for idx in indices]
    atomic_json(OUT / 'batch-indices.json', [x.tolist() for x in indices])
    global_cpu = torch.get_rng_state().clone(); global_cuda = torch.cuda.get_rng_state().clone()
    for ci, (label, path, sha) in enumerate(checkpoints()):
        assert path.exists() and digest(path) == sha, str(path)
        saved = torch.load(path, map_location='cpu', weights_only=False)
        model = make_model('cuda'); model.load_state_dict(saved['model'], strict=True); model.train()
        # Construction initializes weights; diagnostic sampling must preserve the post-load stream.
        before_cpu = torch.get_rng_state().clone(); before_cuda = torch.cuda.get_rng_state().clone()
        initial_hash = tensor_hash(model.state_dict())
        point = {'label': label, 'path': str(path), 'sha256': sha, 'batches': []}
        for bi, obs in enumerate(batches):
            base = 14001400 + ci * 100000 + bi * 10000
            nested = []
            for a in range(32):
                gen = torch.Generator(device='cuda').manual_seed(base + a)
                eps = torch.randn((1, len(obs), 6), device='cuda', generator=gen)
                inner = []
                for m in range(8):
                    inner.append(draw_gradient(model, obs, 1, 1, 0, base + 1000 + a*8 + m, eps).detach().cpu())
                nested.append(torch.stack(inner))
            nested = torch.stack(nested).double()
            assert torch.isfinite(nested).all()
            vm = nested.var(1, unbiased=True).sum(1).mean().item()
            va_raw = variance(nested.mean(1)) - vm/8
            del nested
            reference = torch.stack([draw_gradient(model, obs, 1, 1, base+2000+r, base+3000+r).detach().cpu()
                                     for r in range(128)]).double()
            assert torch.isfinite(reference).all()
            reference_mean = reference.mean(0)
            cell = {'batch': bi, 'nested': {'mask_trace': vm, 'action_trace_raw': va_raw,
                                           'estimated_A1K1_trace': va_raw+vm,
                                           'estimated_A1K4_trace': va_raw+vm/4,
                                           'estimated_A4K1_trace': (va_raw+vm)/4},
                    'reference_trace_variance': variance(reference),
                    'reference_mean_variance_trace_estimate': variance(reference)/128, 'arms': {}}
            del reference
            for ai, (actions, masks) in enumerate(ARMS):
                gs = torch.stack([draw_gradient(model, obs, actions, masks, base+4000+ai*1000+r, base+7000+ai*1000+r).detach().cpu()
                                  for r in range(128)])
                assert torch.isfinite(gs).all()
                cell['arms'][f'A{actions}K{masks}'] = stats(gs, reference_mean)
                # Full gradients preserved on remote only; compact statistics synced.
                torch.save(gs, OUT / f'{label}-b{bi}-A{actions}K{masks}-gradients.pt')
                del gs
            # Training-style backward timing is separate from gradient statistics.
            timings, peaks = measure_costs(model, obs, ci, bi)
            for key,values in timings.items():
                cell['arms'][key].update(seconds_mean=statistics.mean(values),seconds_median=statistics.median(values),
                                         seconds_p90=sorted(values)[53],peak_allocated_bytes=peaks[key])
            atomic_json(OUT/f'{label}-b{bi}-timings.json',timings)
            point['batches'].append(cell); report['active_point']=point
            atomic_json(OUT/'report.json',report)
        assert initial_hash == tensor_hash(model.state_dict())
        assert torch.equal(before_cpu,torch.get_rng_state()) and torch.equal(before_cuda,torch.cuda.get_rng_state())
        point['weights_unchanged']=point['global_rng_preserved']=True
        agg={}
        for a,k in ARMS:
            key=f'A{a}K{k}'
            v=statistics.mean(b['arms'][key]['trace_variance'] for b in point['batches'])
            c=statistics.mean(b['arms'][key]['seconds_mean'] for b in point['batches'])
            med=statistics.mean(b['arms'][key]['seconds_median'] for b in point['batches'])
            agg[key]={'trace_variance':v,'seconds_mean':c,'seconds_median':med,'variance_cost':v*c}
        ref=agg['A1K1']['variance_cost']
        for key,x in agg.items():
            x['variance_cost_ratio']=x['variance_cost']/ref
            x['median_cost_ratio']=x['trace_variance']*x['seconds_median']/(agg['A1K1']['trace_variance']*agg['A1K1']['seconds_median'])
        quarter_ratios=[]
        for q in range(4):
            def prod(key):
                return statistics.mean(b['arms'][key]['quarter_trace_variances'][q] for b in point['batches'])*agg[key]['seconds_mean']
            quarter_ratios.append(prod('A4K1')/prod('A1K1'))
        point['aggregate']=agg;point['A4K1_quarter_ratios']=quarter_ratios
        point['gate_point_pass']=agg['A4K1']['variance_cost_ratio']<=.8
        point['gate_unstable']=any(x>.8 for x in quarter_ratios)
        report['points'].append(point);report.pop('active_point',None)
        atomic_json(OUT/'report.json',report);del model
    report.update(status='completed',gate_points=sum(p['gate_point_pass'] and not p['gate_unstable'] for p in report['points']),
                  wall_seconds=time.monotonic()-start,interpretation='Conditional finite-checkpoint learned-surrogate gradient variance only. No real-gradient error or online efficacy claim. Reference has MC error.')
    report['diagnostic_gate_pass']=report['gate_points']>=2
    atomic_json(OUT/'report.json',report)
    # Automatically seal small evidence independent of a live model/chat.
    manifest=[]
    for p in sorted(OUT.iterdir()):
        if p.is_file():manifest.append({'path':str(p),'size':p.stat().st_size,'sha256':digest(p)})
    atomic_json(OUT/'artifact_manifest.json',manifest)
    print(json.dumps({'status':report['status'],'diagnostic_gate_pass':report['diagnostic_gate_pass'],'wall_seconds':report['wall_seconds']}))

if __name__ == '__main__':
    main()
