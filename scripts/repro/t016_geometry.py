"""One frozen read-only local-function diagnostic; no environments or optimizers."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

import h5py
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / '_so2_work/t016'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def source_forward(path):
    # Execute the frozen native forward, rather than approximating Dropout/LN.
    tree = ast.parse(Path(path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'ParallelizedEnsembleFlattenMLP')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    scope = {'torch': torch, 'F': F}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), scope)
    return scope['forward']


class FrozenCritic:
    """Bind checkpoint tensors to the existing native forward without a learner."""
    def __init__(self, state, native, device):
        state = {k: v.to(device) for k, v in state.items() if k.startswith('critic.')}
        self._tensors = state
        self.num_q = state['critic.fc0.W'].shape[0]
        self.use_layer_norm = 'critic.layer_norm_weights.0' in state
        self.dropout_rate = .005 if self.use_layer_norm else 0.
        self.training = True
        self.hidden_activation = F.relu
        self.output_activation = lambda x: x
        self.layer_norm = None
        self.fcs = []
        for i in range(3):
            w, b = state[f'critic.fc{i}.W'], state[f'critic.fc{i}.b']
            self.fcs.append(lambda h, n, w=w, b=b: h @ w[:n] + b[:n])
        w, b = state['critic.last_fc.W'], state['critic.last_fc.b']
        self.last_fc = lambda h, n: h @ w[:n] + b[:n]
        if self.use_layer_norm:
            self.layer_norm_weights = [state[f'critic.layer_norm_weights.{i}'] for i in range(3)]
            self.layer_norm_biases = [state[f'critic.layer_norm_biases.{i}'] for i in range(3)]
        self.forward = types.MethodType(native, self)
        assert all(not v.requires_grad for v in state.values())

    def values(self, obs, actions, masks=None):
        n, j, _ = actions.shape
        x = torch.cat((obs[:, None, :].expand(-1, j, -1), actions), dim=-1).reshape(n*j, -1)
        if masks is not None:
            self._dropout_pair_context = FixedMask(masks)
        y = self.forward(x)['pred'].reshape(self.num_q, n, j)
        assert getattr(self, '_dropout_pair_context', None) is None
        return y


class FixedMask:
    def __init__(self, masks):
        self.masks = masks

    def apply(self, h, layer, p, training):
        assert training and p == .005
        return h * self.masks[layer] / (1-p)


def actor_mean(state, obs, device):
    h = obs
    for key in ('actor.0', 'actor.2.main.0', 'actor.2.main.2'):
        h = F.relu(F.linear(h, state[key+'.weight'].to(device), state[key+'.bias'].to(device)))
    return F.linear(h, state['actor.2.mu.weight'].to(device), state['actor.2.mu.bias'].to(device))


def load(path):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    assert 'model' in obj and 'target_model' in obj
    state = obj['model']
    assert all(isinstance(v, torch.Tensor) and torch.isfinite(v).all() for v in state.values())
    return state


def covariance(functions):
    # Uniform empirical function distribution, not a claim of independent draws.
    centered = functions - functions.mean(axis=0, keepdims=True)
    return np.einsum('fnd,fne->nde', centered, centered) / len(functions)


def norm_cov(c):
    tr = np.trace(c, axis1=-2, axis2=-1)
    assert np.all(tr > 1e-12), 'unresolved/degenerate contrast covariance'
    return c / tr[:, None, None]


def contrasts(q, radius):
    return (q[..., 1::2] - q[..., 2::2]) / (2*radius)


def summarize(q, reference, radius):
    c = covariance(contrasts(q, radius))
    t = covariance(contrasts(reference, radius))
    cn, tn = norm_cov(c), norm_cov(t)
    shape = np.linalg.norm(cn-tn, axis=(1,2))
    td = np.zeros_like(t)
    ii = np.arange(6)
    td[:, ii, ii] = t[:, ii, ii]
    diagonal = np.linalg.norm(norm_cov(td)-tn, axis=(1,2))
    q_noise = q - q.mean(axis=0, keepdims=True)
    total = np.mean(q_noise*q_noise, axis=(0,2))
    offset = np.mean(q_noise.mean(axis=2)**2, axis=0)
    out = {}
    for label, ix in [('all', np.arange(q.shape[1])), ('half0', np.arange(q.shape[1]//2)),
                      ('half1', np.arange(q.shape[1]//2, q.shape[1]))]:
        out[label] = dict(shape_error=float(shape[ix].mean()),
                         teacher_diagonal_shape_error=float(diagonal[ix].mean()),
                         contrast_trace_ratio=float(np.trace(c[ix], axis1=-2, axis2=-1).mean()/
                                                    np.trace(t[ix], axis1=-2, axis2=-1).mean()),
                         offset_energy_fraction=float(offset[ix].sum()/total[ix].sum()))
    return out, c, t


def main():
    started = time.monotonic()
    protocol_path = WORK/'diagnostic-protocol.json'
    p = json.loads(protocol_path.read_text())
    out = WORK/'diagnostic-v1'
    assert not out.exists(), 'no repeated/overwritten diagnostic'
    out.mkdir()
    assert digest(ROOT/p['source_path']) == p['source_sha256']
    assert digest(ROOT/'SO2/ding/model/template/ensemble_qac.py') == p['native_source_sha256']
    assert digest(p['dataset_path']) == p['dataset_sha256']
    for item in p['inputs']:
        assert digest(item['path']) == item['sha256'], item['path']
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = 'cuda'
    assert torch.cuda.is_available()
    torch.cuda.reset_peak_memory_stats()
    native = source_forward(ROOT/'SO2/ding/model/template/ensemble_qac.py')
    with h5py.File(p['dataset_path'], 'r') as f:
        valid = np.flatnonzero(~f['timeouts'][:].astype(bool))
        gen = np.random.default_rng(p['sample_seed'])
        # Selection order is preserved: two halves are fixed before values are seen.
        indices = gen.choice(valid, size=p['states'], replace=False)
        sorted_indices = np.sort(indices)
        observations = f['observations'][sorted_indices][np.searchsorted(sorted_indices, indices)]
    obs = torch.tensor(observations, device=device)
    rows, arrays = [], {'indices': indices, 'observations': observations}
    for seed in range(5):
        group = [x for x in p['inputs'] if x['seed'] == seed]
        paths = {x['method']: x['path'] for x in group}
        anchor_state = load(paths['anchor'])
        mu = actor_mean(anchor_state, obs, device)
        directions = torch.zeros(13, 6, device=device)
        for axis in range(6):
            directions[1+2*axis, axis] = p['radius']
            directions[2+2*axis, axis] = -p['radius']
        actions = (mu[:, None, :] + directions[None, :, :]).tanh()
        arrays[f'actions_s{seed}'] = actions.cpu().numpy()
        # Independent RNG, common panels across A1/T011. No training RNG loaded.
        mg = torch.Generator(device='cpu').manual_seed(p['mask_seed']+seed)
        panels = []
        for k in range(p['mask_samples_per_head']):
            panels.append([(torch.rand((2,1,256), generator=mg) >= .005).to(device)
                           for _ in range(3)])
        teacher = FrozenCritic(load(paths['10Q']), native, device)
        assert teacher.num_q == 10 and not teacher.use_layer_norm
        with torch.no_grad():
            tq = teacher.values(obs, actions).cpu().numpy().astype(np.float64)
        arrays[f'teacher_s{seed}'] = tq
        ct = covariance(contrasts(tq, p['radius']))
        # All 45 two-head subsets: no best-subset selection.
        subset_errors = []
        for a in range(10):
            for b in range(a+1,10):
                cs = covariance(contrasts(tq[[a,b]], p['radius']))
                subset_errors.append(np.linalg.norm(norm_cov(cs)-norm_cov(ct), axis=(1,2)))
        arrays[f'teacher_twohead_shape_errors_s{seed}'] = np.asarray(subset_errors)
        for method in ('A1', 'T011'):
            critic = FrozenCritic(load(paths[method]), native, device)
            assert critic.num_q == 2 and critic.use_layer_norm
            collected = []
            gradients = []
            with torch.no_grad():
                for masks in panels:
                    collected.append(critic.values(obs, actions, masks).cpu().numpy())
            for masks in panels:
                ac = actions[:, :1].detach().clone().requires_grad_(True)
                qc = critic.values(obs, ac, masks)
                gh = [torch.autograd.grad(qc[h].sum(), ac, retain_graph=(h == 0))[0][:,0]
                      for h in range(2)]
                gradients.append(torch.stack(gh).detach().cpu().numpy())
            arrays[f'{method}_action_gradients_s{seed}'] = np.stack(gradients)
            # Both heads have equal mass; mask panels are repeated Monte Carlo,
            # not independent trained members.
            q = np.stack(collected).astype(np.float64)
            arrays[f'{method}_s{seed}'] = q
            flat = q.reshape(-1, p['states'], 13)
            stats, c, _ = summarize(flat, tq, p['radius'])
            panel_stats = [summarize(x.reshape(-1,p['states'],13), tq, p['radius'])[0]
                           for x in np.split(q,2,axis=0)]
            within = np.mean([covariance(contrasts(q[:,h],p['radius'])) for h in range(2)],axis=0)
            between = covariance(contrasts(q.mean(axis=0),p['radius']))
            for half in ('all','half0','half1'):
                stats[half]['mask_panel_shape_error_spread'] = abs(
                    panel_stats[0][half]['shape_error']-panel_stats[1][half]['shape_error'])
            rows.append(dict(seed=seed, method=method, metrics=stats,
                             twohead_teacher_mean_shape_error=float(np.mean(subset_errors)),
                             within_head_trace=float(np.trace(within,axis1=-2,axis2=-1).mean()),
                             between_head_trace=float(np.trace(between,axis1=-2,axis2=-1).mean())))
            assert np.allclose(c,within+between,rtol=1e-8,atol=1e-8)
            # A common additive action-independent offset must disappear exactly
            # up to floating-point rounding in the contrast representation.
            assert np.allclose(contrasts(flat+123.0,p['radius']),contrasts(flat,p['radius']),atol=1e-8)
            assert torch.equal(critic._tensors['critic.fc0.W'].cpu(),load(paths[method])['critic.fc0.W'])
            print(json.dumps({'seed':seed,'method':method,'summary':stats['all']}),flush=True)
    arrays_path = out/'functions.npz'
    np.savez_compressed(arrays_path,**arrays)
    torch.cuda.synchronize()
    write(out/'summary.json',dict(status='completed', rows=rows,
          protocol_sha256=digest(protocol_path),source_sha256=digest(ROOT/p['source_path']),
          input_sha256={x['path']:x['sha256'] for x in p['inputs']},
          arrays_sha256=digest(arrays_path),environment_steps=0,optimizer_steps=0,
          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
          peak_reserved_bytes=torch.cuda.max_memory_reserved(),
          scientific_seconds=time.monotonic()-started,
          torch_version=torch.__version__,device=torch.cuda.get_device_name()))


if __name__ == '__main__':
    main()
