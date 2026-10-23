"""One preregistered CPU-only source-dropout screen; no environment imports."""
import argparse
import copy
import hashlib
import json
import os
import pickle
import platform
import resource
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

START = time.monotonic()
STOP = False
ARMS = 6
HEADS = 2


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def rng(seed, case, stream):
    return np.random.default_rng([20261024, seed, case, stream])


def reference(protocol):
    p0 = np.array(protocol['mdp']['P_next_0']).reshape(4)
    transition = np.stack([p0 / 2, p0 / 2, (1-p0) / 2, (1-p0) / 2], axis=1)
    reward = np.array(protocol['mdp']['reward']).reshape(4)
    gamma = protocol['mdp']['gamma']
    q = np.linalg.solve(np.eye(4) - gamma * transition, reward)
    return q, transition, reward


def dataset(protocol, seed, case):
    canonical = 0 if case == 0 else 1
    config = protocol['data']['cases'][canonical]
    gen = rng(seed, canonical, 1)
    n0, n1 = protocol['data']['origin_A_rows'], protocol['data']['origin_B_rows']
    origin = np.repeat([0, 1], [n0, n1])
    p_state0 = np.where(origin == 0, config['origin_A_state_0'], config['origin_B_state_0'])
    state = (gen.random(len(origin)) >= p_state0).astype(np.int64)
    action = (gen.random(len(origin)) >= .5).astype(np.int64)
    p_next0 = np.array(protocol['mdp']['P_next_0'])[state, action]
    next_state = (gen.random(len(origin)) >= p_next0).astype(np.int64)
    reward = np.array(protocol['mdp']['reward'])[state, action]
    return dict(row_id=np.arange(len(origin)), origin=origin, state=state,
                action=action, next_state=next_state, reward=reward)


def features(state, action):
    return torch.from_numpy(np.stack([2*state-1, 2*action-1], axis=-1).astype(np.float64))


def source_labels(protocol, case, origins):
    return (origins != protocol['data']['cases'][case]['offline_origin']).astype(np.int64)


def probabilities(protocol, labels):
    return np.stack([np.where(labels == 0, arm['p_off'], arm['p_on']) for arm in protocol['arms']])


class Critics(nn.Module):
    def __init__(self, protocol, seed):
        super().__init__()
        dims = [2] + protocol['network']['hidden_widths'] + [1]
        gen = rng(seed, 0, 2)
        self.weights, self.biases = nn.ParameterList(), nn.ParameterList()
        self.scales, self.offsets = nn.ParameterList(), nn.ParameterList()
        for layer, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
            scale = .01 if layer == len(dims)-2 else 1/np.sqrt(din)
            weight = gen.normal(size=(HEADS, din, dout))*scale
            self.weights.append(nn.Parameter(torch.from_numpy(np.repeat(weight[None], ARMS, axis=0))))
            self.biases.append(nn.Parameter(torch.zeros(ARMS, HEADS, 1, dout, dtype=torch.float64)))
            if layer < len(dims)-2:
                self.scales.append(nn.Parameter(torch.ones(ARMS, HEADS, 1, dout, dtype=torch.float64)))
                self.offsets.append(nn.Parameter(torch.zeros(ARMS, HEADS, 1, dout, dtype=torch.float64)))

    def forward(self, x, p=None, gen=None):
        h = x[None, None].expand(ARMS, HEADS, -1, -1)
        for layer, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            h = torch.matmul(h, weight) + bias
            if layer < len(self.weights)-1:
                if p is not None:
                    uniform = torch.from_numpy(gen.random((1, HEADS, len(x), h.shape[-1])))
                    probability = torch.from_numpy(p)[:, None, :, None]
                    h = h * (uniform >= probability) / (1-probability)
                h = F.layer_norm(h, (h.shape[-1],), eps=1e-5)
                h = F.relu(h*self.scales[layer] + self.offsets[layer])
        return h.squeeze(-1)


def global_rng_hash():
    return hashlib.sha256(pickle.dumps(np.random.get_state()) + torch.get_rng_state().numpy().tobytes()).hexdigest()


def check(protocol, output):
    started = time.monotonic()
    before = global_rng_hash()
    q, transition, reward = reference(protocol)
    assert np.max(abs(q-np.array([1.09, -.09, -1.99, -1.01]))) < 1e-12
    assert np.max(abs(q - reward - protocol['mdp']['gamma']*transition@q)) < 1e-12
    d1, d2 = dataset(protocol, 0, 1), dataset(protocol, 0, 2)
    assert all(np.array_equal(d1[k], d2[k]) for k in d1)
    idx = rng(0, 1, 3).permutation(len(d1['origin']))[:64]
    labels1 = source_labels(protocol, 1, d1['origin'][idx])
    labels2 = source_labels(protocol, 2, d2['origin'][idx])
    p1, p2 = probabilities(protocol, labels1), probabilities(protocol, labels2)
    assert np.array_equal(labels1, 1-labels2)
    assert np.array_equal(p1[0], p2[1]) and np.array_equal(p1[1], p2[0])
    for row, label in zip(idx, labels1):
        assert label == int(d1['origin'][row] != 0)
    rates = []
    for probability in [0, .001, .005, .009]:
        draws = rng(9, 0, 10).random(200000)
        observed = float(np.mean(draws < probability))
        bound = 6*np.sqrt(probability*(1-probability)/len(draws))+1e-4
        assert abs(observed-probability) <= bound
        rates.append(dict(p=probability, observed=observed, tolerance=bound))
    model = Critics(protocol, 0)
    assert all(torch.equal(param[0], param[arm]) for param in model.parameters() for arm in range(1, ARMS))
    x = features(d1['state'][idx], d1['action'][idx])
    p = np.full((ARMS, len(x)), .005)
    loss = model(x, p, rng(0, 1, 4)).square().mean((1, 2)).sum()
    loss.backward()
    gradients = [param.grad[0].clone() for param in model.parameters()]
    model.zero_grad(set_to_none=True)
    model(x, p, rng(0, 1, 4))[0].square().mean().backward()
    assert all(torch.allclose(old, param.grad[0], atol=1e-12, rtol=1e-12)
               for old, param in zip(gradients, model.parameters()))
    # Recreating private streams makes checking independent of training draws.
    assert np.array_equal(rng(1, 0, 3).integers(100, size=20), rng(1, 0, 3).integers(100, size=20))
    assert before == global_rng_hash()
    assert all(param.device.type == 'cpu' for param in model.parameters())
    record = dict(status='passed', seconds=time.monotonic()-started,
                  exact_Q=q.tolist(), bellman_residual=float(np.max(abs(q-reward-protocol['mdp']['gamma']*transition@q))),
                  masks=rates, global_rng_unchanged=True, optimizer_updates=0,
                  data_label_exchange=True, arm_gradient_equivalence=True,
                  protocol_sha256=digest(args.protocol), source_sha256=digest(__file__))
    assert record['seconds'] < protocol['budget']['semantic_check_seconds_cap']
    write(output, record)
    print(json.dumps(record), flush=True)


def evaluate(model, protocol, case, q):
    states, actions = np.repeat([0, 1], 2), np.tile([0, 1], 2)
    with torch.no_grad():
        heads = model(features(states, actions)).numpy()
    pred = heads.mean(axis=1)
    squared = (pred-q[None])**2
    canonical = 0 if case == 0 else 1
    cfg = protocol['data']['cases'][canonical]
    a, b = cfg['origin_A_state_0'], cfg['origin_B_state_0']
    origin_weights = np.array([[a/2,a/2,(1-a)/2,(1-a)/2], [b/2,b/2,(1-b)/2,(1-b)/2]])
    source_weights = origin_weights if case != 2 else origin_weights[::-1]
    gaps = pred[:, [1, 3]]-pred[:, [0, 2]]
    true_gaps = q[[1, 3]]-q[[0, 2]]
    return dict(mse=squared.mean(axis=1).tolist(), source_mse=(squared@source_weights.T).tolist(),
                wrong_action_fraction=(gaps*true_gaps[None] <= 0).mean(axis=1).tolist(),
                head_mse=((heads-q[None,None])**2).mean(axis=(1,2)).tolist(),
                disagreement=((heads[:,0]-heads[:,1])**2).mean(axis=1).tolist(),
                predicted_Q=pred.tolist())


def fit(protocol, output):
    global STOP
    output.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGTERM, lambda *_: globals().__setitem__('STOP', True))
    signal.signal(signal.SIGINT, lambda *_: globals().__setitem__('STOP', True))
    q, _, _ = reference(protocol)
    order = [(seed, case) for seed in protocol['data']['seed_blocks'] for case in range(3)]
    order = [order[i] for i in rng(0, 0, 99).permutation(len(order))]
    before = global_rng_hash()
    meta = dict(status='running', start_unix=time.time(), pid=os.getpid(),
                protocol_sha256=digest(args.protocol), source_sha256=digest(__file__),
                git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                device='cpu', threads=torch.get_num_threads(), order=order,
                implementation_or_semantic_updates=0, environment_calls=0, GPU_jobs=0,
                optimizer_calls=0, logical_arm_updates=0, completed_blocks=0,
                synthetic_data_rows=0, replay_draws=0, forward_calls=0)
    write(output/'run_meta.json', meta)
    with (output/'rows.jsonl').open('x') as rows:
        for seed, case in order:
            canonical = 0 if case == 0 else 1
            pool = dataset(protocol, seed, case)
            datafile = output/f'inputs-s{seed}-c{canonical}.npz'
            if not datafile.exists():
                np.savez(datafile, **pool)
                meta['synthetic_data_rows'] += len(pool['row_id'])
            model = Critics(protocol, seed)
            target = copy.deepcopy(model).requires_grad_(False)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9,.999), eps=1e-8)
            gd, ga = rng(seed, canonical, 3), rng(seed, canonical, 4)
            gm, gt = rng(seed, canonical, 5), rng(seed, canonical, 6)
            completed = False
            block_started = time.monotonic()
            for step in range(protocol['updates']+1):
                if step in protocol['eval_updates']:
                    record = dict(seed=seed, case=case, update=step, **evaluate(model, protocol, case, q))
                    rows.write(json.dumps(record, allow_nan=False)+'\n'); rows.flush()
                    meta['forward_calls'] += 1
                    write(output/'run_meta.json', meta)
                if step == protocol['updates']:
                    completed = True
                    break
                if STOP or time.monotonic()-START >= protocol['budget']['wall_soft_stop_seconds']:
                    break
                n0 = protocol['data']['origin_A_rows']
                idx = np.r_[gd.integers(n0, size=32), n0+gd.integers(protocol['data']['origin_B_rows'], size=32)]
                idx = idx[gd.permutation(64)]
                labels = source_labels(protocol, case, pool['origin'][idx])
                x = features(pool['state'][idx], pool['action'][idx])
                next_action = (ga.random(64) >= .5).astype(np.int64)
                nx = features(pool['next_state'][idx], next_action)
                with torch.no_grad():
                    nextq = target(nx, np.full((ARMS,64),protocol['target_drop_p']), gt)
                    # No exact-Q access: only empirical reward, next state and sampled action.
                    y = torch.from_numpy(pool['reward'][idx])[None,None] + protocol['mdp']['gamma']*nextq
                prediction = model(x, probabilities(protocol, labels), gm)
                per_arm_loss = (prediction-y).square().mean((1,2))
                if not torch.isfinite(per_arm_loss).all():
                    raise RuntimeError('nonfinite loss; no restart authorized')
                optimizer.zero_grad(set_to_none=True)
                per_arm_loss.sum().backward()
                optimizer.step()
                with torch.no_grad():
                    for dest, src in zip(target.parameters(), model.parameters()):
                        dest.lerp_(src, protocol['network']['target_polyak_theta'])
                meta['optimizer_calls'] += 1; meta['logical_arm_updates'] += ARMS
                meta['replay_draws'] += 64; meta['forward_calls'] += 2
            if completed:
                meta['completed_blocks'] += 1
                print(json.dumps(dict(block=meta['completed_blocks'],seed=seed,case=case,
                                      block_seconds=time.monotonic()-block_started)),flush=True)
            else:
                break
    meta.update(status='completed' if meta['completed_blocks'] == len(order) else 'partial',
                end_unix=time.time(), process_seconds=time.monotonic()-START,
                cpu_seconds=time.process_time(), maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                global_rng_unchanged=before == global_rng_hash())
    assert meta['global_rng_unchanged']
    write(output/'run_meta.json', meta)
    manifest = {p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()}
    write(output/'seal.json',dict(status=meta['status'],sha256=manifest))
    print(json.dumps(meta),flush=True)
    return 0 if meta['status'] == 'completed' else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol',required=True)
    parser.add_argument('--check',type=Path)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    protocol = json.loads(Path(args.protocol).read_text())
    if args.check:
        check(protocol,args.check)
    else:
        raise SystemExit(fit(protocol,args.output))
