"""Unvalidated T020 D0/correctness/shadow worker. No environment or training CLI.

Only t020_control may launch this file, after source/protocol freeze and the
complete T019 resource barrier. Every import/load/test/save is inside its timer.
"""
import argparse
import copy
import hashlib
import json
import os
import random
import signal
import time
from pathlib import Path

import so2_compat
import h5py
import numpy as np
import torch
from torch.distributions import Independent, Normal
from t014_a1_common import make_policy, equivalent
from t014_a1_mask import paired_forward
from t020_actor import (CaptureNative, CrossMaskActor, ReplayMask, actor_terms,
                        freeze_parameters, masked_reference)
from t020_statistics import calibrate, panel_report


MODES = ("native", "square", "k2", "known", "cross")
COUNTS = {"critic": 0, "actor": 0}
END = float(os.environ.get("SO2_T020_MONOTONIC_DEADLINE", "0"))
PHASE_END = float("inf")


def guard():
    assert END > 0, "Launch only through frozen T020 control"
    if time.monotonic() > min(END-5, PHASE_END):
        raise TimeoutError("scientific worker save margin reached")


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    os.replace(temporary, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            guard()
            h.update(block)
    return h.hexdigest()


def array_hash(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state[0]); np.random.set_state(state[1])
    torch.set_rng_state(state[2]); torch.cuda.set_rng_state_all(state[3])


def policy_from_checkpoint(state):
    guard()
    policy, _ = make_policy("cuda", mode="matched")
    policy._model.load_state_dict(state["model"], strict=True)
    policy._target_model.load_state_dict(state["target_model"], strict=True)
    for key, optimizer in (("optimizer_q", policy._optimizer_q),
                           ("optimizer_policy", policy._optimizer_policy)):
        if key in state:
            optimizer.load_state_dict(copy.deepcopy(state[key]))
    policy._forward_learn_cnt = 0
    policy._a1_paired_updates = 0
    assert not policy._auto_alpha and policy._actor_update_freq == 10
    assert policy._actor_q_aggregation == "mean" and not policy._value_norm
    assert not policy._cfg.learn.only_value
    policy._model.train(); policy._target_model.train()
    return policy


def load_inputs(spec, out):
    """One prescribed offline-state panel; no environment/replay construction."""
    guard()
    assert spec["panel_kind"] == "offline_state_only_no_online_replay_snapshot"
    checkpoints = spec["checkpoints"]
    assert [(x["seed"], x["online_env_steps"]) for x in checkpoints] == [
        (0, 0), (0, 25000), (1, 0), (1, 25000)]
    states, receipts = [], []
    parent_hashes = {}
    for item in checkpoints:
        guard()
        path = Path(item["remote_path"])
        assert path.stat().st_size == item["expected_size"]
        assert digest(path) == item["expected_sha256_from_historical_seal"]
        state = torch.load(path, map_location="cpu", weights_only=False)
        assert {"model", "target_model", "optimizer_q", "optimizer_policy"} <= set(state)
        assert all(torch.isfinite(t).all() for key in ("model", "target_model")
                   for t in state[key].values())
        parent = item["eta1_parent_path"]
        if parent not in parent_hashes:
            parent_hashes[parent] = digest(parent)
        assert parent_hashes[parent] == item["eta1_parent_sha256"]
        if item["online_env_steps"] == 0:
            assert not state["optimizer_q"]["state"]
            assert not state["optimizer_policy"]["state"]
            original = torch.load(parent, map_location="cpu", weights_only=False)
            equivalent(state["model"], original["model"])
            equivalent(state["target_model"], original["target_model"])
            del original
        states.append(state)
        receipts.append({"seed": item["seed"], "step": item["online_env_steps"],
                         "checkpoint_sha256": item["expected_sha256_from_historical_seal"],
                         "eta1_parent_sha256": parent_hashes[parent]})
    assert digest(spec["dataset_path"]) == spec["dataset_sha256"]
    with h5py.File(spec["dataset_path"], "r") as source:
        n = len(source["observations"])
        valid = np.flatnonzero(~source["timeouts"][:n-1].astype(bool))
        assert len(valid) >= 2560
        generator = torch.Generator().manual_seed(spec["row_seed"])
        indices = valid[torch.randperm(len(valid), generator=generator)[:2560].numpy()]
        # h5py fancy indexing requires sorted indices; restore prescribed order.
        order = np.argsort(indices); inverse = np.argsort(order)
        sorted_rows = indices[order]
        panel = {"obs": source["observations"][sorted_rows][inverse],
                 "action": source["actions"][sorted_rows][inverse],
                 "reward": source["rewards"][sorted_rows][inverse],
                 "done": source["terminals"][sorted_rows][inverse],
                 "next_obs": source["observations"][sorted_rows+1][inverse]}
    assert panel["obs"].shape == (2560, 17) and panel["action"].shape == (2560, 6)
    assert all(np.isfinite(x).all() for x in panel.values())
    subset = torch.randperm(2560, generator=torch.Generator().manual_seed(
        spec["subset_seed"]))[:256]
    epsilon = torch.randn((2560, 6), generator=torch.Generator().manual_seed(
        spec["actor_epsilon_seed"]))
    arrays = dict(panel, rows=indices, auxiliary_indices=subset.numpy(),
                  actor_epsilon=epsilon.numpy())
    manifest = {"panel_kind": spec["panel_kind"], "checkpoints": receipts,
                "dataset_sha256": spec["dataset_sha256"],
                "arrays": {k: {"sha256": array_hash(v), "shape": list(v.shape),
                                "dtype": str(v.dtype)} for k, v in arrays.items()},
                "generated_before_gradient_collection": True}
    np.savez(out/"fixed-input-arrays.npz", **arrays)
    save(out/"input-identity.json", manifest)
    batch = [{key: torch.as_tensor(value[i], dtype=torch.float32)
              for key, value in panel.items()} for i in range(2560)]
    device_panel = {k: torch.as_tensor(v, dtype=torch.float32, device="cuda")
                    for k, v in panel.items()}
    return states, batch, device_panel, subset.to("cuda"), epsilon.to("cuda")


def actor_graph(policy, panel, epsilon, seed):
    torch.manual_seed(seed)
    model, critic = policy._model, policy._model.critic
    mu, sigma = model(panel["obs"], mode="compute_actor")["logit"]
    latent = mu+sigma*epsilon
    action = latent.tanh()
    log_probability = Independent(Normal(mu, sigma), 1).log_prob(latent)
    log_probability = log_probability-torch.log(1-action.square()+1e-6).sum(-1)
    capture = CaptureNative()
    assert getattr(critic, "_dropout_pair_context", None) is None
    critic._dropout_pair_context = capture
    try:
        q = model.compute_critic({"obs": panel["obs"], "action": action})["q_value"]
    finally:
        critic._dropout_pair_context = None
    base = (policy._alpha*log_probability-q.mean(0)).mean()
    return action, q, capture, base


def flatten_gradient(loss, parameters):
    guard()
    values = torch.autograd.grad(loss, parameters, retain_graph=True)
    vector = torch.cat([x.detach().reshape(-1) for x in values]).double().cpu()
    assert torch.isfinite(vector).all()
    return vector


def collect(policy, panel, indices, epsilon, spec, panel_index, tau_squared):
    """Unit-strength H and paired G0 on the same action/g graph for all controls."""
    parameters = tuple(policy._model.actor.parameters())
    all_banks = {name: [] for name in MODES}
    all_banks.update(min=[], BC=[])
    for bank in range(2):
        values = {name: [] for name in all_banks}
        for repeat in range(16):
            guard()
            gseed = spec["g_mask_seed"]+10000*panel_index+100*bank+repeat
            wseed = spec["w_mask_seed"]+10000*panel_index+100*bank+repeat
            with torch.random.fork_rng(devices=[0]), freeze_parameters(policy._model.critic):
                action, q, capture, base = actor_graph(policy, panel, epsilon, gseed)
                g0 = flatten_gradient(base, parameters)
                values["native"].append(g0)
                values["min"].append(flatten_gradient(
                    base+q.mean(0).mean()-q.min(0).values.mean(), parameters))
                values["BC"].append(flatten_gradient(
                    (action-panel["action"]).square().mean(), parameters))
                for mode in ("square", "k2", "known", "cross"):
                    terms = actor_terms(
                        policy._model.critic, policy._model.compute_critic,
                        panel["obs"], action, panel["action"], q, capture,
                        mode, 1.0, tau_squared,
                        torch.Generator().manual_seed(spec["subset_seed"]),
                        torch.Generator(device="cuda").manual_seed(wseed),
                        fixed_indices=indices)
                    assert terms.d_w is None or not terms.d_w.requires_grad
                    h = flatten_gradient(terms.auxiliary, parameters)
                    # Save total unit G, then subtract this paired G0 for calibration.
                    values[mode].append(g0+h)
            for name in all_banks:
                assert len(values[name]) == repeat+1
        for name in all_banks:
            all_banks[name].append(torch.stack(values[name]))
    return all_banks


def calibrate_tau(policy, panel, epsilon, spec):
    """Reuse the same 32 calibration g-mask seeds, not an extra selected bank."""
    variances = []
    with torch.no_grad(), torch.random.fork_rng(devices=[0]):
        for bank in range(2):
            for repeat in range(16):
                guard()
                action, q, capture, _ = actor_graph(
                    policy, panel, epsilon, spec["g_mask_seed"]+100*bank+repeat)
                qb = masked_reference(policy._model.critic, policy._model.compute_critic,
                    panel["obs"], panel["action"], ReplayMask(capture.masks))
                contrast = (q[0]-qb[0])-(q[1]-qb[1])
                variances.append(float((contrast.square()/2).mean()))
    return float(np.mean(variances))


def cycle(policy, adapter, batch):
    guard()
    assert COUNTS["critic"]+10 <= 1000 and COUNTS["actor"]+1 <= 100
    before = policy._forward_learn_cnt
    result = None
    for _ in range(10):
        COUNTS["critic"] += 1
        actor_due = (policy._forward_learn_cnt+1) % 10 == 0
        COUNTS["actor"] += int(actor_due)
        result = adapter(policy, batch) if adapter else paired_forward(policy, batch)
    assert policy._forward_learn_cnt == before+10
    return result


def correctness(state, batch, panel, indices, epsilon, spec, out):
    # Compare F.dropout with captured native dropout before relying on replay masks.
    import torch.nn.functional as functional
    for device in ("cpu", "cuda"):
        guard()
        original = torch.ones((2, 256, 256), device=device, requires_grad=True)
        capture_input = original.detach().clone().requires_grad_(True)
        saved = rng_state()
        a = functional.dropout(original, .005, True)
        ga, = torch.autograd.grad(a.square().sum(), original)
        after = rng_state(); restore_rng(saved)
        b, mask = torch.ops.aten.native_dropout(capture_input, .005, True)
        gb, = torch.autograd.grad(b.square().sum(), capture_input)
        assert torch.equal(a, b) and torch.equal(ga, gb)
        equivalent(after, rng_state())
        assert torch.equal(b, torch.ops.aten.native_dropout_backward(
            capture_input, mask, 1/.995))
        restore_rng(saved)
        with torch.no_grad():
            no_grad_output, _ = torch.ops.aten.native_dropout(capture_input, .005, True)
        assert torch.equal(b, no_grad_output)
        equivalent(after, rng_state())
    left, right = policy_from_checkpoint(state), policy_from_checkpoint(state)
    initial_rng = rng_state()
    native = cycle(left, None, batch)
    native_rng = rng_state(); restore_rng(initial_rng)
    zero = CrossMaskActor("cross", 0, spec["subset_seed"], spec["w_mask_seed"])
    observed = cycle(right, zero, batch)
    equivalent(native, observed); equivalent(native_rng, rng_state())
    equivalent(left.learn_mode.state_dict(), right.learn_mode.state_dict())
    equivalent([p.grad for p in left._model.parameters()],
               [p.grad for p in right._model.parameters()])
    assert left._forward_learn_cnt == right._forward_learn_cnt == 10
    assert left._a1_paired_updates == right._a1_paired_updates == 10
    assert zero.subset_generator is None and not zero.weight_generators
    del left, right
    policy = policy_from_checkpoint(state)
    critic = policy._model.critic
    for parameter in critic.parameters():
        parameter.grad = torch.ones_like(parameter)
    prior_grads = [p.grad.clone() for p in critic.parameters()]
    with torch.random.fork_rng(devices=[0]), freeze_parameters(critic):
        action, q, capture, base = actor_graph(policy, panel, epsilon, spec["g_mask_seed"])
        native_rng = rng_state()
        terms = actor_terms(critic, policy._model.compute_critic, panel["obs"],
            action, panel["action"], q, capture, "cross", 1.0, None,
            torch.Generator().manual_seed(spec["subset_seed"]),
            torch.Generator(device="cuda").manual_seed(spec["w_mask_seed"]),
            fixed_indices=indices)
        equivalent(native_rng, rng_state())
        assert terms.d_g.requires_grad and not terms.d_w.requires_grad
        assert any(not torch.equal(capture.masks[k][:, indices], terms.w_masks[k])
                   for k in capture.masks)
        terms.auxiliary.backward()
        assert all(torch.equal(p.grad, old) for p, old in zip(critic.parameters(), prior_grads))
    assert all(p.requires_grad for p in critic.parameters())
    del policy
    policy = policy_from_checkpoint(state)
    nonzero = CrossMaskActor("cross", .1, spec["subset_seed"], spec["w_mask_seed"])
    cycle(policy, nonzero, batch)
    assert nonzero.last_report["g_layers"] == nonzero.last_report["w_layers"] == 3
    assert policy._forward_learn_cnt == policy._a1_paired_updates == 10
    assert all(p.requires_grad for p in policy._model.critic.parameters())
    save(out/"correctness.json", {"passed": True, "lambda0_exact": True,
        "native_rng_unchanged_by_aux": True, "critic_prior_TD_grad_preserved": True,
        "native_dropout_CPU_CUDA_output_gradient_RNG_exact": True,
        "optimizer_calls_cumulative": dict(COUNTS)})


def shadow(state, batch, calibration, tau_squared, spec, out):
    guard()
    policies = {name: policy_from_checkpoint(state) for name in MODES}
    adapters = {name: None if name == "native" else CrossMaskActor(
        name, calibration[name]["strength"], spec["subset_seed"],
        spec["w_mask_seed"], tau_squared=tau_squared) for name in MODES}
    orders = [list(MODES[i:]+MODES[:i]) for i in range(5)]
    orders += [list(reversed(x)) for x in orders]
    records = []
    for block, order in enumerate(orders):
        for name in order:
            guard()
            torch.manual_seed(spec["shadow_seed"]+block)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            cycle(policies[name], adapters[name], batch)
            torch.cuda.synchronize()
            records.append({"block": block, "method": name, "warmup": block == 0,
                "seconds": time.monotonic()-started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
            save(out/"shadow-progress.json", {"records": records, "counts": dict(COUNTS)})
    ratios = []
    for block in range(1, 10):
        values = {x["method"]: x["seconds"] for x in records if x["block"] == block}
        ratios.append(values["cross"]/values["native"])
    cost = "within_reference" if max(ratios) <= 1.10 else (
        "above_reference" if min(ratios) > 1.10 else "inconclusive_timing_variation")
    result = {"status": "completed", "records": records,
        "paired_cross_native_ratios": ratios, "cost_classification": cost,
        "shadow_cycles": 50, "counts_including_correctness": dict(COUNTS),
        "peak_scope": "process peak including all five resident identical-start clones",
        "environment_interactions": 0, "reusable_training_artifact": False}
    save(out/"shadow.json", result)
    return result


def run(protocol_path, out):
    global PHASE_END
    guard(); torch.set_num_threads(2)
    protocol = json.loads(protocol_path.read_text()); spec = protocol["inputs"]
    phases = {}; overall = {"status": "partial", "interpretation": "not_run",
        "proxy_scalar_expectation": "2R for cross proxy at fixed critic",
        "cross_autodiff_gradient_objective": "R; stopped coefficient, not derivative of the proxy expectation",
        "counter_accounting": "Attempted full learner update upper bound; exact on completed paths",
        "completed_panels": [], "shadow": {"status": "not_run"}}
    started = time.monotonic()
    try:
        phase = time.monotonic(); PHASE_END = phase+90
        states, batch, panel, indices, epsilon = load_inputs(spec, out)
        phases["load_seconds"] = time.monotonic()-phase
        assert phases["load_seconds"] <= 90
        policy = policy_from_checkpoint(states[0])
        order = [{"name": k, "shape": list(v.shape), "dtype": str(v.dtype)}
                 for k, v in policy._model.actor.named_parameters()]
        save(out/"parameter-order.json", {"order": order,
            "sha256": hashlib.sha256(json.dumps(order, sort_keys=True).encode()).hexdigest(),
            "written_before_gradients": True})
        phase = time.monotonic(); PHASE_END = phase+120
        correctness(states[0], batch, panel, indices, epsilon, spec, out)
        phases["correctness_seconds"] = time.monotonic()-phase
        assert phases["correctness_seconds"] <= 120
        phase = time.monotonic(); PHASE_END = phase+240
        tau = calibrate_tau(policy, panel, epsilon, spec)
        if tau <= 1e-12:
            phases["D0_seconds"] = time.monotonic()-phase
            overall.update(status="completed_inconclusive", interpretation="tau_near_zero_no_amplification")
            return
        unit = collect(policy, panel, indices, epsilon, spec, 0, tau)
        calibration = {name: calibrate(unit["native"],
            [a-b for a, b in zip(unit[name], unit["native"])],
            1 if name == "known" else 10) for name in MODES if name != "native"}
        save(out/"calibration.json", {"checkpoint": spec["checkpoints"][0],
            "tau_squared": tau, "coefficients": calibration,
            "paired_G_minus_same_G0": True, "frozen_before_other_panels": True})
        torch.save(unit, out/"calibration-unit-gradients.pt")
        if not all(x["eligible"] for x in calibration.values()):
            phases["D0_seconds"] = time.monotonic()-phase
            overall.update(status="completed_inconclusive", interpretation="finite_MC_calibration_weak_or_unstable")
            return
        reports = []
        for index, state in enumerate(states):
            guard()
            if index:
                del policy
                policy = policy_from_checkpoint(state)
                current_order = [{"name": k, "shape": list(v.shape), "dtype": str(v.dtype)}
                                 for k, v in policy._model.actor.named_parameters()]
                assert current_order == order
                unit = collect(policy, panel, indices, epsilon, spec, index, tau)
            total = {"native": unit["native"]}
            for name, coefficient in calibration.items():
                total[name] = [g0+coefficient["strength"]*(g-g0)
                               for g, g0 in zip(unit[name], unit["native"])]
            report = panel_report(total, unit["min"], unit["BC"])
            report.update(seed=spec["checkpoints"][index]["seed"],
                          step=spec["checkpoints"][index]["online_env_steps"])
            reports.append(report)
            overall["completed_panels"] = [[x["seed"], x["step"]] for x in reports]
            torch.save(dict(total=total, min=unit["min"], BC=unit["BC"]),
                       out/("panel-%d-gradients.pt" % index))
            save(out/"diagnostic-progress.json", {"panels": reports})
            if time.monotonic()-phase > 240:
                raise TimeoutError("D0 sub-budget reached")
        phases["D0_seconds"] = time.monotonic()-phase
        at25k = [x for x in reports if x["step"] == 25000]
        distinct = len(at25k) == 2 and all(x["panel_direction_gate"] for x in at25k)
        overall.update(status="completed", panels=reports,
            interpretation="direction_distinguished_under_fixed_MC_rule" if distinct else
            "not_distinguished_under_this_panel_and_MC_budget_no_equivalence_claim")
        del policy
        if END-time.monotonic() > 100:
            phase = time.monotonic(); PHASE_END = phase+90
            overall["shadow"] = shadow(states[0], batch, calibration, tau, spec, out)
            phases["shadow_seconds"] = time.monotonic()-phase
            assert phases["shadow_seconds"] <= 90
        else:
            overall["shadow"] = {"status": "not_completed_budget_priority_to_D0"}
    except BaseException as error:
        overall.update(status="partial", error=repr(error), interpretation="incomplete_no_effectiveness_claim")
        raise
    finally:
        overall.update(phases=phases, worker_wall_seconds=time.monotonic()-started,
                       optimizer_counts=dict(COUNTS), environment_interactions=0)
        overall["panels_not_completed"] = [[x["seed"], x["online_env_steps"]]
            for x in spec["checkpoints"] if [x["seed"], x["online_env_steps"]]
            not in overall["completed_panels"]]
        save(out/"result.json", overall)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    assert arguments.output.is_dir() and not (arguments.output/"result.json").exists()
    def interrupted(signum, frame):
        raise TimeoutError("finite worker deadline signal %d" % signum)
    signal.signal(signal.SIGTERM, interrupted)
    run(arguments.protocol, arguments.output)
