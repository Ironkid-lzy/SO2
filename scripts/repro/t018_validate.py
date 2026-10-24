"""Bounded synthetic semantic tests. No environment or throughput benchmark."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from t010_edac import OfflineEDAC, make_model, seed_all, rng_state, restore_rng, tensor_hash
from t018_edac import ParameterizedEDAC


def same(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: same(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): same(x, y)
    else:
        assert a == b, (a, b)


def loss_grads(trainer, batch, initial_rng):
    restore_rng(initial_rng)
    losses = trainer.losses({k: v.clone() for k, v in batch.items()})
    after_rng = rng_state()
    params = list(trainer.model.parameters()) + [trainer.log_alpha]
    grads = {}
    for key in losses:
        if losses[key].requires_grad:
            grads[key] = torch.autograd.grad(losses[key], params, retain_graph=True, allow_unused=True)
        else:
            grads[key] = tuple(None for _ in params)
    return {k: v.detach().clone() for k, v in losses.items()}, grads, after_rng


def check_device(device):
    seed_all(0, cuda=torch.cuda.is_available())
    native = OfflineEDAC(make_model(device))
    initial = copy.deepcopy(native.snapshot({"task": "T018-validation"}))
    default = ParameterizedEDAC(copy.deepcopy(native.model))
    zero = ParameterizedEDAC(copy.deepcopy(native.model), es_coefficient=0)
    default.restore(initial); zero.restore(initial)
    gen = torch.Generator(device="cpu").manual_seed(481)
    batch = {k: torch.randn(256, n, generator=gen).to(device) for k, n in
             (("observations", 17), ("next_observations", 17), ("actions", 6), ("rewards", 1))}
    batch["actions"] = batch["actions"].tanh()
    batch["terminals"] = torch.zeros(256, 1, device=device)
    lrng = copy.deepcopy(initial["rng"])
    ref, rg, rrng = loss_grads(native, batch, lrng)
    got, gg, grng = loss_grads(default, batch, lrng)
    same(ref, got); same(rg, gg); same(rrng, grng)
    zerol, zg, zrng = loss_grads(zero, batch, lrng)
    for k in ("alpha", "actor", "td"):
        same(ref[k], zerol[k]); same(rg[k], zg[k])
    same(rrng, zrng)
    assert zerol["es"].item() == 0 and not zerol["es"].requires_grad
    same(zerol["critic"], zerol["td"])
    same(zg["critic"], zg["td"])
    restore_rng(lrng)
    with patch("torch.autograd.grad", side_effect=AssertionError("ES action differentiation executed")):
        zero.losses({k: v.clone() for k, v in batch.items()})
    results = []
    for trainer in (native, default, zero):
        trainer.restore(copy.deepcopy(initial))
        values = trainer.update({k: v.clone() for k, v in batch.items()})
        results.append((values, copy.deepcopy(trainer.snapshot({"task": "T018-validation"}))))
    same(results[0], results[1])
    for key in ("optimizer_policy", "optimizer_alpha", "log_alpha", "rng"):
        same(results[0][1][key], results[2][1][key])
    for key in results[0][1]["model"]:
        if key.startswith("actor."):
            same(results[0][1]["model"][key], results[2][1]["model"][key])
    assert results[2][1]["updates"] == 1
    return {"device": device, "passed": True,
            "default_loss_parameter_grad_update_optimizer_rng_exact": True,
            "eta0_common_loss_grad_and_rng_exact": True,
            "eta0_no_autograd_grad_in_losses": True,
            "eta0_actor_alpha_single_update_exact": True,
            "eta0_critic_is_td": True,
            "native_ES": ref["es"].item(),
            "formal_training_updates": 0, "validation_optimizer_updates": 3,
            "environment_interactions": 0, "speed_measurement": False}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", required=True)
    args = p.parse_args(); out = Path(args.out)
    assert not out.exists(), "immutable test receipt already exists"
    torch.set_num_threads(2)
    assert torch.cuda.is_available(), "CPU+CUDA validation required"
    reports = [check_device(d) for d in ("cpu", "cuda")]
    # Audit archived untrained initialization; mismatch is evidence, not permission to retune.
    source = Path("/home/lzy/Projects/SO2_t010/_so2_work/runs/EXP-025/EXP-025-s0-attempt1/ckpt/update_0.pth")
    h = hashlib.sha256(source.read_bytes()).hexdigest()
    archive = torch.load(source, map_location="cpu", weights_only=False)
    assert archive["updates"] == 0 and archive["train_env_steps"] == 0
    assert all(not archive[k]["state"] for k in ("optimizer_policy", "optimizer_q", "optimizer_alpha"))
    seed_all(0, cuda=True); fresh = OfflineEDAC(make_model("cuda"))
    fresh_state = {k: v.cpu() for k, v in fresh.model.state_dict().items()}
    fresh_matches = all(torch.equal(v, archive["model"][k]) for k, v in fresh_state.items())
    same(archive["model"], archive["target_model"])
    init = {"update0_sha256": h, "updates": 0, "optimizer_states_empty": True,
            "model_equals_target": True, "fresh_seed0_model_equals_archive": fresh_matches,
            "archived_model_tensor_hash": tensor_hash(archive["model"]),
            "fresh_model_tensor_hash": tensor_hash(fresh_state),
            "initial_eval_recomputed": False, "historical_batch_sequence_independently_verified": False}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"passed": True, "devices": reports, "initialization": init}, indent=2)+"\n")
    print(out)


if __name__ == "__main__":
    main()
