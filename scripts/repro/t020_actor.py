"""T020 diagnostic-only actor adapter; no training entry point.

Unvalidated until the bounded T020 correctness job runs. In particular, native
dropout capture must match the original F.dropout RNG/output/gradient path.
All TD, target, PVU, entropy, optimizer counts and EMA remain in frozen A1.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import math
import torch
from t014_a1_mask import paired_forward


class CaptureNative:
    """Expose row/head/layer masks from the single existing actor Q graph."""
    def __init__(self):
        self.masks = {}

    def apply(self, h, layer, p, training):
        assert training and p == .005 and h.ndim == 3 and h.shape[0] == 2
        assert layer not in self.masks
        output, mask = torch.ops.aten.native_dropout(h, p, True)
        self.masks[layer] = mask
        return output


class ReplayMask:
    def __init__(self, masks, indices=None):
        self.masks, self.indices = masks, indices

    def apply(self, h, layer, p, training):
        assert training and p == .005 and not torch.is_grad_enabled()
        mask = self.masks[layer]
        if self.indices is not None:
            mask = mask.index_select(1, self.indices)
        assert mask.shape == h.shape and mask.device == h.device
        return torch.ops.aten.native_dropout_backward(h, mask, 1.0 / (1.0-p))


class ExplicitAuxMask:
    """A separate generator; no auxiliary draws from native CPU/CUDA RNG."""
    def __init__(self, generator):
        self.generator = generator
        self.masks = {}

    def apply(self, h, layer, p, training):
        assert training and p == .005 and not torch.is_grad_enabled()
        assert layer not in self.masks
        mask = torch.bernoulli(
            torch.full_like(h, 1.0-p), generator=self.generator
        ).to(torch.bool)
        self.masks[layer] = mask
        return torch.ops.aten.native_dropout_backward(h, mask, 1.0 / (1.0-p))


@contextmanager
def freeze_parameters(module):
    """Preserve TD gradients; actor forwards cannot add parameter gradients."""
    parameters = list(module.parameters())
    flags = [p.requires_grad for p in parameters]
    try:
        for p in parameters:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(parameters, flags):
            p.requires_grad_(flag)


def masked_reference(critic, forward, obs, action, context):
    assert getattr(critic, "_dropout_pair_context", None) is None
    critic._dropout_pair_context = context
    try:
        with torch.no_grad():
            return forward({"obs": obs, "action": action.detach()})["q_value"]
    finally:
        critic._dropout_pair_context = None


@dataclass
class ActorTerms:
    auxiliary: torch.Tensor
    d_g: torch.Tensor
    d_w: object
    indices: object
    g_masks: dict
    w_masks: object


def actor_terms(critic, forward, obs, action, reference, q_g, capture,
                mode, strength, tau_squared, subset_generator,
                weight_generator, subset_size=256, fixed_indices=None):
    """One actor graph; g reference and all w queries are detached.

    cross proxy scalar expectation is 2R, while its autodiff gradient is that
    of R. K2 also uses a stopped coefficient; scalar and gradient semantics
    must be reported separately, not equated to its symmetric squared loss.
    """
    assert q_g.ndim == 2 and q_g.shape[0] == 2
    assert len(capture.masks) == 3
    assert all(not p.requires_grad for p in critic.parameters())
    assert obs.shape[0] == action.shape[0] == reference.shape[0] == q_g.shape[1]
    assert mode in ("cross", "square", "k2", "known")
    assert math.isfinite(strength) and 0 < strength <= (1 if mode == "known" else 10)
    reference = reference.detach()
    if mode == "known":
        assert math.isfinite(tau_squared) and tau_squared > 1e-12
        q_b = masked_reference(
            critic, forward, obs, reference, ReplayMask(capture.masks)
        )
        d = (q_g[0]-q_b[0]) - (q_g[1]-q_b[1])
        variance = d.detach().square() / 2
        weight = (tau_squared / (tau_squared+variance)).detach()
        auxiliary = strength * ((1-weight)*q_g.mean(0)).mean()
        return ActorTerms(auxiliary, d, None, None, capture.masks, None)

    size = min(int(subset_size), obs.shape[0])
    assert 0 < size <= obs.shape[0]
    if fixed_indices is None:
        indices = torch.randperm(
            obs.shape[0], generator=subset_generator, device="cpu"
        )[:size].to(obs.device)
    else:
        # Diagnostic panels freeze I outside repeated mask sampling.
        indices = fixed_indices.to(obs.device)
        assert indices.ndim == 1 and len(indices) == size
        assert len(torch.unique(indices)) == size
    obs_i, action_i, ref_i = obs[indices], action[indices], reference[indices]
    q_g_i = q_g[:, indices]  # Reuse the existing actor graph, never recompute a.
    q_b_g = masked_reference(
        critic, forward, obs_i, ref_i, ReplayMask(capture.masks, indices)
    )
    d_g = (q_g_i[0]-q_b_g[0]) - (q_g_i[1]-q_b_g[1])
    if mode == "square":
        return ActorTerms(
            strength*d_g.square().mean()/4, d_g, None, indices,
            capture.masks, None
        )
    w = ExplicitAuxMask(weight_generator)
    q_a_w = masked_reference(critic, forward, obs_i, action_i.detach(), w)
    assert len(w.masks) == 3
    q_b_w = masked_reference(
        critic, forward, obs_i, ref_i, ReplayMask(w.masks)
    )
    d_w = ((q_a_w[0]-q_b_w[0]) - (q_a_w[1]-q_b_w[1])).detach()
    coefficient = d_w if mode == "cross" else (d_w+d_g).detach()
    divisor = 2 if mode == "cross" else 4
    auxiliary = strength * (coefficient*d_g).mean() / divisor
    return ActorTerms(auxiliary, d_g, d_w, indices, capture.masks, w.masks)


class CrossMaskActor:
    """Call explicitly from an isolated diagnostic/shadow learner only.

    Zero strength and non-actor updates call frozen A1 directly. Nonzero actor
    updates observe the current TD data, capture the original actor Q graph,
    and shift both actor Q heads by the same scalar auxiliary loss. Mean
    aggregation then adds that scalar once to native entropy-minus-mean-Q.
    No new backward or optimizer call is introduced by the adapter.
    """
    def __init__(self, mode, strength, subset_seed, weight_seed,
                 tau_squared=None, subset_size=256):
        assert mode in ("cross", "square", "k2", "known")
        assert math.isfinite(strength) and 0 <= strength <= (1 if mode == "known" else 10)
        self.mode, self.strength = mode, float(strength)
        self.subset_seed, self.weight_seed = int(subset_seed), int(weight_seed)
        self.tau_squared, self.subset_size = tau_squared, int(subset_size)
        self.subset_generator = None
        self.weight_generators = {}
        self.last_report = None

    def _generators(self, device):
        if self.subset_generator is None:
            self.subset_generator = torch.Generator(device="cpu")
            self.subset_generator.manual_seed(self.subset_seed)
        key = str(device)
        if key not in self.weight_generators:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.weight_seed)
            self.weight_generators[key] = generator
        return self.subset_generator, self.weight_generators[key]

    def __call__(self, policy, batch):
        self.last_report = None
        if self.strength == 0:
            return paired_forward(policy, batch)
        assert policy._cfg.learn.online and not policy._cfg.learn.only_value
        assert not policy._auto_alpha and not policy._value_norm
        assert policy._actor_q_aggregation == "mean"
        assert policy._actor_update_freq == 10
        if (policy._forward_learn_cnt+1) % policy._actor_update_freq:
            return paired_forward(policy, batch)

        model, critic = policy._model, policy._model.critic
        native = model.compute_critic
        had_override = "compute_critic" in model.__dict__
        calls = []
        replay = []

        def observed(data):
            calls.append(1)
            if len(calls) == 1:
                replay.append(data["action"].detach())
                return native(data)  # Original TD path and paired masks.
            assert len(calls) == 2 and torch.is_grad_enabled()
            assert data["action"].requires_grad
            assert getattr(critic, "_dropout_pair_context", None) is None
            capture = CaptureNative()
            with freeze_parameters(critic):
                critic._dropout_pair_context = capture
                try:
                    result = native(data)
                finally:
                    critic._dropout_pair_context = None
                q = result["q_value"]
                subset_rng, weight_rng = self._generators(q.device)
                terms = actor_terms(
                    critic, native, data["obs"], data["action"], replay[0],
                    q, capture, self.mode, self.strength, self.tau_squared,
                    subset_rng, weight_rng, self.subset_size
                )
                # Valid only for mean aggregation and disabled value_norm,
                # asserted above. Lambda=0 never enters this changed path.
                result = dict(result)
                result["q_value"] = q-terms.auxiliary
                self.last_report = {
                    "mode": self.mode, "strength": self.strength,
                    "actor_batch": q.shape[1],
                    "auxiliary_batch": q.shape[1] if self.mode == "known"
                    else min(self.subset_size, q.shape[1]),
                    "g_layers": len(capture.masks),
                    "w_layers": len(terms.w_masks) if terms.w_masks else 0,
                    "proxy_scalar_is_not_objective": self.mode in ("cross", "k2")
                }
                return result

        model.compute_critic = observed
        try:
            result = paired_forward(policy, batch)
            assert len(calls) == 2 and self.last_report is not None
            return result
        finally:
            if had_override:
                model.compute_critic = native
            else:
                delattr(model, "compute_critic")
            critic._dropout_pair_context = None

    def learner_mode(self, policy):
        """Wrap a policy learn interface without patching its shared class."""
        native = policy.learn_mode
        adapter = self

        class LearnProxy:
            def __getattr__(self, name):
                return getattr(native, name)

            def forward(self, data):
                return adapter(policy, data)

        return LearnProxy()
