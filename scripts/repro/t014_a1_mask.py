"""Opt-in matched critic masks; default policy math and RNG path stay native."""
import torch
from ding.policy.edac import EDACPolicy

BASE_FORWARD = EDACPolicy._forward_learn
DEFAULT_MODE = 'independent'
INSTALLED = False


class Pair:
    def __init__(self, debug=False):
        self.masks = {}
        self.current_calls = self.target_calls = 0
        self.debug = debug
        self.target_grad_enabled = []


class Context:
    def __init__(self, pair, target):
        self.pair, self.target = pair, target

    def apply(self, h, layer, p, training):
        assert training and p == .005 and h.ndim == 3 and h.shape[0] == 2
        pair = self.pair
        if not self.target:
            assert layer not in pair.masks
            out, mask = torch.ops.aten.native_dropout(h, p, True)
            pair.masks[layer] = mask
            pair.current_calls += 1
            return out
        assert not torch.is_grad_enabled(), 'target must remain no_grad'
        mask = pair.masks[layer]
        assert mask.shape == h.shape and mask.device == h.device
        # Native target draw is intentionally discarded to retain future RNG use.
        torch.ops.aten.native_dropout(h, p, True)
        pair.target_calls += 1
        if pair.debug:
            pair.target_grad_enabled.append(torch.is_grad_enabled())
        return torch.ops.aten.native_dropout_backward(h, mask, 1.0/(1.0-p))


def paired_forward(self, data):
    mode = getattr(self, '_a1_mode', DEFAULT_MODE)
    if mode == 'independent':
        return BASE_FORWARD(self, data)
    assert mode == 'matched'
    current, target = self._model.critic, self._target_model.critic
    assert self._learn_model.critic is current
    assert getattr(current, '_dropout_pair_context', None) is None
    assert getattr(target, '_dropout_pair_context', None) is None
    pair = Pair(getattr(self, '_a1_debug', False))
    current._dropout_pair_context = Context(pair, False)
    target._dropout_pair_context = Context(pair, True)
    try:
        result = BASE_FORWARD(self, data)
        assert pair.current_calls == pair.target_calls == 3
        assert current._dropout_pair_context is target._dropout_pair_context is None
        self._a1_paired_updates = getattr(self, '_a1_paired_updates', 0) + 1
        if pair.debug:
            self._a1_last_pair = pair
        return result
    finally:
        current._dropout_pair_context = target._dropout_pair_context = None


def install(default=None):
    global INSTALLED, DEFAULT_MODE
    assert default in (None, 'independent', 'matched')
    if default is not None:
        DEFAULT_MODE = default
    if not INSTALLED:
        assert EDACPolicy._forward_learn is BASE_FORWARD
        EDACPolicy._forward_learn = paired_forward
        INSTALLED = True
