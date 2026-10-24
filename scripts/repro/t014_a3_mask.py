"""Actor-only Dropout-off: consume original masks, preserve matched TD path."""
import torch
from ding.policy.edac import EDACPolicy
import t014_a1_mask as a1

DEFAULT_OFF = False
INSTALLED = False
A1_FORWARD = a1.paired_forward


class ActorContext:
    def __init__(self):
        self.layers = []

    def apply(self, h, layer, p, training):
        assert training and p == .005 and h.ndim == 3 and h.shape[0] == 2
        assert layer == len(self.layers) and torch.is_grad_enabled()
        # Consume exactly the same native RNG draw as actor-on, retaining identity
        # derivative through h. This is not a mask expectation through LN/ReLU.
        torch.ops.aten.native_dropout(h, p, True)
        self.layers.append(layer)
        return h


def actor_forward(self, data):
    if not getattr(self, '_a3_actor_off', DEFAULT_OFF):
        return A1_FORWARD(self, data)
    assert getattr(self, '_a1_mode', a1.DEFAULT_MODE) == 'matched'
    critic, target = self._model.critic, self._target_model.critic
    actor_due = self._cfg.learn.online and (self._forward_learn_cnt + 1) % self._actor_update_freq == 0
    calls = []
    actor = ActorContext()

    def before(module, inputs):
        calls.append(len(calls))
        if len(calls) == 1:
            assert isinstance(module._dropout_pair_context, a1.Context)
        else:
            assert len(calls) == 2 and actor_due
            assert module._dropout_pair_context is None
            module._dropout_pair_context = actor

    handle = critic.register_forward_pre_hook(before)
    try:
        result = A1_FORWARD(self, data)
        assert len(calls) == (2 if actor_due else 1)
        assert actor.layers == ([0, 1, 2] if actor_due else [])
        assert critic.training and target.training
        assert critic._dropout_pair_context is target._dropout_pair_context is None
        self._a3_actor_queries = getattr(self, '_a3_actor_queries', 0) + int(actor_due)
        return result
    finally:
        handle.remove()
        critic._dropout_pair_context = target._dropout_pair_context = None


def install(default_off=False):
    global INSTALLED, DEFAULT_OFF
    a1.install('matched')
    DEFAULT_OFF = default_off
    if not INSTALLED:
        assert EDACPolicy._forward_learn is A1_FORWARD
        EDACPolicy._forward_learn = actor_forward
        INSTALLED = True
