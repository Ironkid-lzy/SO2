"""Observation-only guards: no forwards, mode changes, or random draws."""
import math
from t010_edac import *
from t010_run import finite_model

def install_guards(run):
    from ding.policy.edac import EDACPolicy
    original=EDACPolicy._forward_learn
    state={}
    def guarded(self,data):
        state["policy"]=self
        try:
            out=original(self,data)
            for k,v in out.items():
                if isinstance(v,(int,float)):
                    assert math.isfinite(v), f"nonfinite learner scalar {k}"
            tensors=[v.detach().reshape(-1) for v in out.values() if isinstance(v,torch.Tensor)]
            assert not tensors or bool(torch.isfinite(torch.cat(tensors)).all()), "nonfinite learner loss"
            count=self._forward_learn_cnt
            if count==1 or count%1000==0:
                finite_model(self._model);finite_model(self._target_model)
                assert self._actor_q_aggregation=="mean" and not self._auto_alpha
                assert self._actor_update_freq==10
                atomic_json(run/"learner_guard.json",{"passed":True,"critic_updates":count,
                            "actor_updates":count//10,"alpha_updates":0})
            return out
        except BaseException:
            atomic_torch(run/"partial"/f"guard_learner_{self._forward_learn_cnt}.pth.tar",self.learn_mode.state_dict())
            raise
    EDACPolicy._forward_learn=guarded
    # Evaluation policy values and checkpoint tensors are checked at each point.
    from ding.worker import InteractionSerialEvaluator
    original_eval=InteractionSerialEvaluator.eval
    def evaluate(self,*args,**kwargs):
        if "policy" in state:
            finite_model(state["policy"]._model);finite_model(state["policy"]._target_model)
        result=original_eval(self,*args,**kwargs)
        returns=self.last_episode_reward
        assert len(returns)==20 and all(math.isfinite(v) for v in returns), "invalid evaluation returns"
        return result
    InteractionSerialEvaluator.eval=evaluate
    return state
