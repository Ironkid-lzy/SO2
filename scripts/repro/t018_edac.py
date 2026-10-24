"""Task-local ES ablation; frozen T010 default implementation is unchanged."""
import torch
from t010_edac import OfflineEDAC, q_values, sample_actor


class ParameterizedEDAC(OfflineEDAC):
    def __init__(self, model, es_coefficient=1):
        if es_coefficient not in (0, 1):
            raise ValueError("T018 permits only the frozen eta1 default and eta0 ablation")
        super().__init__(model)
        self.es_coefficient = es_coefficient

    def losses(self, batch):
        if self.es_coefficient == 1:
            # Exact delegation avoids changing any default operation or ordering.
            return super().losses(batch)
        obs, next_obs = batch["observations"], batch["next_observations"]
        actions = batch["actions"].requires_grad_(True)
        act, _, _, logp = sample_actor(self.model, obs)
        alpha_loss = -(self.log_alpha * (logp - 6).detach()).mean()
        alpha = self.log_alpha.exp()
        actor_loss = (alpha * logp - q_values(self.model, obs, act).min(0)[0]).mean()
        prediction = q_values(self.model, obs, actions)
        next_act, _, _, next_logp = sample_actor(self.model, next_obs, reparameterize=False)
        target_value = q_values(self.target, next_obs, next_act).min(0)[0] - alpha * next_logp
        target = batch["rewards"] + (1 - batch["terminals"]) * .99 * target_value
        td = (prediction - target.detach().unsqueeze(0)).square().mean(dim=(1, 2)).sum()
        # Retain the native ES query's shapes/layout/dropout RNG consumption.
        # Its output is discarded, with no graph, action derivative or ES backward.
        # This intentionally retains one cheap forward rather than approximating
        # native CUDA dropout RNG advancement with a custom mask generator.
        with torch.no_grad():
            obs_tile = obs.unsqueeze(0).repeat(2, 1, 1)
            action_tile = actions.unsqueeze(0).repeat(2, 1, 1)
            q_values(self.model, obs_tile, action_tile)
        es = td.new_zeros(())
        return {"alpha": alpha_loss, "actor": actor_loss, "td": td,
                "es": es, "critic": td}
