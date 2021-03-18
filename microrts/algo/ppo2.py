import torch
import torch.nn as nn
import torch.optim as optim
from .replay_buffer2 import ReplayBuffer2
import copy


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


class PPO2:
    def __init__(self, ac_model, lr=None, weight_decay=None, eps=1e-5,
                 log_interval=100, gamma=0.99, entropy_coef=.01, value_loss_coef=1, debug=False, ):
        self.actor_critic = ac_model
        self.target_net = copy.deepcopy(ac_model)
        self.optimizer = optim.Adam(ac_model.parameters(), lr, weight_decay=weight_decay)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.eps = eps
        self.log_interval = log_interval
        self.debug = debug

    def update(self, rollouts: ReplayBuffer2, iter_idx, device="cpu", callback=None):
        if rollouts.__len__() <= 0:
            return
        samples = rollouts.sample(batch_size='all')
        optimizer = self.optimizer
        value_criteria = torch.nn.MSELoss()

        total_loss = 0
        total_rewards = 0
        states, units, utts, action_type, action_parm, action_prod, next_states, rewards, hxses, done_masks, durations, ctf_v, irews = samples.to(
            device)
        if self.actor_critic.recurrent:
            value_old, policy_old, _ = self.target_net.forward(spatial_feature=states, unit_feature=units,
                                                               utt_feature=utts)
            value, policy, _ = self.actor_critic.forward(spatial_feature=states, unit_feature=units, utt_feature=utts)
        else:
            values_old, policy_old, _ = self.target_net.forward(spatial_feature=states, unit_feature=units,
                                                                utt_feature=utts)
            value, policy, _ = self.actor_critic.forward(spatial_feature=states, unit_feature=units, utt_feature=utts)
        ua_type = torch.distributions.Categorical(probs=policy[:][0])
        # ua_idxes_type = ua_type.sample()
        ua_parm = torch.distributions.Categorical(probs=policy[:][1])
        # ua_idxes_parm = ua_parm.sample()
        ua_prod = torch.distributions.Categorical(probs=policy[:][2])
        # ua_idxes_prod = ua_prod.sample()

        value_next = self.target_net.critic_forward(spatial_feature=next_states, utt_feature=utts).detach()

        pi_sa_type = policy[0][:].gather(1, action_type)
        pi_sa_old_type = policy_old[0][:].gather(1, action_type)
        log_pi_sa_type = torch.log(pi_sa_type + self.eps).detach()

        pi_sa_parm = policy[1][:].gather(1, action_parm)
        pi_sa_old_parm = policy_old[1][:].gather(1, action_parm)
        log_pi_sa_parm = torch.log(pi_sa_type + self.eps).detach()

        pi_sa_prod = policy[2][:].gather(1, action_prod)
        pi_sa_old_prod = policy_old[2][:].gather(1, action_prod)
        log_pi_sa_prod = torch.log(pi_sa_prod + self.eps).detach()

        targets = rewards + self.gamma ** durations * value_next * done_masks

        # adv = (targets - ctf_v).detach()
        adv = (targets - value).detach()

        entropy_loss = -ua_type.entropy().mean() - ua_parm.entropy().mean() - ua_prod.entropy().mean()

        ratio_type = pi_sa_type / pi_sa_old_type
        ratio_parm = pi_sa_parm / pi_sa_old_parm
        ratio_prod = pi_sa_prod / pi_sa_old_prod

        clip_ratio = 0.2
        clip_type_adv = torch.clamp(ratio_type, 1 - clip_ratio, 1 + clip_ratio) * adv
        clip_parm_adv = torch.clamp(ratio_parm, 1 - clip_ratio, 1 + clip_ratio) * adv
        clip_prod_adv = torch.clamp(ratio_prod, 1 - clip_ratio, 1 + clip_ratio) * adv

        policy_loss = -(torch.min(ratio_parm * adv, clip_parm_adv)).mean() \
                      - (torch.min(ratio_type * adv, clip_type_adv)).mean() \
                      - (torch.min(ratio_prod * adv, clip_prod_adv)).mean()
        value_loss = value_criteria(value, targets)

        total_loss = policy_loss + value_loss * self.value_loss_coef + self.entropy_coef * entropy_loss
        total_rewards = rewards.mean()

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), .5)
        optimizer.step()

        results = {
            "all_losses": total_loss
        }

        if iter_idx % self.log_interval == 0:
            if callback:
                callback(iter_idx, results)

        with torch.no_grad():
            soft_update(self.target_net, self.actor_critic, tau=0.001)
        rollouts.refresh()