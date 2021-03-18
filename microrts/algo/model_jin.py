import torch
import torch.nn as nn
import torch.nn.functional as F
from microrts.algo.config import model_path
import os
import numpy as np
from torch import Tensor
from microrts.rts_wrapper.envs.datatypes import *
from torch.distributions import Categorical


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

# args = DotDict({
#     'lr': 0.001,
#     'dropout': 0.3,
#     'epochs': 1,
#     'batch_size': 64,
#     'map_channels': 107,
#     'num_channels': 64,  # conv_in of the map filters size
#     'num_map_channels': 64,  # conv_out of the map filters size
#     'pooled_size': 8,  # target output size of conv_out
#     'utt_features_size': 168,  #
#     'num_utt_out': 64,
#     'num_mu_size': 256,
#     'unit_features_size': 17,
#     'lstm_hidden_size': 64,
#     'action_size': 65,
#     'lstm_num_layers': 2,
#     'cuda': torch.cuda.is_available(),
# })


class ActorCritic2(nn.Module):

    def __init__(self, map_size, input_channel=61, recurrent=False, hidden_size=256, ):
        """[summary]
            Arguments:
                map_size {tuple} -- (map_height, map_width)

            Keyword Arguments:
                input_channel {int} -- [description] (default: {61})
                unit_feature_size {int} -- [description] (default: {18})
        """
        super(ActorCritic2, self).__init__()
        self.recurrent = recurrent

        self.input_channel = input_channel

        self.utt_feature_size = 43 * 7
        self.unit_feature_size = 25

        self.pooled_size = 6

        self.conv_out_size = 16
        self.unit_out_feature_size = 64
        self.utt_out_feature_size = 64
        self.critic_out_size = 1

        self.state_feature_size = self.utt_out_feature_size + self.conv_out_size * self.pooled_size * self.pooled_size

        self.actor_parm_size = 4
        self.actor_prod_size = 6
        self.actor_type_size = 6

        hidden_size = 64

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
        self.shared_conv = nn.Sequential(
            init_(nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=1, stride=1, padding=1)),
            init_(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=1)),
            init_(nn.Conv2d(in_channels=64, out_channels=self.conv_out_size, kernel_size=1, stride=1, padding=1)),
            nn.AdaptiveMaxPool2d(self.pooled_size), # self.conv_out_size * self.pooled_size * self.pooled_size
            Flatten(),
        )

        self.unit_mlps = nn.Sequential(
            init_(nn.Linear(self.unit_feature_size, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, self.unit_out_feature_size)), nn.ReLU(),
        )

        self.utt_mlps = nn.Sequential(
            init_(nn.Linear(self.utt_feature_size, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, self.utt_out_feature_size)), nn.ReLU()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(self.state_feature_size, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, 1))
        )

        self.actor_mlps = nn.Sequential(
            init_(nn.Linear(self.state_feature_size + self.unit_out_feature_size, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU(),
        )

        self.actor_prod = nn.Sequential(
            init_(nn.Linear(hidden_size, self.actor_prod_size)),
            nn.Softmax(dim=1)
        )

        self.actor_parm = nn.Sequential(
            init_(nn.Linear(hidden_size, self.actor_parm_size)),
            nn.Softmax(dim=1)
        )

        self.actor_type = nn.Sequential(
            init_(nn.Linear(hidden_size, self.actor_type_size)),
            nn.Softmax(dim=1)
        )

    def critic_forward(self, spatial_feature: Tensor, utt_feature: Tensor):
        conv_out_feature = self.shared_conv(spatial_feature)
        utt_out_feature = self.utt_mlps(utt_feature)
        state_feature = torch.cat([conv_out_feature, utt_out_feature], dim=1) #
        value = self.critic(state_feature)
        return value

    def actor_forward(self, state_feature: Tensor, unit_out_feature: Tensor):
        x = torch.cat([state_feature, unit_out_feature], dim=-1)
        x = self.actor_mlps(x)
        ua_prod = self.actor_prod(x)
        ua_parm = self.actor_parm(x)
        ua_type = self.actor_type(x)
        return ua_type, ua_parm, ua_prod

    def forward(self, spatial_feature: Tensor, utt_feature: Tensor, unit_feature: Tensor = None, hxses: Tensor = None):
        conv_out_feature = self.shared_conv(spatial_feature)
        unit_out_feature = self.unit_mlps(unit_feature)
        utt_out_feature = self.utt_mlps(utt_feature)
        state_feature = torch.cat([conv_out_feature, utt_out_feature], dim=1)  #
        value = self.critic(state_feature)

        ua_type, ua_parm, ua_prod = self.actor_forward(state_feature, unit_out_feature)
        return value, (ua_type, ua_parm, ua_prod), hxses
