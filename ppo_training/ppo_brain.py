import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal weight initialization for PPO stability"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOBrain(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=64):
        super(PPOBrain, self).__init__()

        # Decoupled Actor (Policy)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        )
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

        # Decoupled Critic (Value)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0)
        )

    def forward(self, x):
        """Standard forward pass returning mean, std, and value"""
        mu = self.actor(x)
        std = self.log_std.exp().expand_as(mu)
        value = self.critic(x)
        return mu, std, value

    def get_action_and_value(self, x, action=None):
        """Used strictly during PPO rollout and update phases"""
        mu, std, value = self.forward(x)
        dist = Normal(mu, std)

        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), value.squeeze(-1)