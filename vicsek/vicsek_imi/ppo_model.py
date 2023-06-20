import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal
from torch.nn import functional as F
from torch_geometric.nn import GATConv


# Concatenates the state and one-hot version of an action
def _join_state_action(state, action, action_size):
    return torch.cat([state, F.one_hot(action, action_size).to(dtype=torch.float32)], dim=1)


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim,hidden_sizes, layer_norm):
        
        super().__init__()
       
        self.actor_fc1 = nn.Linear(obs_dim, hidden_sizes)
        self.actor_fc2 = nn.Linear(hidden_sizes,act_dim)

        self.critic_fc1 = nn.Linear(obs_dim, hidden_sizes)
        self.critic_fc2 = nn.Linear(hidden_sizes, 1)
    
        if layer_norm:
            
            self.layer_norm(self.actor_fc1, std=1.0)
            self.layer_norm(self.actor_fc2, std=0.01)
        
            self.layer_norm(self.critic_fc1, std=1.0)
            self.layer_norm(self.critic_fc2, std=1.0)
        
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
    
    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        
    def actor(self, obs):
        x = torch.tanh(self.actor_fc1(obs))
        mu = self.actor_fc2(x)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def critic(self,obs):
        x = torch.tanh(self.critic_fc1(obs))
        value =self.critic_fc2(x)
        return torch.squeeze(value)

    def forward(self, obs):
            pi =self.actor(obs)
            v =self.critic(obs)
            return pi, v

    def log_prob(self, state, action):        
        return self.actor(state).log_prob(action).sum(axis=-1)


class GraphActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, local_size):
        super().__init__()
        self.actor_gat1 = GATConv(obs_dim - local_size, hidden_sizes)
        self.actor_fc1 = nn.Linear(hidden_sizes + local_size, act_dim)
        self.critic_gat1 = GATConv(obs_dim - local_size, hidden_sizes)
        self.critic_fc1 = nn.Linear(hidden_sizes + local_size, 1)
        self.local_size = local_size

        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))


    def actor(self, obs, adj):
        state = obs[:, :-self.local_size]
        local = obs[:, -self.local_size:]
        x = torch.tanh(self.actor_gat1(state, adj))
        x = torch.cat((x, local), dim=1)
        mu = self.actor_fc1(x)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def critic(self, obs, adj):
        state = obs[:, :-self.local_size]
        local = obs[:, -self.local_size:]
        x1 = torch.tanh(self.critic_gat1(state, adj))
        x1 = torch.cat((x1, local), dim=1)
        value = self.critic_fc1(x1)
        return torch.squeeze(value)

    def forward(self, obs, adj):
        pi = self.actor(obs, adj)
        v = self.critic(obs, adj)
        return pi, v

    def log_prob(self, state, action, adj):
        return self.actor(state, adj).log_prob(action).sum(axis=-1)


class AIRLDiscriminator(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, discount, state_only=False):
        super().__init__()
        self.action_size, self.state_only = action_size, state_only
        self.discount = discount
        self.g = nn.Linear(state_size if state_only else state_size + action_size, 1)  # Reward function r
        self.h = nn.Sequential(nn.Linear(state_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))  # Shaping function Φ

    def reward(self, state, action):
        #return self.g(state if self.state_only else _join_state_action(state, action, self.action_size)).squeeze(dim=1)
        return self.g(state if self.state_only else _join_state_action(state, action, self.action_size))

    def value(self, state):
        #return self.h(state).squeeze(dim=1)
        return self.h(state)

    def forward(self, state, action, next_state, policy, terminal):
        f = self.reward(state, action) + (1 - terminal) * (self.discount * self.value(next_state) - self.value(state))
        f_exp = f.exp()
        return f_exp / (f_exp + policy)

    def predict_reward(self, state, action, next_state, policy, terminal):
        D = self.forward(state, action, next_state, policy, terminal)
        return torch.log(D) - torch.log1p(-D)


# class GAILDiscriminator(nn.Module):
#   def __init__(self, state_size, action_size, hidden_size, state_only=False):
#     super().__init__()
#     self.action_size, self.state_only = action_size, state_only
#     input_layer = nn.Linear(state_size if state_only else state_size + action_size, hidden_size)
#     self.discriminator = nn.Sequential(input_layer, nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1), nn.Sigmoid())

#   def forward(self, state, action):
#     D = self.discriminator(state if self.state_only else _join_state_action(state, action, self.action_size)).squeeze(dim=1)
#     return D
  
#   def predict_reward(self, state, action):
#     D = self.forward(state, action)
#     return torch.log(D) - torch.log1p(-D)


# class GMMILDiscriminator(nn.Module):
#   def __init__(self, state_size, action_size, state_only=True):
#     super().__init__()
#     self.action_size, self.state_only = action_size, state_only
#     self.gamma_1, self.gamma_2 = None, None

#   def predict_reward(self, state, action, expert_state, expert_action):
#     state_action = state if self.state_only else _join_state_action(state, action, self.action_size)
#     expert_state_action = expert_state if self.state_only else _join_state_action(expert_state, expert_action, self.action_size)
    
#     # Use median heuristics to set data-dependent bandwidths
#     if self.gamma_1 is None:
#       self.gamma_1 = 1 / _squared_distance(state_action, expert_state_action).median().item()
#       self.gamma_2 = 1 / _squared_distance(expert_state_action, expert_state_action).median().item()

#     # Return sum of maximum mean discrepancies
#     distances = _squared_distance(state_action, expert_state_action)
#     return _gaussian_kernel(distances, gamma=self.gamma_1).mean(dim=1) + _gaussian_kernel(distances, gamma=self.gamma_2).mean(dim=1)



# class EmbeddingNetwork(nn.Module):
#   def __init__(self, input_size, hidden_size):
#     super().__init__()
#     self.embedding = nn.Sequential(nn.Linear(input_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, input_size))

#   def forward(self, input):
#     return self.embedding(input)


# class REDDiscriminator(nn.Module):
#   def __init__(self, state_size, action_size, hidden_size, state_only=False):
#     super().__init__()
#     self.action_size, self.state_only = action_size, state_only
#     self.gamma = None
#     self.predictor = EmbeddingNetwork(state_size if state_only else state_size + action_size, hidden_size)
#     self.target = EmbeddingNetwork(state_size if state_only else state_size + action_size, hidden_size)
#     for param in self.target.parameters():
#       param.requires_grad = False

#   def forward(self, state, action):
#     state_action = state if self.state_only else _join_state_action(state, action, self.action_size)
#     prediction, target = self.predictor(state_action), self.target(state_action)
#     return prediction, target

#   def predict_reward(self, state, action, sigma=1):  # TODO: Set sigma based such that r(s, a) from expert demonstrations ≈ 1
#     prediction, target = self.forward(state, action)
#     return _gaussian_kernel(F.pairwise_distance(prediction, target, p=2).pow(2), gamma=1)
