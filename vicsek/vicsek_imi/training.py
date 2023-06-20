import torch
from torch import autograd
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from utils import gen_graph_batch, tran_adj_coo
import numpy as np


# Dataset that returns transition tuples of the form (s, a, r, s', terminal)
class TransitionDataset(Dataset):
    def __init__(self, transitions):
        super().__init__()
        self.states, self.actions, self.rewards, self.terminals = transitions['states'], transitions['actions'].detach(), transitions['rewards'], transitions['terminals']
        self.dm = transitions['dm']

    # Allows string-based access for entire data of one type, or int-based access for single transition
    def __getitem__(self, idx):
        if isinstance(idx, str):
            if idx == 'states':
                return self.states
            elif idx == 'actions':
                return self.actions
        else:
            return self.states[idx], self.actions[idx], self.rewards[idx], self.states[idx + 1], self.terminals[idx], self.dm[idx]

    def __len__(self):
        return self.terminals.size(0) - 1  # Need to return state and next state


# Computes and stores generalised advantage estimates ψ in the set of trajectories
def compute_advantages(trajectories, next_value, discount, trace_decay):
    with torch.no_grad():  # Do not differentiate through advantage calculation
        reward_to_go, advantage = torch.tensor([0.]), torch.tensor([0.])
        trajectories['rewards_to_go'], trajectories['advantages'] = torch.empty_like(trajectories['rewards']), torch.empty_like(trajectories['rewards'])
        for t in reversed(range(trajectories['states'].size(0))):
            reward_to_go = trajectories['rewards'][t] + (1 - trajectories['terminals'][t]) * (discount * reward_to_go)  # Reward-to-go/value R
            trajectories['rewards_to_go'][t] = reward_to_go
            td_error = trajectories['rewards'][t] + (1 - trajectories['terminals'][t]) * discount * next_value - trajectories['values'][t]  # TD-error δ
            advantage = td_error + (1 - trajectories['terminals'][t]) * discount * trace_decay * advantage  # Generalised advantage estimate ψ
            trajectories['advantages'][t] = advantage
            next_value = trajectories['values'][t]


# Performs one PPO update (assumes trajectories for first epoch are attached to agent)
def ppo_update(agent, trajectories, agent_optimiser, ppo_clip, epoch, policy, value_loss_coeff=1, entropy_reg_coeff=1):
    # Recalculate outputs for subsequent iterations
    if epoch > 0:
        if policy == 'gat':
            batch = gen_graph_batch(trajectories['states'], trajectories['dm'])
            trajectories = {k: trajectories[k].reshape(trajectories[k].shape[0] * trajectories[k].shape[1], -1) for k in trajectories.keys()}
            policy, trajectories['values'] = agent(batch.x, batch.edge_index)
        else:
            policy, trajectories['values'] = agent(trajectories['states'])
        trajectories['values'] = trajectories['values'].unsqueeze(1)
        trajectories['log_prob_actions'] = policy.log_prob(trajectories['actions'].detach()).sum(axis=-1, keepdim=True)
        trajectories['entropies'] = policy.entropy().sum(axis=-1, keepdim=True)
    policy_ratio = (trajectories['log_prob_actions'] - trajectories['old_log_prob_actions']).exp()
    policy_loss = -torch.min(policy_ratio * trajectories['advantages'], torch.clamp(policy_ratio, min=1 - ppo_clip, max=1 + ppo_clip) * trajectories['advantages']).mean()  # Update the policy by maximising the clipped PPO objective
    value_loss = F.mse_loss(trajectories['values'], trajectories['rewards_to_go'])  # Fit value function by regression on mean squared error
    entropy_reg = -trajectories['entropies'].mean()  # Add entropy regularisation

    agent_optimiser.zero_grad()
    (policy_loss + value_loss_coeff * value_loss + entropy_reg_coeff * entropy_reg).backward(retain_graph=True)
    clip_grad_norm_(agent.parameters(), 1)  # Clamp norm of gradients
    agent_optimiser.step()
    print("policy_loss: {}, value_loss: {}, entropy_reg: {}".format(policy_loss, value_loss, entropy_reg))


# # Performs a behavioural cloning update
# def behavioural_cloning_update(agent, expert_trajectories, agent_optimiser, batch_size):
#     expert_dataloader = DataLoader(expert_trajectories, batch_size=batch_size, shuffle=True, drop_last=True)
#     for expert_transition in expert_dataloader:
#         expert_state, expert_action = expert_transition[0], expert_transition[1]
#         agent_optimiser.zero_grad()
#         behavioural_cloning_loss = -agent.log_prob(expert_state, expert_action).mean()  # Maximum likelihood objective
#         behavioural_cloning_loss.backward()
#         agent_optimiser.step()


# Performs an adversarial imitation learning update
def adversarial_imitation_update(agent, discriminator, expert_trajectories, policy_trajectories, discriminator_optimiser, batch_size, policy, r1_reg_coeff=1):
    expert_dataloader = DataLoader(expert_trajectories, batch_size=batch_size, shuffle=True, drop_last=True)
    policy_dataloader = DataLoader(policy_trajectories, batch_size=batch_size, shuffle=True, drop_last=True)

    # Iterate over mininum of expert and policy data
    for expert_transition, policy_transition in zip(expert_dataloader, policy_dataloader):
        expert_state, expert_action, expert_next_state, expert_terminal = expert_transition[0], expert_transition[1], expert_transition[3], expert_transition[4]
        policy_state, policy_action, policy_next_state, policy_terminal = policy_transition[0], policy_transition[1], policy_transition[3], policy_transition[4]
        if policy == 'gat':
            expert_dm = expert_transition[5]
            policy_dm = policy_transition[5]
            expert_batch = gen_graph_batch(expert_state, expert_dm)
            policy_batch = gen_graph_batch(policy_state, policy_dm)
            expert_action = expert_action.reshape(expert_action.shape[0] * expert_action.shape[1], -1)
            policy_action = expert_action.reshape(policy_action.shape[0] * policy_action.shape[1], -1)
            with torch.no_grad():
                expert_data_policy = agent.log_prob(expert_batch.x, expert_action, expert_batch.edge_index).exp()
                policy_data_policy = agent.log_prob(policy_batch.x, policy_action, policy_batch.edge_index).exp()
            expert_state = expert_state.reshape(expert_state.shape[0] * expert_state.shape[1], -1)
            expert_next_state = expert_next_state.reshape(expert_next_state.shape[0] * expert_next_state.shape[1], -1)
            expert_terminal = expert_terminal.reshape(expert_terminal.shape[0] * expert_terminal.shape[1], -1)
            policy_state = policy_state.reshape(policy_state.shape[0] * policy_state.shape[1], -1)
            policy_next_state = policy_next_state.reshape(policy_next_state.shape[0] * policy_next_state.shape[1], -1)
            policy_terminal = policy_terminal.reshape(policy_terminal.shape[0] * policy_terminal.shape[1], -1)
        else:
            with torch.no_grad():
                expert_data_policy = agent.log_prob(expert_state, expert_action).exp()
                policy_data_policy = agent.log_prob(policy_state, policy_action).exp()
        D_expert = discriminator(expert_state, expert_action, expert_next_state, expert_data_policy, expert_terminal)
        D_policy = discriminator(policy_state, expert_action, policy_next_state, policy_data_policy, policy_terminal)
        # Binary logistic regression
        discriminator_optimiser.zero_grad()
        expert_loss = F.binary_cross_entropy(D_expert, torch.ones_like(D_expert))  # Loss on "real" (expert) data
        autograd.backward(expert_loss, create_graph=True)
        r1_reg = 0
        for param in discriminator.parameters():
            r1_reg += param.grad.norm().mean()  # R1 gradient penalty
        policy_loss = F.binary_cross_entropy(D_policy, torch.zeros_like(D_policy))  # Loss on "fake" (policy) data
        (policy_loss + r1_reg_coeff * r1_reg).backward()
        discriminator_optimiser.step()


def target_estimation_update(discriminator, expert_trajectories, discriminator_optimiser, batch_size):
    expert_dataloader = DataLoader(expert_trajectories, batch_size=batch_size, shuffle=True, drop_last=True)

    for expert_transition in expert_dataloader:
        expert_state, expert_action = expert_transition['states'], expert_transition['actions']

        discriminator_optimiser.zero_grad()
        prediction, target = discriminator(expert_state, expert_action)
        regression_loss = F.mse_loss(prediction, target)
        regression_loss.backward()
        discriminator_optimiser.step()

# dataset = Planetoid(root='dataset', name='Cora', transform=NormalizeFeatures())
# dataset = TUDataset(root='dataset', name='ENZYMES', use_node_attr=True)
# print(len(dataset))
# loader = GLoader(dataset, batch_size=32, shuffle=True)
# for batch in loader:
#     print(batch)
