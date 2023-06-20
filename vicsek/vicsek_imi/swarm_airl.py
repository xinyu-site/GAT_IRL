# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 19:38:46 2020

@author: yx
"""
import argparse
from collections import deque
import os
from tqdm import tqdm
import numpy as np
import torch
from torch.nn import functional as F
import sys
from ppo_model import MLPActorCritic, AIRLDiscriminator, GraphActorCritic
from utils import flatten_list_dicts2, flatten_swarm_traj, flatten_list_dicts, tran_adj_coo
from ma_envs.envs.point_envs import rendezvous
#from evaluation import evaluate_agent
from training import TransitionDataset, adversarial_imitation_update, compute_advantages, ppo_update, target_estimation_update
import warnings
warnings.filterwarnings("ignore")

entropy_record=[]
episode_return_record=[]
nr_agents = 10


parser = argparse.ArgumentParser(description='IL')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
parser.add_argument('--steps', type=int, default=300000, metavar='T', help='Number of environment steps')
parser.add_argument('--hidden-size', type=int, default=64, metavar='H', help='Hidden size')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount')
parser.add_argument('--trace-decay', type=float, default=0.95, metavar='λ', help='GAE trace decay')
parser.add_argument('--ppo-clip', type=float, default=0.2, metavar='ε', help='PPO clip ratio')
parser.add_argument('--ppo-epochs', type=int, default=4, metavar='K', help='PPO epochs')
parser.add_argument('--value-loss-coeff', type=float, default=1, metavar='c1', help='Value loss coefficient')
parser.add_argument('--entropy-loss-coeff', type=float, default=0, metavar='c2', help='Entropy regularisation coefficient')
parser.add_argument('--learning-rate', type=float, default=2e-4, metavar='η', help='Learning rate')
parser.add_argument('--batch-size', type=int, default=2048, metavar='B', help='Minibatch size')
parser.add_argument('--max-grad-norm', type=float, default=1, metavar='N', help='Maximum gradient L2 norm')
parser.add_argument('--evaluation-interval', type=int, default=10000, metavar='EI', help='Evaluation interval')
parser.add_argument('--evaluation-episodes', type=int, default=50, metavar='EE', help='Evaluation episodes')
parser.add_argument('--save-trajectories', action='store_true', default=False, help='Store trajectories from agent after training')
parser.add_argument('--imitation', type=str, default='AIRL', choices=['AIRL', 'BC', 'GAIL', 'GMMIL'], metavar='I', help='Imitation learning algorithm')
parser.add_argument('--state-only', action='store_true', default=True, help='State-only imitation learning')
parser.add_argument('--imitation-epochs', type=int, default=5, metavar='IE', help='Imitation learning epochs')
parser.add_argument('--imitation-batch-size', type=int, default=2048, metavar='IB', help='Imitation learning minibatch size')
parser.add_argument('--imitation-replay-size', type=int, default=4, metavar='IRS', help='Imitation learning trajectory replay size')
parser.add_argument('--r1-reg-coeff', type=float, default=1, metavar='γ', help='R1 gradient regularisation coefficient')
parser.add_argument('--policy-name', type=str, default='mlp', choices=['mlp', 'gat'], help='Type of policy network')


args = parser.parse_args()
torch.manual_seed(args.seed)
#os.makedirs('results', exist_ok=True)

schedule_adam='False'


env = rendezvous.RendezvousEnv(nr_agents=nr_agents,
                               obs_mode='fix_acc',
                               comm_radius=100,
                               world_size=100,
                               distance_bins=8,
                               bearing_bins=8,
                               torus=True,
                               dynamics='unicycle')
env.seed(args.seed)
if args.policy_name == 'mlp':
    agent = MLPActorCritic(env.observation_space.shape[0], 1, args.hidden_size,True)
elif args.policy_name == 'gat':
    agent = GraphActorCritic(env.observation_space.shape[0], 1, args.hidden_size, env.observation_space.dim_local_o)
agent_optimiser = torch.optim.Adam(agent.parameters(), lr=args.learning_rate)


if args.imitation:
    # Set up expert trajectories dataset
    expert_trajectories = torch.load('/home/tyk/project/collective_code/vicsek/vicsek_imi/epl_50/na_10-epi_100-obsm_fix_acc-policy_gat-vicsek_data.pth')
    if args.policy_name == 'gat':
        expert_episodes = int((expert_trajectories['terminals'].shape[0]) / 80)
        expert_trajectories = TransitionDataset(expert_trajectories)
    else:
        expert_trajectories = {k: expert_trajectories[k].reshape(expert_trajectories[k].shape[0] * expert_trajectories[k].shape[1], -1) for k in expert_trajectories}
        expert_trajectories = TransitionDataset(expert_trajectories)
        expert_episodes = int((expert_trajectories.__len__() + 1) / 80 / env.nr_agents)
    
    # Set up discriminator
    discriminator = AIRLDiscriminator(env.observation_space.shape[0], 1, args.hidden_size, args.discount, state_only=args.state_only)
    discriminator_optimiser = torch.optim.RMSprop(discriminator.parameters(), lr=args.learning_rate)
# Metrics
metrics = dict(train_steps=[], train_returns=[], test_steps=[], test_returns=[])


# Main training loop
state, dm = env.reset()
terminal, episode_return, trajectories, policy_trajectory_replay_buffer = False, 0, [], deque(maxlen=args.imitation_replay_size)


pbar = tqdm(range(1, args.steps + 1), unit_scale=1, smoothing=0)
for step in pbar:
    if args.policy_name == 'gat':
        edge_index = tran_adj_coo(dm)
        policy, value = agent(state, edge_index)
        action = policy.sample()
        log_prob_action, entropy = agent.log_prob(state, action, edge_index), policy.entropy().sum(axis=-1)
    else:
        policy, value = agent(state)
        action = policy.sample()
        log_prob_action, entropy = agent.log_prob(state, action), policy.entropy().sum(axis=-1)


    action_0 = np.ones((env.nr_agents, 1)) * 0.5
    action_plus = np.concatenate((action_0, action.numpy()), axis=1)

    next_state, next_dm, reward, terminal ,info = env.step(action_plus)

    episode_return += reward
    trajectories.append(dict(states=state,
                             dm=dm,
                             actions=action,
                             rewards=reward.unsqueeze(1),
                             terminals=torch.tensor([terminal]*env.nr_agents, dtype=torch.float32).unsqueeze(1),
                             log_prob_actions=log_prob_action.unsqueeze(1), 
                             old_log_prob_actions=log_prob_action.detach().unsqueeze(1),
                             values=value.unsqueeze(1), 
                             entropies=entropy.unsqueeze(1)))
    state = next_state
    dm = next_dm
    if terminal:
        # Store metrics and reset environment
        metrics['train_steps'].append(step)
        metrics['train_returns'].append([episode_return])
        pbar.set_description('Step: %i | Return: %f' % (step, episode_return.mean()))
        state, dm = env.reset()
        episode_return = 0
        
        if args.policy_name == 'gat':
            trajectories = {k: torch.stack([d[k] for d in trajectories], dim=0) for k in trajectories[-1].keys()}
        else:
            trajectories = {k: torch.cat([d[k] for d in trajectories], dim=0) for k in trajectories[-1].keys()}
            # time_dim = nr_agents * env.timestep_limit
            # for key in trajectories.keys():
            #     trajectories[key] = trajectories[key].reshape(time_dim, -1)
        
        if (args.policy_name != 'gat' and trajectories['terminals'].shape[0] >= args.batch_size) or \
            (args.policy_name == 'gat' and trajectories['terminals'].shape[0] >= args.batch_size / nr_agents):
            batch_size = args.batch_size if args.policy_name != 'gat' else args.batch_size // nr_agents
            policy_trajectories = trajectories
            trajectories = []  # Clear the set of trajectories
            # Train discriminator and predict rewards
            # Use a replay buffer of previous trajectories to prevent overfitting to current policy
            policy_trajectory_replay_buffer.append(policy_trajectories)
            policy_trajectory_replays = {k: torch.cat([d[k] for d in policy_trajectory_replay_buffer], dim=0) for k in policy_trajectory_replay_buffer[-1].keys()}
            for _ in tqdm(range(args.imitation_epochs), leave=False):
                adversarial_imitation_update(agent, 
                                             discriminator, 
                                             expert_trajectories, 
                                             TransitionDataset(policy_trajectory_replays), 
                                             discriminator_optimiser, 
                                             batch_size, 
                                             args.policy_name, 
                                             args.r1_reg_coeff)
            # Predict rewards
            with torch.no_grad():
                episode_states = policy_trajectories['states']
                episode_actions = policy_trajectories['actions']
                if args.policy_name == 'gat':
                    episode_next_states = torch.cat([policy_trajectories['states'][1:], next_state.unsqueeze(0)])
                else:
                    episode_next_states = torch.cat([policy_trajectories['states'][1:], next_state[-1].unsqueeze(0)])
                episode_policy = policy_trajectories['log_prob_actions'].exp()
                episode_terminals = policy_trajectories['terminals']
                policy_trajectories['rewards'] = discriminator.predict_reward(episode_states, 
                                                                              episode_actions, 
                                                                              episode_next_states, 
                                                                              episode_policy, 
                                                                              episode_terminals)


            # Compute rewards-to-go R and generalised advantage estimates ψ based on the current value function V
            if args.policy_name == 'gat':
                compute_advantages(policy_trajectories, agent(state, tran_adj_coo(dm))[1][0], args.discount, args.trace_decay)
            else:
                compute_advantages(policy_trajectories, agent(state)[1][0], args.discount, args.trace_decay)
            # Normalise advantages
            policy_trajectories['advantages'] = (policy_trajectories['advantages'] - policy_trajectories['advantages'].mean(dim=0)) / (policy_trajectories['advantages'].std(dim=0) + 1e-8)
            #torch.save(policy_trajectories, os.path.join('DEBUG', 'trajectories.pth'))
            
            # Perform PPO updates
            for epoch in tqdm(range(args.ppo_epochs), leave=False):
                ppo_update(agent,
                           policy_trajectories,
                           agent_optimiser,
                           args.ppo_clip,
                           epoch,
                           args.policy_name,
                           args.value_loss_coeff,
                           args.entropy_loss_coeff)


# Save agent and metrics
torch.save(agent.state_dict(), os.path.join('airl_res', 'na_{}-epi_{}-obsm_{}-policy_{}-seed_{}-'.format(env.nr_agents, expert_episodes, env.obs_mode, args.policy_name, args.seed)+'agent.pth'))
if args.imitation in ['AIRL', 'GAIL']: 
    torch.save(discriminator.state_dict(), os.path.join('airl_res', 'na_{}-epi_{}-obsm_{}-policy_{}-seed_{}-'.format(env.nr_agents, expert_episodes, env.obs_mode, args.policy_name, args.seed)+'discriminator_airl.pth'))
torch.save(metrics, os.path.join('airl_res', 'na_{}-epi_{}-obsm_{}-policy_{}-'.format(env.nr_agents, expert_episodes, env.obs_mode, args.policy_name)+'metrics.pth'))
env.close()      


def evaluate_agent(agent, episodes,env):
    t=0
    state, terminal = env.reset(), False
    for i in range(episodes*env.timestep_limit):
        with torch.no_grad():
            policy, value = agent(state)
            action = policy.sample()  # Pick action greedily
            action_num=action.numpy()
            action_num[:,0]=0.5
            next_state, reward, terminal, info = env.step(action.numpy())

            if t % 1 == 0:
                env.render(mode='human')
            state=next_state
            t=t+1
            if terminal:
                state,terminal=env.reset(),False
            
   
    return 0
'''
metri=torch.load(os.path.join('results', 'metrics.pth'))
order_p=metri['train_returns']
for i in range(len(order_p)):
    data[i]=order_p[i][0].mean().item()
plt.plot(data)

'''


def evaluate_agent_dim1(agent, episodes,env):
    ord_par=[]
    
    t=0
    
    state, terminal = env.reset(), False
    for i in range(episodes*env.timestep_limit):
        with torch.no_grad():
            policy,value = agent(state)
            action = policy.sample()  # Pick action greedily
            
 
            
            action_0=np.ones((env.nr_agents,1))*0.5
            action_plus=np.concatenate((action_0,action.numpy()),axis=1)
            
            next_state, reward, terminal,info = env.step(action_plus)
            
            ord_par.append(reward)

            
            if t % 1 == 0:
                env.render(mode='human')
            state=next_state
            t=t+1
            if terminal:
                state,terminal=env.reset(),False
    return ord_par
