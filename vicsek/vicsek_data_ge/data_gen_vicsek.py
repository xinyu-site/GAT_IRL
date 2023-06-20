import numpy as np
import argparse
import torch
from ma_envs.envs.point_envs.rendezvous import RendezvousEnv
import warnings
from utils import flatten_list_dicts2
warnings.filterwarnings("ignore")


# parser = argparse.ArgumentParser(description='data_gen')
# parser.add_argument('--policy-name', default='gat', choices=['gat', 'mlp'])
# args = parser.parse_args()

if __name__ == '__main__':

    n_ag = 10
    v0 = 0.5
    noise = 0.5
    episodes = 100
    ord_p_list = []
    env = RendezvousEnv(nr_agents=n_ag,
                        obs_mode='fix_acc',
                        comm_radius=100,
                        world_size=100,
                        distance_bins=8,
                        bearing_bins=8,
                        dynamics='unicycle',
                        torus=True)
    trajectories = []
    for epi in range(episodes):
        o = env.reset()
        t = 0
        terminal = False
        flip = -1
        a = np.ones((env.nr_agents, 2))
        a[:, 0] = v0

        while not terminal:
            for i, agent in enumerate(env.world.policy_agents):
                ori = []
                dm = env.world.distance_matrix[i, :]
                in_range = (dm < env.comm_radius)
                ave = np.arctan2(np.sin(env.world.agent_states[in_range, 2:3]).sum(),
                                 np.cos(env.world.agent_states[in_range, 2:3]).sum())

                a[i, 1] = ave - agent.state.p_orientation + np.random.uniform(-noise, noise, 1)

            state, dis_matrix, rew, terminal, _ = env.step(a)
            ay = np.sin(env.world.agent_states[:, 2:3]).sum()
            bx = np.cos(env.world.agent_states[:, 2:3]).sum()
            ord_p_list.append(np.sqrt(ay * ay + bx * bx) / env.nr_agents)
            t = t + 1
            action = torch.tensor(a, dtype=torch.float32)
            trajectories.append(dict(states=state,
                                     dm=dis_matrix,
                                     rewards=torch.tensor(rew, dtype=torch.float32).unsqueeze(1),
                                     actions=action[:, 1].unsqueeze(1),
                                     terminals=torch.tensor([terminal] * env.nr_agents, dtype=torch.float32).unsqueeze(1)))

            if t % 1 == 0:
                env.render(mode='human')    
    # if args.policy_name == 'gat':
    #     trajectories = {k: torch.stack([d[k] for d in trajectories], dim=0) for k in trajectories[-1].keys()}
    # else:
    #     trajectories = {k: torch.cat([d[k] for d in trajectories], dim=0) for k in trajectories[-1].keys()}
    trajectories = {k: torch.stack([d[k] for d in trajectories], dim=0) for k in trajectories[-1].keys()}
    torch.save(trajectories,'na_{}-epi_{}-obsm_{}'.format(env.nr_agents, episodes, env.obs_mode)+'vicsek_data.pth') 

