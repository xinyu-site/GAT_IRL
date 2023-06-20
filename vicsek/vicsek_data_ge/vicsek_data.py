import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from ma_envs.commons.utils import EzPickle
from ma_envs import base

from ma_envs.agents.point_agents.rendezvous_agent import PointAgent
from ma_envs.commons import utils as U
import matplotlib.pyplot as plt
import torch
from utils import flatten_list_dicts2, flatten_swarm_traj,flatten_list_dicts
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

class RendezvousEnv(gym.Env, EzPickle):
    metadata = {'render.modes': ['human', 'animate']}

    def __init__(self, nr_agents=5,
                 obs_mode='sum_obs',
                 repul=5,
                 orie=10,
                 comm_radius=40,
                 world_size=100,
                 distance_bins=16,
                 bearing_bins=8,
                 torus=False,
                 dynamics='unicycle'):
        EzPickle.__init__(self, nr_agents, obs_mode, comm_radius, world_size, distance_bins, bearing_bins, torus, dynamics)
        self.nr_agents = nr_agents
        self.world_size = world_size
        self.obs_mode = obs_mode
        self.world = base.World(world_size, torus, dynamics)
        self.distance_bins = distance_bins
        self.bearing_bins = bearing_bins
        self.torus = torus
        self.dynamics = dynamics
        self.bounding_box = np.array([0., 2 * world_size, 0., 2 * world_size])
        self.comm_radius = comm_radius
        self.repul=repul
        self.orie=orie
        self.reward_mech = 'global'
        self.hist = None
        self.world.agents = [
            PointAgent(self) for _ in
            range(self.nr_agents)
        ]
        # self.seed()

        self.vel_hist = []
        self.state_hist = []
        self.timestep = 0
        self.ax = None



    @property
    def state_space(self):
        return spaces.Box(low=-10., high=10., shape=(self.nr_agents * 3,), dtype=np.float32)

    @property
    def observation_space(self):
        return self.agents[0].observation_space

    @property
    def action_space(self):
        return self.agents[0].action_space

    @property
    def agents(self):
        return self.world.policy_agents

    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    @property
    def timestep_limit(self):
        return 100

    @property
    def is_terminal(self):
        # if (np.max(U.get_upper_triangle(self.world.distance_matrix,
        #                                 subtract_from_diagonal=-1)) < 1
        #     and np.mean([agent.state.p_vel**2 for agent in self.agents]) < 0.1**2)\
        #         or self.timestep >= self.timestep_limit:
        if self.timestep >= self.timestep_limit:
            # if self.ax:
            #     plt.close()
            return True
        else:
            return False

    def get_param_values(self):
        return self.__dict__



    def set_states(self):
        self.timestep = 0
        # self.ax = None

        # self.nr_agents = np.random.randint(2, 10)
        # self.nr_agents = 10
        agent_states = np.random.rand(self.nr_agents, 3)
        agent_states[:, 0:2] = self.world_size * ((0.95 - 0.05) * agent_states[:, 0:2] + 0.05)
        agent_states[:, 2:3] = 2 * np.pi * agent_states[:, 2:3]
        np.random.uniform
        self.world.agent_states = agent_states

        agent_list = [
            PointAgent(self)
            for _ in
            range(self.nr_agents)
        ]

        self.world.agents = agent_list
        self.world.reset()

        nr_agents_sensed = np.sum((0 < self.world.distance_matrix) &
                                  (self.world.distance_matrix < self.comm_radius), axis=1)  # / (self.nr_agents - 1)

        obs = []

        for i, bot in enumerate(agent_list):
            ob = bot.get_observation(self.world.distance_matrix[i, :],
                                     self.world.angle_matrix[i, :],
                                     self.world.angle_matrix[:, i],
                                     np.zeros([self.nr_agents, 2]),
                                     nr_agents_sensed,
                                     i
                                     )
            obs.append(ob)
        obs=obs
        return torch.tensor(obs, dtype=torch.float32)





    def reset(self):
        self.timestep = 0
        # self.ax = None

        # self.nr_agents = np.random.randint(2, 10)
        # self.nr_agents = 10
        agent_states = np.random.rand(self.nr_agents, 3)
        agent_states[:, 0:2] = self.world_size * ((0.95 - 0.05) * agent_states[:, 0:2] + 0.05)
        agent_states[:, 2:3] = 2*np.pi * agent_states[:, 2:3]

        self.world.agent_states = agent_states

        agent_list = [
            PointAgent(self)
            for _ in
            range(self.nr_agents)
        ]

        self.world.agents = agent_list
        self.world.reset()

        nr_agents_sensed = np.sum((0 < self.world.distance_matrix) &
                                  (self.world.distance_matrix < self.comm_radius), axis=1)  # / (self.nr_agents - 1)

        obs = []

        for i, bot in enumerate(agent_list):
            ob = bot.get_observation(self.world.distance_matrix[i, :],
                                     self.world.angle_matrix[i, :],
                                     self.world.angle_matrix[:, i],
                                     np.zeros([self.nr_agents, 2]),
                                     nr_agents_sensed,
                                     i
                                     )
            obs.append(ob)
        obs=obs
        return torch.tensor(obs, dtype=torch.float32)

    def step(self, actions):

        self.timestep += 1

        # assert len(actions) == self.nr_agents
        # print(actions)
        clipped_actions = np.clip(actions[0:self.nr_agents, :], self.agents[0].action_space.low, self.agents[0].action_space.high)

        for agent, action in zip(self.agents, clipped_actions):
            agent.action.u = action

        self.world.step()

        next_obs = []

        velocities = np.vstack([agent.state.w_vel for agent in self.agents])
        # print(self.agents[0].state.p_vel)
        # self.vel_hist.append(velocities)
        nr_agents_sensed = np.sum((0 < self.world.distance_matrix) &
                                  (self.world.distance_matrix < self.comm_radius), axis=1)  # / (self.nr_agents - 1)

        for i, bot in enumerate(self.agents):
            ob = bot.get_observation(self.world.distance_matrix[i, :],
                                     self.world.angle_matrix[i, :],
                                     self.world.angle_matrix[:, i],
                                     velocities,
                                     nr_agents_sensed,
                                     i
                                     )
            next_obs.append(ob)

        rewards = self.get_reward3(actions)

        #done = [self.is_terminal]*self.nr_agents
        done = self.is_terminal
        
        info = {'state': self.world.agent_states, 'actions': actions, 'action_penalty': 0.05 * np.mean(actions**2),
                'velocities': np.vstack([agent.state.p_vel for agent in self.agents])}
       
        next_obs,rewards=torch.tensor(next_obs, dtype=torch.float32),torch.tensor(rewards, dtype=torch.float32)
        return next_obs, rewards, done, info
    
    
    def get_reward2(self, actions):
        a=np.ones(2)*50
        dist_rew=np.linalg.norm(self.world.agent_states[:,0:2]-a*50,axis=1,keepdims=True).mean()
        dist_rew_norm=dist_rew/1000
        action_pen = 0.001 * np.mean(actions**2)
        r = - dist_rew_norm - action_pen
        r = np.ones((self.nr_agents,)) * r
        return r
    
    def get_reward3(self, actions):

        all_distances = U.get_upper_triangle(self.world.distance_matrix, subtract_from_diagonal=-1)
        all_distances_cap = np.where(all_distances > self.comm_radius, self.comm_radius, all_distances)
        all_distances_cap_norm = all_distances_cap / self.comm_radius  # (self.world_size * np.sqrt(2) / 2)
        
        dist_rew = np.mean(all_distances_cap_norm)
        action_pen = 0.001 * np.mean(actions**2)
        r = - dist_rew - action_pen
        r = np.ones((self.nr_agents,)) * r
        # print(dist_rew, action_pen)

        return r

    def render(self, mode='human'):  # , close=True):  check if works with older gym version
        if mode == 'animate':
            output_dir = "video"
            if self.timestep == 0:
                import shutil
                import os

                shutil.rmtree(output_dir)
                os.makedirs(output_dir, exist_ok=True)

        if not self.ax:
            fig, ax = plt.subplots()
            # ax.set_aspect('equal')
            # ax.set_xlim((0, self.world_size))
            # ax.set_ylim((0, self.world_size))
            self.ax = ax
            # self.fig2, self.axes = plt.subplots(1, 2)

        # else:
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.set_xlim((0, self.world_size))
        self.ax.set_ylim((0, self.world_size))
            # [ax.clear() for ax in self.axes]

        # self.ax.add_patch(
        #     patches.Rectangle(
        #         (5, 5),  # (x,y)
        #         self.world_size,  # width
        #         self.world_size,  # height
        #         fill=False
        #     )
        # )

        comm_circles = []
        comm_circles1 = []
        comm_circles2 = []
        self.ax.scatter(self.world.agent_states[:, 0], self.world.agent_states[:, 1], c='b', s=10)
        # self.ax.scatter(self.nodes_all[:, 0], self.nodes_all[:, 1], c='k')
        # self.ax.scatter(self.center_of_mass[0], self.center_of_mass[1], c='g')
        # self.ax.scatter(self.center_of_mass_torus[0], self.center_of_mass_torus[1], c='r')
        # self.ax.plot(self.actors[:, 0], self.actors[:, 1], marker=(3, 0, self.actors[:, 2]), markersize=20, linestyle='None')
        for i in range(self.nr_agents):
            # self.ax.plot(self.actors[i, 0], self.actors[i, 1], marker=(3, 0, self.actors[i, 2]/np.pi*180-90), markersize=20,
            #              linestyle='None', color='g' if i != 0 else 'b')
            comm_circles.append(plt.Circle((self.world.agent_states[i, 0],
                                            self.world.agent_states[i, 1]),
                                           self.comm_radius, color='g' if i != 0 else 'b', fill=False))
            comm_circles1.append(plt.Circle((self.world.agent_states[i, 0],
                                            self.world.agent_states[i, 1]),
                                           self.repul, color='g' if i != 0 else 'b', fill=False))
            comm_circles2.append(plt.Circle((self.world.agent_states[i, 0],
                                            self.world.agent_states[i, 1]),
                                           self.orie, color='g' if i != 0 else 'b', fill=False))

            self.ax.add_artist(comm_circles[i])
            self.ax.add_artist(comm_circles1[i])
            self.ax.add_artist(comm_circles2[i])
            # self.ax.text(self.world.agent_states[i, 0], self.world.agent_states[i, 1],
            #              i, ha='center',
            #              va='center', size=25)
        # circles.append(plt.Circle((self.evader[0],
        #                            self.evader[1]),
        #                           self.evader_radius, color='r', fill=False))
        # self.ax.add_artist(circles[-1])
        # self.axes[0].imshow(self.agents[0].histogram[0, :, :], vmin=0, vmax=10)
        # self.axes[1].imshow(self.agents[0].histogram[1, :, :], vmin=0, vmax=1)
        if mode == 'human':
            plt.pause(0.1)
        elif mode == 'animate':
            if self.timestep % 2 == 0:
                plt.savefig(output_dir + format(self.timestep//2, '04d'))

            if self.is_terminal:
                import os
                os.system("ffmpeg -r 10 -i " + output_dir + "%04d.png -c:v libx264 -pix_fmt yuv420p -y /tmp/out.mp4")

if __name__ == '__main__':
  
    
    
    n_ag =20
    v0=0.5
    noise=0.5
    episodes=2
    ord_p_list=[]
    env = RendezvousEnv(nr_agents=n_ag,
                        obs_mode='fix_acc',
                        comm_radius=100,
                        world_size=100,
                        distance_bins=8,
                        bearing_bins=8,
                        dynamics='unicycle',
                        torus=True)
    trajectories=[]
    for epi in range(episodes):
        o = env.reset()
        t=0
        terminal = False
        flip = -1
        a = np.ones((env.nr_agents,2))
        a[:,0]=v0

        while not terminal:
            for i, agent in enumerate(env.world.policy_agents):   
                ori=[]
                dm=env.world.distance_matrix[i, :]
                in_range = (dm < env.comm_radius) 
                ave=np.arctan2(np.sin(env.world.agent_states[in_range, 2:3]).sum(),np.cos(env.world.agent_states[in_range, 2:3]).sum())
               
                a[i, 1] = ave-agent.state.p_orientation+np.random.uniform(-noise,noise,1)
            
            
            state, rew, terminal, _ = env.step(a)
            ay=np.sin(env.world.agent_states[:, 2:3]).sum()
            bx=np.cos(env.world.agent_states[:, 2:3]).sum()
            ord_p_list.append(np.sqrt(ay*ay+bx*bx)/env.nr_agents)
            t=t+1
            action=torch.tensor(a,dtype=torch.float32)
            trajectories.append(dict(states=state,rewards=torch.tensor(rew, dtype=torch.float32).unsqueeze(1),actions=action[:,1].unsqueeze(1),terminals = torch.tensor([terminal]*env.nr_agents, dtype=torch.float32).unsqueeze(1)))
           
            if t % 1 == 0:
                env.render(mode='human')
    '''           
    plt.plot(np.array(ord_p_list))
    
    trajectories=flatten_list_dicts2(trajectories)
    
    trajectories={k: flatten_swarm_traj(trajectories[k],env.nr_agents,episodes*env.timestep_limit) for k in trajectories.keys()}
    '''
    #torch.save(trajectories,'na_{}-epi_{}-obsm_{}'.format(env.nr_agents, episodes, env.obs_mode)+'vicsek_data.pth') 

    #torch.save(trajectories,'vicsek_data62.pth')    
    '''
  
    #三种规则数据
    
    n_ag = 10
    v0=0.5
    noise=0
    episodes=1
    
    re=5
    orie=10
    attr=20
    
    
    
   
    
    
    env = RendezvousEnv(nr_agents=n_ag,
                        obs_mode='fix_acc',
                        comm_radius=attr,
                        world_size=100,
                        repul=re,
                        orie=orie,
                        distance_bins=8,
                        bearing_bins=8,
                        dynamics='unicycle',
                        torus=True)
    trajectories=[]
    for epi in range(episodes):
        o = env.reset()
        t=0
        terminal = False
        
        a = np.ones((env.nr_agents,2))
        a[:,0]=v0
        
        
        
        orien=0
        while not terminal:
            theta_ij=np.vstack([U.get_angle(env.world.nodes, a[0:2],torus=env.world.torus, world_size=env.world_size)  for a in env.world.agent_states])
            
            for i, agent in enumerate(env.world.policy_agents):   
                ori=[]
                    
                
                target_theta=agent.state.p_orientation
                dm=env.world.distance_matrix[i, :]
               
                
                in_range0 = (dm < env.repul) 
                in_range1 = ((dm > env.repul) &(dm < env.orie))
                in_range2 = ((dm > env.orie) & (dm < env.comm_radius))
                
                if (in_range0.sum()>0):
                
                    rep=theta_ij[i][in_range0].mean()+np.pi
                    target_theta=rep
                    
                    a[i, 1] = target_theta-agent.state.p_orientation+np.random.uniform(-noise,noise,1)
                    
                if (in_range1.sum()>0):
                    
                    orien=np.arctan2(np.sin(env.world.agent_states[in_range1, 2:3]).sum(),np.cos(env.world.agent_states[in_range1, 2:3]).sum())
                    
                    target_theta=orien
                
                if (in_range2.sum()>0):
                    
                    
                    attra=theta_ij[i][in_range2].mean()
                    
                    target_theta=(orien+attra)/2
                 
                a[i, 1] = target_theta-agent.state.p_orientation+np.random.uniform(-noise,noise,1)
               
            
            
                
                
            
            
            state, rew, terminal, _ = env.step(a)
            ay=np.sin(env.world.agent_states[:, 2:3]).sum()
            bx=np.cos(env.world.agent_states[:, 2:3]).sum()
            
            t=t+1
            action=torch.tensor(a,dtype=torch.float32)
            #trajectories.append(dict(states=state,rewards=torch.tensor(rew, dtype=torch.float32).unsqueeze(1),actions=action[:,1].unsqueeze(1),terminals = torch.tensor([terminal]*env.nr_agents, dtype=torch.float32).unsqueeze(1)))
           
            if t % 1 == 0:
                env.render(mode='human')
              
    #plt.plot(np.array(ord_p_list))
     
    trajectories=flatten_list_dicts2(trajectories)
    
    trajectories={k: flatten_swarm_traj(trajectories[k],env.nr_agents,episodes*env.timestep_limit) for k in trajectories.keys()}
    
    torch.save(trajectories,'vicsek_data.pth')    
   
    '''