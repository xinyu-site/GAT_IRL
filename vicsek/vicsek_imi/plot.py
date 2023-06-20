import os
import numpy as np
import matplotlib.pyplot as plt
import torch
filename='/home/tyk/project/collective_code/vicsek/vicsek_imi/airl_res/na_10-epi_100-obsm_fix_acc-policy_mlp-metrics.pth'
rewards=torch.load(filename)
rewards = rewards['train_returns']
reward=[]
for i in range(len(rewards)):
    reward.append(float(rewards[i][0][0]))
x1 = np.linspace(0, len(reward)-1,len(reward))
x = x1.tolist()
plt.plot(x, reward)
plt.savefig('result_mlp.png')