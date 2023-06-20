import os

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch



def flatten_swarm_traj(a,agent_num,timestep_limit):
    
    row=[]
    out=[]
    b=torch.chunk(a,agent_num,dim=0)    
    for i in b:
        row.append(torch.chunk(i,timestep_limit,dim=1))
       
    for i in range(agent_num):   
        out.append(torch.cat(row[i],dim=0))
    out=tuple(out)
    
    return torch.cat(out,dim=0) 

def flatten_list_dicts(list_dicts):    
  return {k: torch.cat([d[k] for d in list_dicts], dim=1) for k in list_dicts[-1].keys()}


# Makes a lineplot with scalar x and statistics of vector y
def lineplot(x, y, filename, xaxis='Steps', yaxis='Returns'):
  y = np.array(y)
  y_mean, y_std = y.mean(axis=1), y.std(axis=1)
  sns.lineplot(x, y_mean, color='coral')
  plt.fill_between(x, y_mean - y_std, y_mean + y_std, color='coral', alpha=0.3)
  plt.xlim(left=0, right=x[-1])
  plt.ylim(bottom=0, top=500)  # Return limits for CartPole-v1
  plt.xlabel(xaxis)
  plt.ylabel(yaxis)
  plt.savefig(os.path.join('results', filename + '.png'))
  plt.close()



def flatten_list_dicts2(list_dicts):    
  return {k: torch.cat([d[k] for d in list_dicts], dim=1) for k in list_dicts[-1].keys()}

def flatten_swarm_traj2(a,agent_num,timestep_limit):
    c={k: torch.cat([d[k] for d in a], dim=1) for k in a[-1].keys()}
    row=[]
    out=[]
    b=torch.chunk(c,agent_num,dim=0)    
    for i in b:
        row.append(torch.chunk(i,timestep_limit,dim=1))
       
    for i in range(agent_num):   
        out.append(torch.cat(row[i],dim=0))
    out=tuple(out)
    out=torch.cat(out,dim=0) 
    
    return {k: flatten_swarm_traj(out[k],agent_num,timestep_limit) for k in out.keys()}