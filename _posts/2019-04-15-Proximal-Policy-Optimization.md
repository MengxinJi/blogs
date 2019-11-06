---
layout: post
title: "Proximal Policy Optimization"
description: ""
date: 2019-04-15
tags: [OpenAI, Reinforcement Learning, Policy Optimization, Notebook]
comments: true
---

---
Introduction Paper: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

Authors: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov

Introduction Paper: [Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/abs/1707.02286)

Authors: Nicolas Heess, Dhruva TB, Srinivasan Sriram, Jay Lemmon, Josh Merel, Greg Wayne, Yuval Tassa, Tom Erez, Ziyu Wang, S. M. Ali Eslami, Martin Riedmiller, David Silver

___


In July 2017, DeepMind and OpenAI post articles on PPO (Proximal Policy Optimization) on arXiv respectively, i.e., OpenAI's "[Proximal Policy Optimization Algorithms]((https://arxiv.org/abs/1707.06347))" and DeepMind's "[Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/abs/1707.02286)". PPO is usually considered to be the approximate algorithm of TRPO (Trust Region Policy Optimization), which is more adaptable to large-scale operations. DeepMind's article also proposed Distributed PPO for distributed training. In this post, I will start with TRPO.


### Trust Region Policy Optimization

TRPO was proposed due to the idea that we should avoid parameter updates that change the policy too much at one step so as to improve training stability. Hence, TRPO takes this into consideration by enforcing a KL divergence constraint on the size of policy update at each iteration. In previous literature, suppose the strategy is controlled by the parameter, and the goal of each optimization is to find the divergence within a certain range:

<p align="center">
  <img width="300" height="120" src="https://mengxinji.github.io/Blog/images/ppo/klobj.svg">
</p>
  
Removing some complicated procedure, TRPO uses the following approximates:

<p align="center">
  <img width="320" height="210" src="https://mengxinji.github.io/Blog/images/ppo/approximate.svg">
</p>

Hence, the objective function becomes the following form from TRPO algorithm. In particular, we can still separate the algorithm procedure into off-policy and on-policy:

* If off-policy, the objective function measures the total advantage over the sate visitation distribution and actions, while the rollouts is following a different behavior policy distribution; 
<p align="left">
  <img width="450" height="200" src="https://mengxinji.github.io/Blog/images/ppo/obj.svg">
</p>
* If on-policy, the behavior policy is the previous policy.
![onpolicy](https://mengxinji.github.io/Blog/images/ppo/onpolicyobj.svg){: width=120 height=40 style="float:center; padding:16px"}

As introduced above, TRPO aims to maximize the objective function subject to, trust region constraint which enforces the distance between old and new policies measured by KL-divergence to be small enough, within a parameter:
![constraint](https://mengxinji.github.io/Blog/images/ppo/klconstraint.svg){: width=120 height=40 style="float:center; padding:16px"}



### Proximal Policy Optimization

PPO can be viewed as an approximation of TRPO, but unlike TRPO, which uses a second-order Taylor expansion, PPO uses only a first-order approximation, which makes PPO very effective in RNN networks and in a wide distribution space.
<p align="center">
  <img width="450" height="500" src="https://mengxinji.github.io/Blog/images/ppo/ppo.svg">
</p>

The first half of Estimate Advantage is obtained through the rollout strategy, and the second half of V is obtained from a value network. (Value network can be trained by the data obtained by rollout, where the mean square error is used).

Here, a > 1, when KL divergence is greater than expected, it will increase the weight of KL divergence in J(PPO) to reduce KL divergence. In this way the control training is maintained within a certain KL divergence change.

When updating Actors, there are actually two ways, one is to update with the KL penalty as we discussed earlier.
![KL](https://mengxinji.github.io/Blog/images/ppo/KLpenalty.svg){: width=120 height=40 style="float:center; padding:16px"}

There is also a clipped surrogate objective, mentioned from OpenAI's PPO paper.
![KL](https://mengxinji.github.io/Blog/images/ppo/Clip.svg){: width=120 height=40 style="float:center; padding:16px"}




### Example of PPO Using [LunarLander](https://gym.openai.com/envs/LunarLander-v2/) From OpenAI Gym


```python
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym, os
from itertools import count
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```


```python
class Model(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(Model, self).__init__()
        self.affine = nn.Linear(state_dim, n_latent_var)
        
        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim = -1)
                )
        
        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 1)
                )
        
        # Memory:
        self.actions = []
        self.states = []
        self.logprobs = []
        self.state_values = []
        self.rewards = []
        
    def forward(self, state, action=None, evaluate=False):
        # if evaluate is True then we also need to pass an action for evaluation
        # else we return a new action from distribution
        if not evaluate:
            state = torch.from_numpy(state).float().to(device)
        
        state_value = self.value_layer(state)
        
        action_probs = self.action_layer(state)
        action_distribution = Categorical(action_probs)
        
        if not evaluate:
            action = action_distribution.sample()
            self.actions.append(action)
            
        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)
        
        if evaluate:
            return action_distribution.entropy().mean()
        
        if not evaluate:
            return action.item()
        
    def clearMemory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]
```


```python
class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = Model(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                              lr=lr, betas=betas)
        self.policy_old = Model(state_dim, action_dim, n_latent_var).to(device)
        
        self.MseLoss = nn.MSELoss()
        
    def update(self):   
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward in reversed(self.policy_old.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list in tensor
        old_states = torch.tensor(self.policy_old.states).to(device).detach()
        old_actions = torch.tensor(self.policy_old.actions).to(device).detach()
        old_logprobs = torch.tensor(self.policy_old.logprobs).to(device).detach()
        

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            dist_entropy = self.policy(old_states, old_actions, evaluate=True)
            # Finding the ratio (pi_theta / pi_theta__old):
            logprobs = self.policy.logprobs[0].to(device)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            

            # Finding Surrogate Loss:
            state_values = self.policy.state_values[0].to(device)
            advantages = rewards - state_values.squeeze().detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
            self.policy.clearMemory()
            
        self.policy_old.clearMemory()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
```


```python
############## Hyperparameters ##############
env_name = "LunarLander-v2"
#env_name = "CartPole-v1"
# creating environment
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = 4

render = False
log_interval = 10
n_latent_var = 64           # number of variables in hidden layer
n_update = 2              # update policy every n episodes
lr = 0.0007
betas = (0.9, 0.999)
gamma = 0.99                # discount factor
K_epochs = 5                # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
random_seed = None
#############################################

if random_seed:
    torch.manual_seed(random_seed)
    env.seed(random_seed)

ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
print(lr,betas)
```

```python
# Plot duration curve: 
# From http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
episode_durations = []
def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
```


```python
running_reward = 0
avg_length = 0
for i_episode in range(1, 11):
    state = env.reset()
    for t in range(100): #10000
        # Running policy_old:
        action = ppo.policy_old(state)
        state_n, reward, done, _ = env.step(action)

        # Saving state and reward:
        ppo.policy_old.states.append(state)
        ppo.policy_old.rewards.append(reward)
        
        state = state_n

        running_reward += reward
        if render:
            env.render()
        if done:
            #print(i_episode, t)
            episode_durations.append(t + 1)
            plot_durations()
            break
    
    avg_length += t
    # update after n episodes
    if i_episode % n_update == 0:

        ppo.update()

    # log
    if running_reward > (log_interval*200):
        print("########## Solved! ##########")
        torch.save(ppo.policy.state_dict(), 
                   './LunarLander_{}_{}_{}.pth'.format(
                    lr, betas[0], betas[1]))
        break

    if i_episode % log_interval == 0:
        avg_length = int(avg_length/log_interval)
        running_reward = int((running_reward/log_interval))

        print('Episode {} \t avg length: {} \t reward: {}'.format(
                i_episode, avg_length, running_reward))
        running_reward = 0
        avg_length = 0
```

![optimizer](https://mengxinji.github.io/Blog/images/ppo/optimize.png){: width=150 height=100 style="float:bottom; padding:16px"}



