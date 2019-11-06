---
layout: post
title: "Notebook: DQN"
description: ""
date: 2019-03-09
tags: [Reinforcement Learning, Algorithm, OpenAI, Notebook]
comments: true
---

## Deep Q-Network

### Algorithm deep Q-learning with experience replay
- Initialize replay memory D to capacity N
- Initialize action-value function Q with random weights $\theta$
- Initialize target action-value function $\hat{Q}$ with weights $\overline{\theta} = \theta$
- For episode = 1 - M do:
    - Initialize sequence s1 = {x1} and preprocessed sequance $\phi 1 = \phi(s1)$
    - For t = 1 - T, do:
        - With probability $\epsilon$ select a random action at
        - otherwise select $at = argmax(a)Q(\phi(st), a; \theta)$
        - Execute action at in emulator and observe reward rt and image x(t+1)
        - Set s(t+1) = st , at, x(t+1) and prepocess \phi(t+1) = \phi(s(t+1))
        - Store transition (\phi(t), at, rt, \phi(t + 1)) in D
        - Sample random minibatch of transitions from D
        - Set y_j = r_j if episode terminates at step j+1; else r_j + \gamma max_{a'}\hat{Q}(\phi_{j+1}, a'; \overline{\theta})
        - Perform a gradient descent step on (y_j - Q(\theta_j, a_j; \theta))^2 with respect to the network parameters \theta
        - Every C steps reset \hat{Q} = Q
    - End For
- End For

To simplify, we may consider adding something on Q-learning, including:
- Memory
- Use network to calculat Q value
- Fixed Q-targets


```python
import argparse
import pickle
from collections import namedtuple
from itertools import count

import os, time
import numpy as np
import matplotlib.pyplot as plt

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter
```


```python
# Hyper-parameters
seed = 1
render = False
num_episodes = 500
env = gym.make('MountainCar-v0').unwrapped
num_state = env.observation_space.shape[0]
num_action = env.action_space.n
torch.manual_seed(seed)
env.seed(seed)

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])
```

### Define Network


```python
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.fc2 = nn.Linear(100, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = self.fc2(x)
        return action_prob
    '''
    def __init__(self, h, w):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc2 = nn.Linear(32, num_action)
        
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, 2) # 448 or 512

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        action_prob = self.fc2(x)
        return action_prob
        #return self.head(x.view(x.size(0), -1))
     '''
```


```python
class DQN():

    capacity = 8000
    learning_rate = 1e-3
    memory_count = 0
    batch_size = 256
    gamma = 0.995
    update_count = 0

    def __init__(self):
        super(DQN, self).__init__()
        self.target_net, self.act_net = Net(), Net()
        self.memory = [None]*self.capacity
        self.optimizer = optim.Adam(self.act_net.parameters(), self.learning_rate)
        self.loss_func = nn.MSELoss()
        self.writer = SummaryWriter('./DQN/logs')
        self.cost_his = []

    def select_action(self,state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        value = self.act_net(state)
        action_max_value, index = torch.max(value, 1)
        action = index.item()
        if np.random.rand(1) >= 0.9: # epslion greedy
            action = np.random.choice(range(num_action), 1).item()
        return action

    def store_transition(self,transition):
        index = self.memory_count % self.capacity
        self.memory[index] = transition
        self.memory_count += 1
        return self.memory_count >= self.capacity

    def update(self):
        if self.memory_count >= self.capacity:
            state = torch.tensor([t.state for t in self.memory]).float()
            action = torch.LongTensor([t.action for t in self.memory]).view(-1,1).long()
            reward = torch.tensor([t.reward for t in self.memory]).float()
            next_state = torch.tensor([t.next_state for t in self.memory]).float()

            reward = (reward - reward.mean()) / (reward.std() + 1e-7)
            with torch.no_grad():
                target_v = reward + self.gamma * self.target_net(next_state).max(1)[0]

            #Update...
            for index in BatchSampler(SubsetRandomSampler(range(len(self.memory))), batch_size=self.batch_size, drop_last=False):
                v = (self.act_net(state).gather(1, action))[index]
                loss = self.loss_func(target_v[index].unsqueeze(1), (self.act_net(state).gather(1, action))[index])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar('loss/value_loss', loss, self.update_count)
                self.cost_his.append(loss)
                self.update_count +=1
                if self.update_count % 100 ==0:
                    self.target_net.load_state_dict(self.act_net.state_dict())
        else:
            print("Memory Buff is too less")
    
    
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
        
    
def main():

    agent = DQN()
    for i_ep in range(num_episodes):
        state = env.reset()
        if render: env.render()
        for t in range(10000):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            if render: env.render()
            transition = Transition(state, action, reward, next_state)
            agent.store_transition(transition)
            state = next_state
            if done or t >=9999:
                agent.writer.add_scalar('live/finish_step', t+1, global_step=i_ep)
                agent.update()
                if i_ep % 10 == 0:
                    print("episodes {}, step is {} ".format(i_ep, t))
                break
    agent.plot_cost()
```


```python
if __name__ == '__main__':
    main()
```


