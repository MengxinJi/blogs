---
layout: post
title: "Actor Critic"
description: ""
date: 2019-04-08
tags: OpenAI, Reinforcement Learning, Model-Free RL, Policy Optimization, Notebook
comments: true
---

---
Introduction Paper: [Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf)

Authors: Vijay R. Konda John N. Tsitsiklis

___


### Actor Critic

In this post, I will introduce the another policy gradient algorithm: Actor-Critic method. In general, there are two main components in policy gradient: policy model, and value function. Value function can help policy update, e.g., reducing gradient variance in vanilla policy gradients. In particular, we use Critic layer to approximately estimate value function:

![value](https://mengxinji.github.io/Blog/images/actorcritic/value.svg){: width=120 height=50 style="float:center; padding:16px"}

To introduce Actor-Critic method, we separate them into two models, which may share parameters optionally:

* `Critic Layer`: updates the value function parameters w, it could be actor-value Q or state-value V, depending on the algorithm;

* `Actor Layer`: updates the policy parameters, in the direction suggested by the critic. 

Hence, we may consider Actor-Critic method as approximate policy gradient.
![gradient](https://mengxinji.github.io/Blog/images/actorcritic/gradient.svg){: width=120 height=40 style="float:center; padding:16px"}

In this post, I introduce a simple action-value actor-critic method procedure: 

<p align="left">
  <img width="600" height="250" src="https://mengxinji.github.io/Blog/images/actorcritic/update.svg">
</p>

As mentioned earlier, this algorithm actually uses an approximate policy gradient, which will introduce bias, and lead to fail to converge to a suitable policy. One solution is to design Q-value that Simultaneously satisfies the following two conditions (Compatible Function Approximation Theorem):

* The gradient of the approximation value function is exactly equivalent to the gradient of the logarithm of the strategy function: 
![Q_logP](https://mengxinji.github.io/Blog/images/actorcritic/Q_logP.svg){: width=120 height=40 style="float:center; padding:16px"}

* The value function parameter w minimizes the mean square error:
![TDsq.svg](https://mengxinji.github.io/Blog/images/actorcritic/TDsq.svg){: width=120 height=40 style="float:center; padding:12px"}

`Short Proof`: If the Q-value satisfies the above two conditions, then we have: 
<p align="center">
  <img width="400" height="180" src="https://mengxinji.github.io/Blog/images/actorcritic/proof.svg">
</p>

### Actor Critic with Baseline

In addition to introducing the Critic layer to reduce the variance, we also consider subtracting a baseline from Q function to reduce the variance. Specifically, we require the baseline function B subtracted from the policy gradient,  only to be related to the state, and has nothing to do with the behavior, so that the gradient itself is not changed.

A good choice of B(s) is value function V(s). Hence, we introduce an advantage function A(s, a) with the following definition: 
![advantage](https://mengxinji.github.io/Blog/images/actorcritic/advantage.svg){: width=120 height=40 style="float:center; padding:16px"}

Usually we use TD error to estimate the advantage function, because it is an unbiased estimator of advantage function.
<p align="center">
  <img width="500" height="150" src="https://mengxinji.github.io/Blog/images/actorcritic/TDerror.svg">
</p>

Hence, we have:
 ![gradient2](https://mengxinji.github.io/Blog/images/actorcritic/gradient2.svg){: width=120 height=80 style="float:center; padding:16px"}



### Example of Actor Critic Using [cartpool](https://gym.openai.com/envs/CartPole-v0/) From OpenAI Gym


```python
import gym, os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
```


```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("CartPole-v0").unwrapped
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
lr = 0.0001
```


```python
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value
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
def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def trainIters(actor, critic, n_iters):
    optimizerA = optim.Adam(actor.parameters())
    optimizerC = optim.Adam(critic.parameters())
    for iter in range(n_iters):
        state = env.reset()
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0
        env.reset()

        for i in count():
            #env.render()
            state = torch.FloatTensor(state).to(device)
            dist, value = actor(state), critic(state)
            action = dist.sample()
            next_state, reward, done, _ = env.step(action.cpu().numpy())

            log_prob = dist.log_prob(action).unsqueeze(0)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))
            
            state = next_state

            if done:
                #print('Iteration: {}, Score: {}'.format(iter, i))
                episode_durations.append(i + 1)
                plot_durations() 
                break

        next_state = torch.FloatTensor(next_state).to(device)
        next_value = critic(next_state)
        returns = compute_returns(next_value, rewards, masks)
        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        optimizerA.zero_grad()
        optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizerA.step()
        optimizerC.step()
    #env.close()
```


```python
actor = Actor(state_size, action_size).to(device)
critic = Critic(state_size, action_size).to(device)
trainIters(actor, critic, n_iters=500)
```



![optimizer](https://mengxinji.github.io/Blog/images/actorcritic/optimize.png){: width=150 height=100 style="float:bottom; padding:16px"}






