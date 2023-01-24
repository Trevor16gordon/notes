# Model Free RL
In most real world use cases we do not know the dynamics of the environment so we can't use a model based reinforcement learning approach. Model free RL get's around this by learning from episodes of experience.

## Monte Carlo Learning
- Learn from episodes of experience either directly with the environment or with a simulater
- Update the value of states by average the total future reward of that state. By law of large numbers this should converge to the true value
- You can focus Monte Carlo learning on a small subset of states rather than learning all possible states.
- Advantage of Monte Carlo learning is that they may be less harmed by violations to the Markov property. They don't bootstrap

## Monte Carlo Control
- You can start with even likilhood of all actions at each state
- After running through episodes you can update your policy based on your estimate of future rewards for all states.
- Maintaining sufficient exploration is an issue with Monte Carlo learning/control. If you follow a greedy policy you are not gaurunteed to explore all states.

$
V\left( s_{t}\right) \leftarrow V\left( s_{t}\right) +\dfrac{1}{N\left( s_{t}\right) }\left( G_{t}-V\left( s_{t}\right) \right) 
$

## On and Off Policy Control
- Normally you are learning based on episodic experience from your own policy, and then iterating on that policy. 
- In many cases you may have data collected using some policy that wasn't your own
- Off policy control is using data from another policy to update your policy. 
  - Importance Sampling: The samples are weighted by the ratio of the two policies choosing that action.


## TD Learning
- For TD learning we don't perform a "Full Backup" on the data. We update the value of states only using the next state. 

## Sarsa

$$
Q\left( S,A\right) \leftarrow Q\left( S,A\right) +\alpha \left( R+\Upsilon Q\left( S',A'\right) -Q\left( S,A\right) \right) 
$$


## Q Learning

$$
 Q\left( S,A\right) \leftarrow Q\left( S,A\right) +\alpha \left( R+\Upsilon \max _{a\cdot }Q\left( S',A'\right) -Q\left( S,A\right) \right) 
 $$

