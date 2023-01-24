# Planning and Learning

## Monte Carlo Tree Search


- Monte carlo tree search is made up 4 steps: 1) Selection, 2) Expansion 3) Simulation 4) Back Propogation
- Requires a model of the environment: That is, from a current state $s$ and action $a$, the model (or an appoximation of the model paramaterized by $\psi$) for predicting the reward is known $r_{t+1} = \mathbf{R_{\psi}}(r_{t+1} | a_i, s_i)$, and a model for the distribution on next states $s_{t+1} \sim \mathbf{p_{\psi}}(s_{t+1} | a_i, s_i)$,
- With a modle of the environment, we want to select actions to balance exploration and exploitation

### 1 Action Selection
- Use Upper Confidence Bound (UCB) to select actions to minimize regret
- For a given state $s$, choose the action to minimize regret. Regret is the number of times that the non optimal action has been selected. Keep bounds on the value of each action. 

$
UCB_1 = \bar{X_j} + C \cdot \sqrt{\frac{\ln n}{n_j}}
$

where $\bar{X_j}$ is the current estimate of the value for the action, C is a constant (ex: $\sqrt 2$), $n$ is the number of times an action has been chosen for this state. $n_j$ is the number of times the jth action has been chosen. 
- Note that the confidence bound decreases as the action has been tried more.

### 2 Expansion


### 3 Simulation
- Simulation or roll out actions to see how we perform from this state
- May be random from this case to improve speed
- Run many times to see if we won (Or a new estimate of winning from this point)


### 4 Back Propogation
- Update records for all of the nodes in the path of the simulation
- Update counts / current estimates


## Pros and Cons
Pros
- Tree growth focuses on promising areas
- Avoid the problem of globally approximating an action-value

Cons
- Memory intensive. Need to keep the size of the entire tree in memory


