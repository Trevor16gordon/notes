## Bandit Problems and MDP Intro




# Markov
A markov process is a stochastic process that is memoryless. That means that the future only depends on the current state, not on the history.

## Markov Reward Processes
A markov reward process has
**States**: Finite set of states
**Rewards**: Rs is the expecation of awards for a given state
**State Transition Prob Matrix**: Probability of moving to the other states given a current state
**Discount Factor**: How much to weigh current rewards vs future ones. In range [0,1].


## Markov Decision Processes
A markov decision process is an extension of the MRP where the agent now has a set of actions.
**Actions**: Finite set of actions
**States**: Finite set of states
**Rewards**: Rs is the expecation of awards for a given state
**State Transition Prob Matrix**: Probability of moving to the other states given a current state and current action
**Discount Factor**: How much to weigh current rewards vs future ones. In range [0,1].


# Policy
A policy's value function assigns a value to each state or to each state, action pair. There is one optimal value function but many possible optimal policies
