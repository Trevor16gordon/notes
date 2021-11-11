# Markov Decision Processes 

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

\begin{align}
\pi \left( a|s\right) =P\left[ A_{t}=a|S_{t}=s\right] 
\end{align}

## Greedy Policy
For the greedy policy, always choose the action with the highest expected reward. The drawback to this policy is that an agents estimate of the value of actions is likely not the true value. In this case, it is usefull to explore other states. The following policies provide a way for exploring

## Epsilon Greedy Policy
This policy aims to help balance exploration vs exploitation. With probability (1-e) choose the greedy policy. With probability e/n choose any action.

## Softmax Policy
The softmax policy allows you to tweak the temperature over time to explore less and exploit more as time goes to infinity. 

\begin{align}
\sigma \left( Q(a)\right) =\dfrac{e^{\dfrac{Q(a)}{\tau}}}{\sum ^{k}_{j=1}e^{\dfrac{Q(a)}{\tau};}}
\end{align}

In the limit as the temperature τ → 0, softmax action selection becomes the same as greedy action selection. 
Proof:

\begin{align}
\begin{aligned}\lim _{\tau \rightarrow \infty }P\left( a\right) =\dfrac{\exp \left( Q_{t}\left( a\right) /\tau \right) }{\Sigma _{i=1}^{n}\exp \left( Q_{t}\left( i\right) /\tau \right) }\\
\lim _{\tau \rightarrow \infty }P\left( a\right) =\dfrac{\exp \left( Q_{t}\left( a\right) /\infty \right) }{\sum ^{n}_{i=1}\exp \left( Q_{t}\left( i\right) /\infty \right) }\\
\lim _{t\rightarrow \infty }P\left( a\right) =\dfrac{1}{n}\end{aligned}
\end{align}

In the limit as τ → ∞, softmax action selection yields equiprobable selection among all actions.
Proof:

\begin{align}
\begin{aligned}p\left( a=1\right) =\dfrac{e^{Q_{t}\left( 1\right) /\tau }}{e^{Q_{t}\left( 1\right) /\tau }+e^{Q_{t}\left( 2\right) /\tau }}\\
\dfrac{1}{Q_{t}\left( 2\right) /\tau }\\
P\left( a=1\right) =1+\dfrac{e}{e^{Qt}}\dfrac{}{\left( 1\right) /\tau }\\
P\left( a=1\right) =\dfrac{1}{1+e^{\dfrac{1}{T}\left( Q+121-Q+\left( 1\right) \right) }}\end{aligned}
\end{align}