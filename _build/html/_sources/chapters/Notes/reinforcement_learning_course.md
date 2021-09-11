# Reinforcement Learning Course


## Overview of Reinforcement Learning
Reinforcement learning (RL) is a type of machine learning that aims to resemble how humans learn. RL agents learn by interact with their agent and modifying their actions based on feedback from their experience.

What makes RL different from other machine learning paradigms
- No supervisor, only reward signal
- Feedback is delayed
- Time matters, sequential decision making

Applications: RL has applications in any problems where the environment is dynamic and where rigid rules based decision making won't suffice.

### Elements of RL

Policy is a mapping from percieved states of the environment to the probability action space $S$.

\begin{align} s\in S\rightarrow \pi \left( a|s\right)\end{align}

Reward is a mapping from each percieved state of the environment to a single number indicating the intrinsic desirability of the space.

Return is a cumulative sequence of recieved rewards after a given timestep.

Finite state return:
\begin{align}(G_{t}=R_{t+1}+R_{t+2}+R_{t+3}+\ldots +R_{T})\end{align}

Discounted Return: 
\begin{align}
0\leq \Upsilon \leq 1
G_{T}=\sum ^{\infty }_{k=0}\gamma ^{k}R_{t+k+1}\end{align}

Value Function is an estimate of how good it is to be in a specific state. For a Markov Decision Process (MDP) the value of a state is defined formally as:

\begin{align}v_{\pi }\left( s\right) =E_{\pi }\left[ G_{t}|S_{T}=S\right] =E_{\pi }\left[ \sum ^{\infty }_{k:0}\Upsilon ^{k}R_{t+k+1}|S_{t}=S\right]\end{align}

For an MDP the value of an action state pair is

\begin{align}q_{\pi }\left( s, a\right) =E_{\pi }\left[ G_{t}|S_{T}=S, A_{T}=a\right] =E_{\pi }\left[ \sum ^{\infty }_{k:0}\Upsilon ^{k}R_{t+k+1}|S_{t}=S, A_{T}=a\right]\end{align}

The Model is used to mimic the environment and is a translation from the current state and desired action to resulting state.

\begin{align}P_{ss'}^{a}=P\left[ S_{t+'}=S^{'}|S_{t}=s,A_{t}=a\right]\end{align}


