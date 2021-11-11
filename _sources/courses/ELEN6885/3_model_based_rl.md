# Model Based RL
For model based RL we need to have full knowledge of all states and the dynamics of the environment: the state transition matrix. Dynamic programming methods operate on sweeps of the states performing a full backup. That means a state is updated based on all possible futures states, the rewards, and their probability of occuring

Bootstrapping: Using estimates of the value to update other estimates of value until the system stabilizes at certain values

Full backups are related to Belleman equations. When convergence has occured the Bellman optimality equation has been satisfied.

## Policy Evaluation
- Iterative evaluation of an environment by updating the value function of states with the expectation of the value function of the next states
  see this notebook[this notebook](https://trevor16gordon.github.io/notes/courses/ELEN6885/rl_balancing_exploration_notebook.html) for example implementation

## Policy Iteration
- The policy can be updated based on the new value function across states. One way to update is using the greedy policy. It can be proven that updating witht he greedy policy will give you a policy that is at least as good as the old policy
see this notebook[this notebook](https://trevor16gordon.github.io/notes/courses/ELEN6885/rl_balancing_exploration_notebook.html) for example implementation

## Value Iteration
- Iterate on q(s, a)
- Need to store more values that just using value iteration
see this notebook[this notebook](https://trevor16gordon.github.io/notes/courses/ELEN6885/rl_balancing_exploration_notebook.html) for example implementation



# Bellman Expectation Equations

The bellman expecation equation for the value function shown below says that the value of a state is the expectation of the total discounted future rewards G given that state.

\begin{align}
\begin{aligned}v_{\pi }\left( s\right) =E_{\pi }\left[ G_{t}|S_{t}=s\right]\end{aligned}
\end{align}

Similarily the bellman expecation equation for the state-value function shown below says that the value of a state action pair is the expectation of the total discounted future rewards G given that state and action.

\begin{align}
\begin{aligned}q_{\pi }\left( s,a\right) =E_{\pi }\left[ G_{t}|S_{t}=S,A_{t}=a\right] \end{aligned}
\end{align}


The bellman expecation equation for the value function shown below says that the value of a state is the immediate reward plus the discounted value of the next state.
\begin{align}
\begin{aligned}V_{\pi }\left( s\right) =E_{\pi }\left[ R_{t+1}+\Upsilon V_{\pi }\left( S_{t+1}\right) |S_{t}=S\right] \end{aligned} 
\end{align}

Similarily the bellman expecation equation for the state-value function shown below says that the value of a state action pair is the immediate reward plus the discounted value of the next state-action.

\begin{align}
\begin{aligned}q_{\pi }\left( s\right) =E_{\pi }\left[ R_{t+1}+\Upsilon q_{-\pi }\left( S_{t+1},A_{t+1}\right) |S_{t}=s,A_{t}=a\right] \end{aligned} 
\end{align}