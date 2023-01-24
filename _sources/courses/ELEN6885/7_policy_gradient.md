# Policy Gradients


\begin{aligned}v_{\pi }\left( s\right) =E_{\pi }\left[ G_{t}|S_{t}=s\right]\end{aligned}



Policy gradient algorithms in reinforcement learning directly parameterize the policy function $\pi_{\theta}(s|a)$ and optimize the parameters $\theta$ either directly using gradient ascent or while optimizing some other objective $J(\theta)$
- The key idea is to increase the probability of actions which lead to the highest returns and to push down the actions that lead to lower returns


What should be set $J(\theta)$ to be?
- We might consider the expected value of the first state in an episodic environment:
$\begin{aligned}J_1(\theta)=\mathbf{E}[V^{\pi_{\theta}}(s_1)]\end{aligned}$
- In continuing environment we can use the average value per time step
$J_2(\theta)=\sum_{s}d^{\pi_{\theta}} (s) \sum_a \pi_{\theta}(s|a) R$ where $d^{\pi_{\theta}}$ is a distribution on states
- Now taking the gradient and rearranging
$\begin{matrix*}[l]
\nabla_{\theta} J_2(\theta)=\nabla_{\theta} \left( \sum_{s}d^{\pi_{\theta}} (s) \sum_a \pi_{\theta}(s|a) R \right)\\
\nabla_{\theta} J_2(\theta)= \sum_{s}d^{\pi_{\theta}} (s)  \sum_a \nabla_{\theta} \left( \pi_{\theta}(s|a)\right) R \\
\nabla_{\theta} J_2(\theta)= \sum_{s}d^{\pi_{\theta}} (s)  \sum_a \frac{\pi_{\theta}(s|a)}{\pi_{\theta}(s|a)} \nabla_{\theta} \left( \pi_{\theta}(s|a)\right) R \\
\nabla_{\theta} J_2(\theta)= \sum_{s}d^{\pi_{\theta}} (s)  \sum_a \pi_{\theta}(s|a) \nabla_{\theta} \left( \ln \pi_{\theta}(s|a)\right) R \\
\nabla_{\theta} J_2(\theta)= \mathbf{E} [\nabla_{\theta} \ln \pi_{\theta}(s|a) R ]\\
\end{matrix*}
$
- Leads us to to the score function is $\nabla_{\theta} \ln \pi_{\theta}(s|a)$
- The policy gradient theorm says for the objectives (start state reward, average reward, average value reward), we can replace r with $Q^{\pi_{\theta}}(s,a)$
- Now Monte carlo policy gradient (REINFORCE) algorithm is simply using the return $v_t$ as an unbiased estimate of $Q^{\pi_{\theta}}(s,a)$
- We can have a neural network paramaterized with weights $\theta$ but we need to make sure the sum over probabilities sum to 1.
- One possibility is the softmax policy: 
$\begin{matrix*}[l]
\nabla_{\theta} \ln \pi_{\theta}(s|a) = \nabla_{\theta} \ln \frac{e^{\phi(s,a)^T \theta}}{\sum_a e^{\phi(s,a)^T \theta}}\\
\nabla_{\theta} \ln \pi_{\theta}(s|a) = \phi(s,a) - \textbf{E}[\phi (s,\cdot)]\\
\end{matrix*}
$

- Many other improvements to policy gradient algorithms are intended to reduct the variance.


Reinforce
- Just uses the straight up reward
- Training at the end of the episode
\theta = \theta + lr * gamma_discount * reward * log(gradient(output_prob)) / output_prob


Actor Critic
- Use another neural network to approximate the value function $Q_{W}(s,a) \approx Q^{\pi_{\theta}}(s,a)$
- This approximates the policy gradient
- Update $Q_{W}(s,a)$ based on the TD error
- Actor is policy network
    - Actor is trained by taking gradient steps in the directin of positive suprise
    - It in increases probabilities of states that exceeded their expectations
    - It in decreases probabilities of states that were below their expectations
- Critic is value network (tells it if it did well or not)


Reducing Variance Using a Baseline
- Problem is that $Q^{\pi_{\theta}}(s,a)$ may have very high variance
- We can try to reduce the variance by subracting a baseline that doesn't change the expectation, but reducing variance.
- Baseline $(B(s))$ that is action independent
- 
$
\begin{matrix*}[l]
\nabla_{\theta} J = \textbf{E}_{\pi_{\theta}}[\nabla_{\theta}\log \pi_{\theta}(s,a) \cdot (Q^{\pi_{\theta}}(s,a) - B(s))]\\
\nabla_{\theta} J = \textbf{E}_{\pi_{\theta}}[\nabla_{\theta}\log \pi_{\theta}(s,a) \cdot Q^{\pi_{\theta}}(s,a)] - \textbf{E}_{\pi_{\theta}}[\nabla_{\theta}\log \pi_{\theta}(s,a) \cdot B(s)]\\
\\
\textbf{E}_{\pi_{\theta}}[\nabla_{\theta}\log \pi_{\theta}(s,a) \cdot B(s)]\\
= \sum_{s \in S} d^{\pi_{\theta}}(s) \sum_{a \in A} \nabla_{\theta}  \pi_{\theta}(s|a) \cdot B(s)\\
= \sum_{s \in S} d^{\pi_{\theta}}(s) B(s)\sum_{a \in A} \nabla_{\theta}  \pi_{\theta}(s|a) \\
= \sum_{s \in S} d^{\pi_{\theta}}(s) B(s)\nabla_{\theta} \sum_{a \in A}   \pi_{\theta}(s|a) \\
= \sum_{s \in S} d^{\pi_{\theta}}(s) B(s)\nabla_{\theta} (1) \\
= 0\\
\therefore 
\nabla_{\theta} J = \textbf{E}_{\pi_{\theta}}[\nabla_{\theta}\log \pi_{\theta}(s,a) \cdot (Q^{\pi_{\theta}}(s,a) - B(s))] =  \textbf{E}_{\pi_{\theta}}[\nabla_{\theta}\log \pi_{\theta}(s,a) \cdot Q^{\pi_{\theta}}(s,a)]\\\\
\end{matrix*}
$

Here are some more examples of approximate policy gradient algorithms:

$
\begin{matrix*}[l]
\nabla_{\theta} J && = \textbf{E}_{\pi_{\theta}}[\nabla_{\theta}\log \pi_{\theta}(s,a) \cdot \color{red} v_t \color{black}] && \textrm{Reinforce}\\
\nabla_{\theta} J && = \textbf{E}_{\pi_{\theta}}[\nabla_{\theta}\log \pi_{\theta}(s,a) \cdot \color{red} Q^W(s,a) \color{black}] && \textrm{Q actor critic}\\
\nabla_{\theta} J && = \textbf{E}_{\pi_{\theta}}[\nabla_{\theta}\log \pi_{\theta}(s,a) \cdot \color{red} A^W(s,a) \color{black}] && \textrm{Advantage Actor Critic}\\
\nabla_{\theta} J && = \textbf{E}_{\pi_{\theta}}[\nabla_{\theta}\log \pi_{\theta}(s,a) \cdot \color{red} \delta \color{black}] && \textrm{TD Actor Critic}\\
\end{matrix*}
$



Problems with policty gradients
- The critic might have very bad estimates of states it hasn't visited and think those states are more valuable
Solution
- Don't change the policy too much
- Keep the KL divergence of old policy and new policy less than threshold
- Trust regision policy optimization
- PPO: Proximal Policy Optimization
