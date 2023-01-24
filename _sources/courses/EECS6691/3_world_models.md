# World Models Project Papers


**Learning and Querying Fast Generative Models for Reinforcement Learning**
- Key challenge is computationally efficient models for reinforcement learning
- They show that working in the latent space is advantageous
- Model based approaches directly predicting the future states


Main Contributions
1) we provide the first comparison of deterministic and stochastic, pixel-space and state-space models w.r.t. speed and accu- racy, applied to challenging environments from the Arcade Learning Environment (ALE, Bellemare et al., 2013)
- Auto regressive models RARs equivelent to RNNS
- State space models: Only need to know the hidden state to predict the next one.
    - Deterministic SSMs
    - Stochastic SSMs
        - Stochastic decoder dSSM-VAE
- Chunk time sequences together. Actions are concatenated and rewards are summed.

2) we demonstrate state-of-the-art environment modeling accuracy (as measured by log-likelihoods) with stochastic state-space models that efficiently produce diverse yet consistent roll- outs


3) using state-space models, we show model-based RL results on MS PACMAN, and obtain significantly improved performance compared to strong model-free baselines


4) we show that learning to query the model further increases policy performance.


Conclusions: On a conceptual level, we present (to the best of our knowledge) the first results on what we termed learning-to-query: We show a learning a rollout policy by backpropagating policy gradients leads to consistent (if modest) improvements.


