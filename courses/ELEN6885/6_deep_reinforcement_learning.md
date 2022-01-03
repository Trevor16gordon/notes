# Deep Reinforcement Learning


Before we had a lookup table for Q/V values

- Use function approximation that takes in state and actions and gives a q value. Function has weights that need to be trained.
- Best action needs to find max out of all actions


Normally we train using labels. We don't have labels here so we train using target like TD Target.

actual = R + sicount_fac * max_q (a_prime, s_prime)

