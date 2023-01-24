# Reinforcement Learning Research


## IEEE 1900.5.2: Standard Method for Modeling Spectrum Consumption: Introduction and Use Cases

- 3 dimensions for spectral use: Temporal (changing in time), spatials (Changing over different land areas), and spectral (Across frequency bands)
- Transmitters and recievers consume spectrum because a reciever can be interferred from other bands. It has requirements that may prevent other devices from transmitting.
- SCMS can be transmitter models, reciever models, system models, 
- Minimum required constructs for a transmitter are reference power, spetrum mask, power map, propagation map, and a location.
- Receiver models minimum required constructs are reference power, underlay mask, power map, propogation map, location

Potential Use Cases for SCMs:
- Spectrum Access System
- Beacon Based Spectrum Sharing


Q:
- What are the Base parameters for SCMs? (Engineering details)
    - Emission/Selectivity curves?
    - Antenna patterns?
    - Tx power, Rx noise figure
    - IPC
- What are the "Key Spectrum Consumption Modeling Parameters" (Operational Details)
    - Operational Parameters?
    - Channel Parameters?
    - Environmental Effects?
    - Obfuscate sensitive info
- What are the Interference / coexistance algorithms? (Environmental Effects)


Q:
- What does "per solid angle" mean. Both power map and prop map reference it.
- Schedule in the 11 constructs. How far into the future is communication planned?
- You mentioned measurements needed to be taken. How does that play into the 11 constructs?
- "The IEEE 1900.5.2 standard specifies a method to compute the compatibility of spectrum use between different SDSs that have expressed the boundaries of their spectrum use via an SCM" - Is the compatability a 'solved' problem? Or is this just a way to get a rough estimate?


- What if reciever is moving around? Does compatability get updated frequently?
- Cellphone as a transmitter / reciever. Power map changing all the time?



# Prasad Github

noincumbent.py
- Rewards are the change in reward from one slot to the next slot where reward at each slot is log of the number of successful transmissions up to that point

incumbent.py
- Incumbent is the main user of the communication channel and the other radios need to learn when to use the channel?
- State transition matrix P what is the meaning of this? [[0.2, 0.8], [0.6, 0.4]
    - If we weren't just transmiting we will probably transmit in the next case.
- Step function is just checking for collisions and increasing the rewards based on that.
- creatEpsilonGreedy does what it says
- Q table represented by a dictionary lookup table with a default that will give np.zeros with size of the action space.
- State is defined only by the time slot

noincumbentv2.py
- Main difference is the history of rewards is part of the state space.
- 10 slots for transmitting. Shorter amount of time
- 11*11 states for each link by 2
    - All permutations of possible rewards


Explanation of formulation of Markov Decision Processes

State:
- t: Time slot
- T: total time slots
- I(t): number of idle transmission slots up until time t
- C(t): number of collisions slots up until time t
- D1(t): Number of successful transmissions for agent 1 at t
- D2(t): Number of successful transmissions for agent 2 at t



Q:
- Am a correct to say these agents are learning information specific to time slot numbers?
- Any specific takeaways from this that you were suprised about or that encouraged you about moving to more complex scenarios?
- In the future make this time invariant?
- In the future 
    - agent's don't have info about other rewards
    - state space include delay / time on / time off
- Friis equation?
- 



# Deep Multi-User Reinforcement Learning for Distributed Dynamic Spectrum Access - Oshri Naparstek and Kobi Cohen

Understanding
- N users
- K channels
- History for user n defined as all actions and oberservations for user N up to time t
- Strategy is just a probability distribution over the N channels
- Reward is based on the achievable data rate for the channel 
    - B * Log2(1 + SNR)
    - B: Channel bandwitdth
    - SNR: Signal to noise ratio of user n on channel k
- Rn is the discounted sum of all future rewards until the end of the time horizon
- When time is bounded, let gamma equal to 1

Background on Q learning and Deep RL
- Minimize the TD (Time difference) error between current Q value and the target (reward plus the max value of the next state.)


Their Model
- Input layer: (2K + 2)
    - First K + 1 is a one hot encoding showing where the agent transmitted in the last step (+1 refers to not transmitting)
    - Next K entries are the capacity for each channel. Transmission rate if channel is free
    - Last entry is the 1 if the last ACK signal was received. 0 if not.
- LSTM Layer - learns how to aggregate experiences over time.
- Advantage layer
- Value layer
- Q Layer
- Output layer [K + 1]
- Double K Learning:
    - Problem is: ____
    - DQN1 used to select actions
    - DQN2 used to evaluate how good a selection was


Offline Training
-


Online Training
- Don't use experience replay


- Q:  we develop a mechanism that restricts the strategy space for all users when training the DQN
- Q: Do they have don't translate as a strategy?
- Q: When a problem is partially observable we set the state to the observed history. (Or a sliding window)
- Q: Dynamics are non Markovian and determined by multi user interactions. Classical DQNs do not perform well in this setting.
- Q: There are states that are good or bad regardless of the taken action. There fore they estimate the Value function separate from the advantage function