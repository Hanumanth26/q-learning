# Q Learning Algorithm


## AIM
To develop a Python program to find the optimal policy for the given RL environment using Q-Learning and compare the state values with the Monte Carlo method.
## PROBLEM STATEMENT
Implementing the Q-Learning algorithm to train an agent in the Frozen Lake environment and compare its performance with the Monte Carlo algorithm. The objective is to evaluate the success rates, convergence speed, and optimality of policies learned by each algorithm. Performance will be assessed through metrics like average reward per episode and policy stability.

## Q LEARNING ALGORITHM
Include the steps involved in the Q Learning algorithm

Step 1:
Initialize Q-table and hyperparameters.

Step 2:
Choose an action using the epsilon-greedy policy and execute the action, observe the next state, reward, and update Q-values and repeat until episode ends.

Step 3:
After training, derive the optimal policy from the Q-table.

Step 4:
Implement the Monte Carlo method to estimate state values.

Step 5:
Compare Q-Learning policy and state values with Monte Carlo results for the given RL environment.

## Q LEARNING FUNCTION
### Name:hanumanth raog
### Register Number:212222240016
Include the Q Learning function
```
def q_learning(env,
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    select_action=lambda state,Q,epsilon: np.argmax(Q[state]) if np.random.random()>epsilon else np.random.randint(len(Q[state]))
    alphas=decay_schedule(
        init_alpha,min_alpha,
        alpha_decay_ratio,
        n_episodes)
    epsilons=decay_schedule(
        init_epsilon,min_epsilon,
        epsilon_decay_ratio,
        n_episodes)
    for e in tqdm(range(n_episodes),leave=False):
      state,done=env.reset(),False
      action=select_action(state,Q,epsilons[e])
      while not done:
        action=select_action(state,Q,epsilons[e])
        next_state,reward,done,_=env.step(action)
        td_target=reward+gamma*Q[next_state].max()*(not done)
        td_error=td_target-Q[state][action]
        Q[state][action]=Q[state][action]+alphas[e]*td_error
        state=next_state
      Q_track[e]=Q
      pi_track.append(np.argmax(Q,axis=1))
    V=np.max(Q,axis=1)
    pi=lambda s:{s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]
    return Q, V, pi, Q_track, pi_track
```
## OUTPUT:

Optimal State Value Functions
![Screenshot 2024-11-16 131020](https://github.com/user-attachments/assets/4eb0f1d4-0b6e-475f-844d-5b1c6b28e3d1)



Optimal Action Value Functions

![Screenshot 2024-11-16 131020](https://github.com/user-attachments/assets/5320e50f-e32d-4b22-8280-72412bc4a5e7)


Include plot comparing the state value functions of Monte Carlo method and Qlearning.

State value functions of Monte Carlo method:

![Screenshot 2024-11-16 131404](https://github.com/user-attachments/assets/a966c04f-f012-4e9e-9931-3f13454a9d48)


State value functions of Qlearning method:
![Screenshot 2024-11-16 131425](https://github.com/user-attachments/assets/19bfb872-ad90-4d48-815e-ac0786208143)


## RESULT:

Thus, Q-Learning outperformed Monte Carlo in finding the optimal policy and state values for the RL problem.
