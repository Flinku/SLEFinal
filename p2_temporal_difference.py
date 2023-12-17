#%% Initialization of environment, state spaces, and parameters

import numpy as np
import gym
import statistics
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

env = gym.make('CartPole-v1', render_mode=None)

# parameters
learning_rate0 = 0.05 # initial learning rate, alpha
discount_rate = 0.8 # discount rate, gamma
learning_rate_decay = 0.001 # rate at which the learning rate decays with each iteration
n_iter = 20000 # total number of iterations (episodes)
eps = 1 # rate at which random actions are chosen (exploration)
eps_decay_rate = 0.9995 #rate at which epsilon decays

# the boxes system discretizes the state space for use in the q-table
theta_bins = [np.radians(-12),np.radians(-6),np.radians(-1), 0, 
              np.radians(1),np.radians(6),np.radians(12)]
x_bins = [-2.4,-0.8, 0, 0.8,2.4]
theta_dot_bins = [-np.inf, -np.radians(50), np.radians(50), np.inf]
x_dot_bins = [-np.inf, -0.5, 0.5, np.inf]

# initilzing the Q-table to all zeros
Q = np.zeros((len(theta_bins) , len(x_bins) , len(theta_dot_bins) , 
              len(x_dot_bins) , env.action_space.n))

#%% Functions

def discretize_state(state):
    '''This function takes a given state and discretizes it based on the boxes
    system defined before'''
    theta, x, theta_dot, x_dot = state
    
    theta_bin = np.digitize(theta, theta_bins) - 1
    x_bin = np.digitize(x, x_bins) - 1
    theta_dot_bin = np.digitize(theta_dot, theta_dot_bins) - 1
    x_dot_bin = np.digitize(x_dot, x_dot_bins) - 1
    
    return theta_bin, x_bin, theta_dot_bin, x_dot_bin

def plot_states(states):
    '''Plots a run given by the states that are input'''
    # extract variables for plotting
    time_steps = range(len(states))
    thetas = [state[2] for state in states]  # theta values
    positions = [state[0] for state in states]  # x values
    
    # plotting theta as a function of time
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, thetas)
    plt.title('Theta as a Funtion of Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Theta (Radians)')
    plt.show()
    
    # Plotting x as a function of time
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, positions)
    plt.title('x as a Function of Time')
    plt.xlabel('Time Steps')
    plt.ylabel('x')
    plt.show()
    
def plot_render_sample():
    '''This function will render one run using the current highest q-values.
    In addition, it will plot the x and theta values for the same run'''
    success_states = [] # Tracks current state
    env = gym.make('CartPole-v1', render_mode='human') 
    s, info = env.reset()
    if info: print(info)
    done = False
    
    while not done:
        discrete_s = discretize_state(s) # discretize current state
        a = np.argmax(Q[discrete_s]) # choose action with highest q value
        sp, reward, done, truncated, info = env.step(a)
        if info: print(info)
        success_states.append(s)
        s = sp # update state
    
    plot_states(success_states)
    
    print("Number of time steps during which the pole does not fall:", 
          len(success_states))
    
def check_success():
    ''' "The problem is “considered solved when the average reward is
    greater than or equal to 195.0 over 100 consecutive trials."
    
    This function checks if the problem is considered solved by running 100
    trials and averaging their cumulative rewards. It outputs 1 is successful
    and 0 if unsuccessful. In addition, it will plot the x and theta values 
    for the best run it found in its 100 trials'''
    test_reward = [] # Tracks the reward for each test
    best_trial_reward = 0 # Track the best reward from trials
    best_success = [] # Track the state from best trial
    
    for _ in range(100):
        success_states = []
        env = gym.make('CartPole-v1', render_mode=None)
        s, info = env.reset()
        if info: print(info)
        done = False
        trial_reward = 0
        
        while not done:
            discrete_s = discretize_state(s) # discretize s
            a = np.argmax(Q[discrete_s]) # choose action w highest q value
            sp, reward, done, truncated, info = env.step(a)
            if info: print(info)
            success_states.append(s)
            s = sp
            if done: reward = 0
            else: reward = 1
            trial_reward += reward
            
        test_reward.append(trial_reward)
        
        # determine if last trial was better than current best trial
        if trial_reward > best_trial_reward:
            best_success = success_states 
            best_trial_reward = trial_reward
    
    # calculate and print average reward over trials
    avg_reward = statistics.fmean(test_reward)
    if avg_reward >= 195.0: 
        print('Average reward over 100 trials:', avg_reward,
              '...','Success!')
        overall_success = 1
    else: 
        print('Average reward over 100 trials:', avg_reward,
              '...','Failure :(')
        overall_success = 0
    
    plot_states(best_success)
    
    print("Number of time steps during which the pole does not fall in best trial:", 
          len(best_success))
    
    return overall_success

#%% Training

prev_reward = [] # Tracks rewards over 10 episodes
avg_reward = [] # Tracks average rewards over 10 episodes
cum_reward = [] # Tracks cumulative rewards for each episode
episode_reward = 0 # Tracks current episode reward

learning_rates = [] # Tracks the learning rate for each episode
eps_rates = [] # Tracks the exploration rate for each episode
choices = [0,0] # Tracks number of actions spent exploring and exploiting

# begin trials
best_success = [] # Tracks the states for the best success episode
best_episode_reward = 0 # Tracks the reward for the best success episode

for iteration in range(n_iter):
    success_states = [] # Tracks current state
    s, info = env.reset()
    if info: print(info)
    done = False
    
    while not done:
        # epsilon-greedy action choice policy
        if np.random.rand() < eps: 
            a = env.action_space.sample()  # explore random action
            choices[0]+=1
        else:
            discrete_s = discretize_state(s) # discretize s
            a = np.argmax(Q[discrete_s]) # choose action w highest q value
            choices[1]+=1

        sp, reward, done, truncated, info = env.step(a)
        if info: print(info)
            
        # Reward is 0 for the failure, and 1 otherwise
        if done: 
            reward = 0
            cum_reward.append(episode_reward)
        else: 
            reward = 1
            episode_reward += reward
            
        prev_reward.append(reward)
        
        # discretize next state and current state
        discrete_sp = discretize_state(sp)
        discrete_s = discretize_state(s)

        # update q-table with bellman equation
        learning_rate = learning_rate0 / (1 + iteration * learning_rate_decay)
        
        Q[discrete_s + (a,)] = learning_rate * Q[discrete_s + (a,)] + (
            1 - learning_rate) * ( reward + 
                                 discount_rate * np.max(Q[discrete_sp])
                                 )
                                  
        success_states.append(s)
        s = sp #move to next state
            
    # Track and decay learning rate and exploration rate
    learning_rates.append(learning_rate)
    eps_rates.append(eps)
    eps *= eps_decay_rate
    
    # determine if last episode is better than current episode
    if episode_reward > best_episode_reward:
        best_success = success_states
        best_episode_reward = episode_reward
    episode_reward = 0
    
    # Display progress every 10% towards completion
    if iteration % (n_iter/10) == 0:
        print('episode:', iteration, '(',100*iteration/n_iter,'% done )')
    
    # Track average reward every 10 episodes
    if iteration % 10 == 0: #for plotting/metrics
        avg_reward.append(statistics.fmean(prev_reward)) 
        prev_reward = []
        
print('100% complete!')

#%% Showing results

# Plot the best run
plot_states(best_success)
print("Number of time steps during which the pole does not fall in best episode:", 
      len(best_success))
    
# Plotting average reward received per episode
plt.figure(figsize=(12, 6))
plt.plot(avg_reward)
plt.title('Average Reward over Every 10 Episodes')
plt.xlabel('Episode (10e-1)')
plt.ylabel('Average Reward')
plt.show()

# Plotting cumulative reward received per episode
plt.figure(figsize=(12, 6))
plt.plot(cum_reward)
plt.title('Cumulative Reward over Episodes')
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.show()

# print the exploration vs exploitation rate
print(choices[0],'random actions taken vs.',choices[1],'selected actions (',
      100*choices[0]//sum(choices), '% random )')

# Plot the learning and exploration rate over episodes
plt.figure(figsize=(12, 6))
plt.plot(learning_rates)
plt.title('Learning Rate over Episodes')
plt.xlabel('Episode')
plt.ylabel('Alpha')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(eps_rates)
plt.title('Exploration Rate over Episodes')
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.show()

print()
check_success()