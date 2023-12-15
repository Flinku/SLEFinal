import numpy as np
import gym
import time
import statistics
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1', render_mode=None)

# vars and stuff
learning_rate = 0.1
discount_rate = 0.95
learning_rate_decay = 0.1
n_iter = 20000
eps = 1
eps_decay_rate = 0.99

# the boxes system 
theta_bins = [np.radians(-12),np.radians(-6),np.radians(-1), 0, 
              np.radians(1),np.radians(6),np.radians(12)]
x_bins = [-2.4,-0.8, 0, 0.8,2.4]
theta_dot_bins = [-np.inf, -np.radians(50), np.radians(50), np.inf]
x_dot_bins = [-np.inf, -0.5, 0.5, np.inf]

# init Q
Q = np.zeros((len(theta_bins) , len(x_bins) , len(theta_dot_bins) , 
              len(x_dot_bins) , env.action_space.n))

def discretize_state(state):
    theta, x, theta_dot, x_dot = state
    
    theta_bin = np.digitize(theta, theta_bins) - 1
    x_bin = np.digitize(x, x_bins) - 1
    theta_dot_bin = np.digitize(theta_dot, theta_dot_bins) - 1
    x_dot_bin = np.digitize(x_dot, x_dot_bins) - 1
    
    return theta_bin, x_bin, theta_dot_bin, x_dot_bin

def render_current_best():
    env = gym.make('CartPole-v1', render_mode='human')
    s, info = env.reset()
    if info: print(info)
    done = False
    while not done:
        discrete_s = discretize_state(s) # discretize s
        a = np.argmax(Q[discrete_s]) # choose action w highest q value
        sp, reward, done, truncated, info = env.step(a)
        if info: print(info)
        s = sp
    env = gym.make('CartPole-v1', render_mode=None)

# for tracking reward metrics
prev_reward = []
avg_reward = []
cum_reward = [0]

# begin learning trials
for iteration in range(n_iter):
    s, info = env.reset()
    if info: print(info)
    done = False

    while not done:
        # epsilon-greedy
        if np.random.rand() < eps:
            a = env.action_space.sample()  # explore rand action
        else:
            discrete_s = discretize_state(s) # discretize s
            a = np.argmax(Q[discrete_s]) # choose action w highest q value

        sp, reward, done, truncated, info = env.step(a)
        if info: print(info)
            
        # Reward is −1 for the failure, and 0 otherwise, with discounting <-??
        if done: 
            reward = -1
        else: 
            reward = 0
        prev_reward.append(reward)

        # discretize next state and current state
        discrete_sp = discretize_state(sp)
        discrete_s = discretize_state(s)

        # update q w bellman   
        Q[discrete_s + (a,)] = (1-learning_rate) * Q[discrete_s + (a,)] + (
            learning_rate) * ( reward + 
                discount_rate * np.max(Q[discrete_sp]) - Q[discrete_s + (a,)]
        )
        s = sp #move to next state
            
    # decay stuffs
    eps *= eps_decay_rate
    learning_rate = learning_rate / (1 + iteration * learning_rate_decay)
    
    if iteration % (n_iter/10) == 0: #disp and render 10 times
        print('episode:', iteration, '(',100*iteration/n_iter,'% )')
        render_current_best()
        
    if iteration % 10 == 0: #for plotting/metrics
        avg_reward.append(statistics.fmean(prev_reward))
        cum_index = round(iteration/10)
        cum_reward.append(sum(prev_reward))
        prev_reward = []
'''
problem is “considered solved when the average reward is
greater than or equal to 195.0 over 100 consecutive trials.
'''
for _ in range(3): #three runs with best values (in theory... but it sucks)
    render_current_best()
    time.sleep(.5)
    
# Plotting total reward received per episode
plt.figure(figsize=(12, 6))
plt.plot(avg_reward)
plt.title('Average Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.show()

# plotting cum reward
for i in range(len(cum_reward) - 1):
    cum_reward[i + 1] += cum_reward[i]
plt.figure(figsize=(12, 6))
plt.plot(cum_reward)
plt.title('Cumulative Reward over Episodes')
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.show()

# plotting states for one "successful" run (sucks btw)
success_states = []
env = gym.make('CartPole-v1', render_mode='human') #render_mode='human'
s, info = env.reset()
if info: print(info)
done = False
while not done:
    discrete_s = discretize_state(s) # discretize s
    a = np.argmax(Q[discrete_s]) # choose action w highest q value
    sp, reward, done, truncated, info = env.step(a)
    success_states.append(s)
    if info: print(info)
    s = sp
time_steps = range(len(success_states))
thetas = [state[2] for state in success_states]  # theta values
positions = [state[0] for state in success_states]  # x values

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

print("Number of time steps during which the pole does not fail:", 
      len(success_states))