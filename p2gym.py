import numpy as np
import gym
import time
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1', render_mode=None) #render_mode='human'

# vars and stuff
learning_rate = 0.1
discount_rate = 0.95
learning_rate_decay = 0.1
n_iter = 20000
eps = 1
eps_decay_rate = 0.99995

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

prev_count = []  # array of all scores over runs
for iteration in range(n_iter):
    s, info = env.reset()
    if info: print(info)
    done = False
    count = 0 #cart mvmnts

    while not done:
        count += 1
        # epsilon-greedy
        if np.random.rand() < eps:
            a = env.action_space.sample()  # explore rand action
        else:
            discrete_s = discretize_state(s) # discretize s
            a = np.argmax(Q[discrete_s]) # choose action w highest q value

        sp, reward, done, truncated, info = env.step(a)
        if info: print(info)
            
        # Reward is −1 for the failure, and 0 otherwise, with discounting
        if done: 
            reward = -1 
        else: 
            reward = 0 

        # discretize next state and current state
        discrete_sp = discretize_state(sp)
        discrete_s = discretize_state(s)

        # update q w bellman
        Q[discrete_s + (a,)] = learning_rate * Q[discrete_s + (a,)] + (
            1 - learning_rate) * (
                reward + discount_rate * np.max(Q[discrete_sp])
        )
        s = sp #move to next state
        
    prev_count.append(count)
    print('count:', count)
    
    # decay stuffs
    eps *= eps_decay_rate
    learning_rate = learning_rate / (1 + iteration * learning_rate_decay)
    print('iteration:', iteration)
    
env = gym.make('CartPole-v1', render_mode='human') #render_mode='human'
s, info = env.reset()
env.render()
discrete_s = discretize_state(s) # discretize s
a = np.argmax(Q[discrete_s]) # choose action w highest q value
sp, reward, done, truncated, info = env.step(a)