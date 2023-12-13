import gym
import pandas
import math
import matplotlib.pyplot as plt

# System constants
M = 1
m = 0.1
grav = -9.8
length = 0.5
mu_c = 0.0005
mu_p = 0.000002

# Simulation constants
dt = 0.01

# Current state vars
theta = 0
x = 0
thetadot = 0
xdot = 0

# Possible States
thetaStates = [-12, -6, -1, 0, 1, 6, 12]
xStates = [-2.4, -0.8, 0, 0.8, 2.4]
thetadotStates = [-50, 0, 50]
xdotStates = [-0.5, 0, 0.5]


# Sign function
def sgn(val):
    if val > 0:
        return 1
    elif val < 0:
        return -1
    else:
        return 0


# Calculates angular acceleration using formula [1]; returns acceleration in DEG/s^2
def calcThetaDD(theta_deg, thetadot_deg, xdot, force):
    theta_rad = math.radians(theta_deg)
    thetadot_rad = math.radians(thetadot_deg)

    num1 = grav*math.sin(theta_rad) - (mu_p*thetadot_rad/(m*length))
    num2sub1 = -force - (m*length * thetadot_rad*thetadot_rad * math.sin(theta_rad)) + mu_c*sgn(xdot)
    num2sub2 = M + m
    numer = num1 + math.cos(theta_rad)*(num2sub1/num2sub2)
    denom = length * (4/3 - (m*(math.cos(theta_rad)*math.cos(theta_rad)))/(m+M))
    thetadd = math.degrees(numer/denom)

    return thetadd


# Calculates linear acceleration using formula [1]
def calcXDD(theta_deg, thetadot_deg, thetadd_deg, xdot, force):
    theta_rad = math.radians(theta_deg)
    thetadot_rad = math.radians(thetadot_deg)
    thetadd_rad = math.radians(thetadd_deg)

    paren = (thetadot_rad**2) * math.sin(theta_rad) - thetadd_rad*math.cos(theta_rad)
    numer = force + m*length*paren - mu_c*sgn(xdot)
    xdd = numer/(M+m)
    return xdd


# Takes a given state (thetas/x's) and action (force), and outputs a new state
# Output is list of [theta, theta_dot, x, x_dot]
def nextState(theta, thetadot, x, xdot, force):
    angAccel = calcThetaDD(theta_deg=theta, thetadot_deg=thetadot, xdot=xdot, force=force)
    xAccel = calcXDD(theta_deg=theta, thetadot_deg=thetadot, thetadd_deg=angAccel, xdot=xdot, force=force)

    newState = [0, 0, 0, 0]
    newState[0] = theta + dt*thetadot
    newState[1] = thetadot + dt*angAccel
    newState[2] = x + dt*xdot
    newState[3] = xdot + dt*xAccel

    return newState



"""
env = gym.make('CartPole-v1', render_mode='human')


def basic_policy(obs):
    print(obs[0])
    angle = obs[0][2]
    return 0 if angle < 0 else 1


totals = []
for episode in range(500):
    episode_rewards = 0
    obs = env.reset()
    for step in range(1000):
        action = basic_policy(obs)
        obs, reward, done, info = env.step(action)[0]
        episode_rewards += reward
        if done:
            break

    totals.append(episode_rewards)

env.close()
"""
