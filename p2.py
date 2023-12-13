import gym
import pandas
import math
import numpy as np
import copy
import matplotlib.pyplot as plt

# System constants
M = 1
m = 0.1
grav = -9.8
length = 0.5
mu_c = 0.0005
mu_p = 0.000002

# Simulation constants
dt = 0.02
thresh = 0.01

# Current state vars
theta = 0
x = 0
thetadot = 0
xdot = 0

# Lists of state ranges
thetaStates = [["Terminal", "Terminal"], [-12, -6], [-6, -1], [-1, 0], [0, 1], [1, 6], [6, 12]]
xStates = [["Terminal", "Terminal"], [-2.4, -0.8], [-0.8, 0.8], [0.8, 2.4]]
thetadotStates = [[-10000000000, -50], [-50, 50], [50, 10000000000]]
xdotStates = [[-1000000000, -0.5], [-0.5, 0.5], [0.5, 1000000000]]

# List of range IDs that correspond to state range. 0 represents terminal for theta and x states
thetaIDs = range(len(thetaStates))
xIDs = range(len(xStates))
thetadotIDs = range(len(thetadotStates))
xdotIDs = range(len(xdotStates))


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


# Takes a given state (thetas and x's) and action (force), and outputs a new state
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


# Checks if val is within bounds, a 2-element list
def between(val, bounds):
    if len(bounds) != 2:
        raise Exception("Bounds should be a 2-element list")
    elif bounds[0] <= val < bounds[1]:
        return True
    else:
        return False


# Given a state list, outputs the IDs
def genStateIDs(state):
    # Initalizes ids to zero, so if x/theta aren't within an existing range then we know they're in a
    # terminal state, and we don't need to change anything
    idList = [0, 0, 0, 0]
    for i in range(1, len(thetaStates)):
        if between(state[0], thetaStates[i]):
            idList[0] = i
            break

    for i in range(len(thetadotStates)):
        if between(state[0], thetadotStates[i]):
            idList[1] = i
            break

    for i in range(1, len(xStates)):
        if between(state[0], xStates[i]):
            idList[2] = i
            break

    for i in range(len(xdotStates)):
        if between(state[0], xdotStates[i]):
            idList[3] = i
            break

    return idList


def genIncrements(bounds, steps):
    start = bounds[0]
    stop = bounds[1]
    terms = np.linspace(start, stop, steps, endpoint=False)
    return terms.tolist()


# Given a certain state, outputs the probability of moving into the next state given an action
# Action should be either 10 or -10, corresponding to the force
# Assumes all possible configurations within state boxes are equally likely
# Assumes that first and last states for velocity terms are terminal, as the average of any state that
# includes infinity will terminate in the next time step
def stateProbs(stateIDs, action):
    # Makes sure it's not trying to run probability on a terminal point
    if stateIDs[2] == 0 or stateIDs[0] == 0 or stateIDs[1] in [0, 2] or stateIDs[3] in [0, 2]:
        return [0, 0, 0, 0]
    possibleStates = [[0], [0], [0], [0]]
    possibleStates[0] = genIncrements(thetaStates[stateIDs[0]], 20)
    possibleStates[1] = genIncrements(thetadotStates[stateIDs[1]], 20)
    possibleStates[2] = genIncrements(thetaStates[stateIDs[2]], 20)
    possibleStates[3] = genIncrements(thetaStates[stateIDs[3]], 20)

    numPointsTested = len(possibleStates[0]) * len(possibleStates[1]) * len(possibleStates[2]) * len(possibleStates[3])
    print(numPointsTested)
    # Builds a 4-D matrix that shows probability of any state occuring given the action
    xdotBuild = [0, 0, 0]
    xBuild = [copy.deepcopy(xdotBuild), copy.deepcopy(xdotBuild), copy.deepcopy(xdotBuild), copy.deepcopy(xdotBuild)]
    thetadotBuild = [copy.deepcopy(xBuild), copy.deepcopy(xBuild), copy.deepcopy(xBuild)]

    probMatrix = [copy.deepcopy(thetadotBuild), copy.deepcopy(thetadotBuild), copy.deepcopy(thetadotBuild), copy.deepcopy(thetadotBuild),
                  copy.deepcopy(thetadotBuild), copy.deepcopy(thetadotBuild), copy.deepcopy(thetadotBuild)]
    # Iterates through possible xdots
    for i in range(len(possibleStates[3])):
        # possible x's
        for j in range(len(possibleStates[2])):
            # possible theta dots
            for k in range(len(possibleStates[1])):
                # possible thetas
                for l in range(len(possibleStates[0])):
                    nState = nextState(possibleStates[0][l], possibleStates[1][k], possibleStates[2][j],
                                       possibleStates[3][i], action)
                    nStateID = genStateIDs(nState)
                    probMatrix[nStateID[0]][nStateID[1]][nStateID[2]][nStateID[3]] += 1

    return probMatrix


testProb = stateProbs([3, 1, 3, 1], 10)
print(testProb)





# VALUE ITERATION




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
