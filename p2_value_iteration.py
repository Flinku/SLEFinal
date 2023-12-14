import gym
import pandas
import math
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import colors

# System constants
M = 1
m = 0.1
grav = -9.8
length = 0.5
mu_c = 0.0005
mu_p = 0.000002

# Simulation constants
dt = 0.02
# Max allowable delta to stop iterations
thresh = 0.01
# Max overall iterations
iterLimit = 100
# Number of evenly spaced splits to look at to determine probability of moving from one state to another
probabilitySplits = 5

# Lists of state ranges
thetaStates = [["Terminal", "Terminal"], [-12, -6], [-6, -1], [-1, 0], [0, 1], [1, 6], [6, 12]]
thetadotStates = [[-10000000000, -75], [-75, -25], [-25, 25], [25, 75], [75, 10000000000]]
xStates = [["Terminal", "Terminal"], [-2.4, -0.8], [-0.8, 0.8], [0.8, 2.4]]
xdotStates = [[-1000000000, -0.75], [-0.75, -0.25], [-0.25, 0.25], [0.25, 0.75], [0.75, 1000000000]]

# List of range IDs that correspond to state range. 0 represents terminal for theta and x states
thetaIDs = range(len(thetaStates))
thetadotIDs = range(len(thetadotStates))
xIDs = range(len(xStates))
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

    num1 = grav * math.sin(theta_rad) - (mu_p * thetadot_rad / (m * length))
    num2sub1 = -force - (m * length * thetadot_rad * thetadot_rad * math.sin(theta_rad)) + mu_c * sgn(xdot)
    num2sub2 = M + m
    numer = num1 + math.cos(theta_rad) * (num2sub1 / num2sub2)
    denom = length * (4 / 3 - (m * (math.cos(theta_rad) * math.cos(theta_rad))) / (m + M))
    thetadd = math.degrees(numer / denom)

    return thetadd


# Calculates linear acceleration using formula [1]
def calcXDD(theta_deg, thetadot_deg, thetadd_deg, xdot, force):
    theta_rad = math.radians(theta_deg)
    thetadot_rad = math.radians(thetadot_deg)
    thetadd_rad = math.radians(thetadd_deg)

    paren = (thetadot_rad * thetadot_rad) * math.sin(theta_rad) - thetadd_rad * math.cos(theta_rad)
    numer = force + m * length * paren - mu_c * sgn(xdot)
    xdd = numer / (M + m)
    return xdd


# Takes a given state (thetas and x's) and action (force), and outputs a new state
# Output is list of [theta, theta_dot, x, x_dot]
def nextState(theta, thetadot, x, xdot, force):
    angAccel = calcThetaDD(theta_deg=theta, thetadot_deg=thetadot, xdot=xdot, force=force)
    xAccel = calcXDD(theta_deg=theta, thetadot_deg=thetadot, thetadd_deg=angAccel, xdot=xdot, force=force)

    newState = [0, 0, 0, 0]
    # Doing it in this order is necessary, as otherwise acceleration has no impact on position within a time step
    newState[1] = thetadot + dt * angAccel
    newState[0] = theta + dt * newState[1]
    newState[3] = xdot + dt * xAccel
    newState[2] = x + dt * newState[3]

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
        if between(state[1], thetadotStates[i]):
            idList[1] = i
            break

    for i in range(1, len(xStates)):
        if between(state[2], xStates[i]):
            idList[2] = i
            break

    for i in range(len(xdotStates)):
        if between(state[3], xdotStates[i]):
            idList[3] = i
            break

    return idList


# Generates steps number of evenly spaced points within the range of bounds
def genIncrements(bounds, steps):
    start = bounds[0]
    stop = bounds[1]
    terms = np.linspace(start, stop, steps)
    return terms.tolist()


# Generates a 4d array, used for value function, policy, and probability matrix
def gen4darray():
    xdB = [0]*len(xdotStates)
    xB = []
    for i in xStates:
        xB.append(copy.deepcopy(xdB))
    tdB = []
    for i in thetadotStates:
        tdB.append(copy.deepcopy(xB))
    tB = []
    for i in thetaStates:
        tB.append(copy.deepcopy(tdB))

    return tB


# Given a certain state, outputs the probability of moving into the next state given an action
# Action should be either 10 or -10, corresponding to the force
# Assumes all possible configurations within state boxes are equally likely
# Assumes that first and last states for velocity terms are terminal, as the average of any state that
# includes infinity will terminate in the next time step
def stateProbs(stateIDs, action):
    # Makes sure its not trying to run probability on a terminal point
    if stateIDs[0] == 0 or stateIDs[1] in [0, len(thetadotStates)-1] or stateIDs[2] == 0 or stateIDs[3] in [0, len(xdotStates)-1]:
        print(f"Returning none, {stateIDs} is invalid for probability")
        return None
    possibleStates = [[0], [0], [0], [0]]
    # Populates lists with all the incremental values tested
    possibleStates[0] = genIncrements(thetaStates[stateIDs[0]], probabilitySplits)
    possibleStates[1] = genIncrements(thetadotStates[stateIDs[1]], probabilitySplits)
    possibleStates[2] = genIncrements(xStates[stateIDs[2]], probabilitySplits)
    possibleStates[3] = genIncrements(xdotStates[stateIDs[3]], probabilitySplits)

    # Gets the overall number of points tested so it knows what to divide by to get the probability
    numPointsTested = len(possibleStates[0]) * len(possibleStates[1]) * len(possibleStates[2]) * len(possibleStates[3])
    # Builds a 4-D matrix that shows probability of any state occuring given the action
    probMatrix = gen4darray()
    # Iterates through possible xdots
    for i in range(len(possibleStates[3])):
        # possible x's
        for j in range(len(possibleStates[2])):
            # possible theta dots
            for k in range(len(possibleStates[1])):
                # possible thetas
                for l in range(len(possibleStates[0])):
                    # Calculates where it would end up given the exact state values being assumed
                    nState = nextState(possibleStates[0][l], possibleStates[1][k], possibleStates[2][j],
                                       possibleStates[3][i], action)
                    # Figures out what state the resulting position belongs to
                    nStateID = genStateIDs(nState)
                    # Adds a point to that specific end state in the probability matrix
                    probMatrix[nStateID[0]][nStateID[1]][nStateID[2]][nStateID[3]] += 1 / numPointsTested
    return probMatrix


# Determines the reward given a set of state ids. While technically it hasn't necessarily failed if the velocities are
# in the outer range, due to our assumption that the outer ranges include infinity it must fail on the next time step,
# so we just consider those states terminal
def getReward(stateIDs):
    if stateIDs[0] == 0 or stateIDs[1] in [0, len(thetadotStates)-1] or stateIDs[2] == 0 or stateIDs[3] in [0, len(xdotStates)-1]:
        return 0
    else:
        return 1


# Computes the sum of all probabilities * rewards given a probability matrix and previous value function
def getValSum(probMatrix, Vs, discount=1):
    valSum = 0

    for i in range(len(probMatrix)):
        for j in range(len(probMatrix[i])):
            for k in range(len(probMatrix[i][j])):
                for l in range(len(probMatrix[i][j][k])):
                    valSum += probMatrix[i][j][k][l] * (getReward([i, j, k, l]) + (discount * Vs[i][j][k][l]))

    return valSum


# Takes a 4-D previous value policy array, and outputs a new one and the updated delta
def iterValue(vs_old):
    delta = 0
    Vs = copy.deepcopy(vs_old)
    # Policy array; -1 denotes left force, +1 denotes right
    policy = gen4darray()

    # Iterates over every possible state
    # xdots
    for i in xdotIDs[1:-1]:
        # xs
        for j in xIDs[1:]:
            # theta dots
            for k in thetadotIDs[1:-1]:
                # thetas
                for l in thetaIDs[1:]:
                    # print(f"{l}, {k}, {j}, {i}")
                    # Saves the old value at this state to variable v
                    v = Vs[l][k][j][i]

                    # Computes the probability of ending up in all possible states given a forward or backward force
                    forwardProb = stateProbs([l, k, j, i], 10)
                    backwardProb = stateProbs([l, k, j, i], -10)

                    # Gets the value sums
                    forwardValSum = getValSum(forwardProb, Vs)
                    backwardValSum = getValSum(backwardProb, Vs)

                    # Determines which action leads to a better value function
                    if backwardValSum > forwardValSum:
                        policy[l][k][j][i] = -1
                    elif forwardValSum > backwardValSum:
                        policy[l][k][j][i] = 1
                    else:
                        policy[l][k][j][i] = 0

                    newV = max(forwardValSum, backwardValSum)
                    Vs[l][k][j][i] = newV
                    delta = max(delta, abs(v - newV))

    return Vs, delta, policy


V_s = gen4darray()

delta = 1
counter = 0
policy = None

while delta > thresh and counter < 100:
    V_s, delta, policy = iterValue(V_s)
    counter += 1
    print(f"Iteration {counter}, delta={delta}")
print(f"After {counter} iterations, reached a delta of {delta}")


# Generates a 2d array representing the x/theta values for a given theta_dot/x_dot range. Works for V_s and policy
# (Would also work for probability matrix but that wouldn't be used here)
def getVelocValue(thetaVelID, xVelID, fourdarr):
    valueMap = np.array([[fourdarr[1][thetaVelID][1][xVelID], fourdarr[1][thetaVelID][2][xVelID], fourdarr[1][thetaVelID][3][xVelID]],
                         [fourdarr[2][thetaVelID][1][xVelID], fourdarr[2][thetaVelID][2][xVelID], fourdarr[2][thetaVelID][3][xVelID]],
                         [fourdarr[3][thetaVelID][1][xVelID], fourdarr[3][thetaVelID][2][xVelID], fourdarr[3][thetaVelID][3][xVelID]],
                         [fourdarr[4][thetaVelID][1][xVelID], fourdarr[4][thetaVelID][2][xVelID], fourdarr[4][thetaVelID][3][xVelID]],
                         [fourdarr[5][thetaVelID][1][xVelID], fourdarr[5][thetaVelID][2][xVelID], fourdarr[5][thetaVelID][3][xVelID]],
                         [fourdarr[6][thetaVelID][1][xVelID], fourdarr[6][thetaVelID][2][xVelID], fourdarr[6][thetaVelID][3][xVelID]]])
    return valueMap


# Generates the plot of a given theta dot and x dot range
def genPlot(thetaD, xD):
    # Creates the mappings
    map1_1 = getVelocValue(thetaD, xD, V_s)
    policymap = getVelocValue(thetaD, xD, policy)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Sets the tick locations to the cutoff between different ranges
    graphX = np.array([-2.4, -0.8, 0.8, 2.4])
    graphY = np.array([-12, -6, -1, 0, 1, 6, 12])
    GX, GY = np.meshgrid(graphX, graphY)

    ax1.set_xticks(graphX)
    ax1.set_yticks(graphY)
    ax1.set_xlabel("x value (m)")
    ax1.set_ylabel("theta value (deg)")
    # Creates a color map of visually distinct boxes
    valgraph = ax1.pcolormesh(GX, GY, map1_1, edgecolors='black')
    fig.colorbar(valgraph)
    # Adds text labels so the value function can be read
    # Used https://stackoverflow.com/questions/20998083/show-the-values-in-the-grid-using-matplotlib as a reference
    for i in range(len(graphX)-1):
        for j in range(len(graphY) - 1):
            xpos = (graphX[i] + graphX[i+1])/2
            ypos = (graphY[j] + graphY[j + 1]) / 2
            ax1.text(xpos, ypos, '{:0.3f}'.format(map1_1[j][i]), ha='center', va='center',
                     bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    ax1.set_title(f"Value Function; {xdotStates[xD][0]} < x' < {xdotStates[xD][1]}; "
                  f"{thetadotStates[thetaD][0]} < theta' < {thetadotStates[thetaD][1]}")

    # Does the same thing but for the optimal policy plot
    ax2.set_xticks(graphX)
    ax2.set_yticks(graphY)
    ax2.set_xlabel("x value (m)")
    ax2.set_ylabel("theta value (deg)")
    polgraph = ax2.pcolormesh(GX, GY, policymap, edgecolors='black', cmap='RdBu', vmin=-1, vmax=1)
    fig.colorbar(polgraph)
    # Displays an arrow on the box corresponding to whether the optimal policy is going left or right
    numToArrow = {-1: "←", 1: "→", 0: "-"}
    for i in range(len(graphX)-1):
        for j in range(len(graphY)-1):
            xpos = (graphX[i] + graphX[i+1])/2
            ypos = (graphY[j] + graphY[j + 1]) / 2
            ax2.text(xpos, ypos, numToArrow[policymap[j][i]], ha='center', va='center',
                     bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    ax2.set_title(f"Optimal policy; {xdotStates[xD][0]} < x' < {xdotStates[xD][1]}; "
                  f"{thetadotStates[thetaD][0]} < theta' < {thetadotStates[thetaD][1]}")
    plt.show()


genPlot(1, 1)
genPlot(2, 1)
genPlot(3, 1)
genPlot(2, 2)
