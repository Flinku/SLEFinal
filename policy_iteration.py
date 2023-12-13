import numpy as np
import matplotlib.pyplot as plt

# Constants
M = 1.0  # Mass of cart
m = 0.1  # Mass of pole
g = -9.8 # Gravity
l = 0.5  # Half-pole length
mu_c = 0.0005 
mu_p = 0.000002
dt = 0.02 # Time step

# Discretization of state space
theta_bins = [-12, -6, -1, 0, 1, 6, 12] # degrees
x_bins = [-2.4, -0.8, 0.8, 2.4] # meters
theta_dot_bins = [-50, 50, float('inf')] # degrees per second
x_dot_bins = [-0.5, 0.5, float('inf')] # meters per second

# Convert degrees to radians for calculations
theta_bins_rad = [np.radians(theta) for theta in theta_bins]

# System Dynamics
def system_dynamics(state, action):
    theta, theta_dot, x, x_dot = state
    F = action

    # Dynamics equations
    theta_double_dot = (g * np.sin(theta) + np.cos(theta) * (-F - m * l * theta_dot**2 * np.sin(theta) + mu_c * np.sign(x_dot) / (M + m))
                        - (mu_p * theta_dot) / (m * l)) / (l * (4/3 - m * np.cos(theta)**2 / (M + m)))
    x_double_dot = (F + m * l * (theta_dot**2 * np.sin(theta) - theta_double_dot * np.cos(theta)) - mu_c * np.sign(x_dot)) / (M + m)

    # Update state by Euler's method
    new_theta_dot = theta_dot + dt * theta_double_dot
    new_theta = theta + dt * new_theta_dot
    new_x_dot = x_dot + dt * x_double_dot
    new_x = x + dt * new_x_dot 
    
    new_state = [new_theta, new_theta_dot, new_x, new_x_dot]

    return new_state

# Lists of state ranges
thetaStates = [[-12, -6], [-6, -1], [-1, 0], [0, 1], [1, 6], [6, 12]]
xStates = [[-2.4, -0.8], [-0.8, 0.8], [0.8, 2.4]] 
thetadotStates = [[-float('inf'), -50], [-50, 50], [50, float('inf')]] 
xdotStates = [[-float('inf'), -0.5], [-0.5, 0.5], [0.5, float('inf')]]

def discretize_state(theta, x, theta_dot, x_dot):
    # Convert degrees to radians for theta and theta_dot
    theta_rad = np.radians(theta)
    theta_dot_rad = np.radians(theta_dot)

    # Find the index for each state variable
    def find_index(value, ranges):
        for i, (low, high) in enumerate(ranges):
            if low <= value < high:
                return i
        return 0 if value < ranges[0][0] else len(ranges) - 1

    theta_idx = find_index(theta_rad, thetaStates)
    x_idx = find_index(x, xStates)
    theta_dot_idx = find_index(theta_dot_rad, thetadotStates)
    x_dot_idx = find_index(x_dot, xdotStates)

    return (theta_idx, x_idx, theta_dot_idx, x_dot_idx)



# Initialize policy and value function
num_states = len(thetaStates) * len(xStates) * len(thetadotStates) * len(xdotStates)
value_function = np.zeros(num_states)
policy = np.random.choice([10, -10], size=num_states)  # Random initial policy

# Policy iteration parameters
theta_threshold = 0.1
gamma = 0.99  # Discount factor

# Helper function to convert state indices to the state tuple
def state_tuple_from_index(index):
    total_x_states = len(xStates)
    total_thetadot_states = len(thetadotStates)
    total_xdot_states = len(xdotStates)

    theta_idx = index // (total_x_states * total_thetadot_states * total_xdot_states)
    remainder = index % (total_x_states * total_thetadot_states * total_xdot_states)
    x_idx = remainder // (total_thetadot_states * total_xdot_states)
    remainder = remainder % (total_thetadot_states * total_xdot_states)
    theta_dot_idx = remainder // total_xdot_states
    x_dot_idx = remainder % total_xdot_states

    return theta_idx, x_idx, theta_dot_idx, x_dot_idx

def get_state_index(state_tuple):
    theta_idx, x_idx, theta_dot_idx, x_dot_idx = state_tuple
    index = theta_idx * (len(xStates) * len(thetadotStates) * len(xdotStates))
    index += x_idx * (len(thetadotStates) * len(xdotStates))
    index += theta_dot_idx * len(xdotStates)
    index += x_dot_idx
    return index


# Policy Iteration
def policy_iteration():
    global policy, value_function
    policy_stable = False

    while not policy_stable:
        # Policy Evaluation
        delta = float('inf')
        while delta > theta_threshold:
            delta = 0
            for state_index in range(num_states):
                v = value_function[state_index]
                theta_idx, x_idx, theta_dot_idx, x_dot_idx = state_tuple_from_index(state_index)
    
                # Ensure the indices are within the bounds
                theta_idx = min(theta_idx, len(thetaStates) - 1)
                x_idx = min(x_idx, len(xStates) - 1)
                theta_dot_idx = min(theta_dot_idx, len(thetadotStates) - 1)
                x_dot_idx = min(x_dot_idx, len(xdotStates) - 1)

                # Construct the state using the indices
                state = (thetaStates[theta_idx][0], xStates[x_idx][0], thetadotStates[theta_dot_idx][0], xdotStates[x_dot_idx][0])

                # Update value function
                new_state = system_dynamics(state, policy[state_index])
                new_state_tuple = discretize_state(*new_state)
                new_state_index = get_state_index(new_state_tuple)
                reward = -1 if not (-12 <= new_state[0] <= 12 and -2.4 <= new_state[2] <= 2.4) else 0
                value_function[state_index] = reward + gamma * value_function[new_state_index]
                delta = max(delta, abs(v - value_function[state_index]))

        # Policy Improvement
        policy_stable = True
        for state_index in range(num_states):
            old_action = policy[state_index]
            state_tuple = state_tuple_from_index(state_index)  # Convert index to state tuple
            state = (thetaStates[state_tuple[0]][0], xStates[state_tuple[1]][0],
                     thetadotStates[state_tuple[2]][0], xdotStates[state_tuple[3]][0])

            # Find the action that maximizes the expected return
            expected_returns = []
            for action in [10, -10]:
                new_state = system_dynamics(state, action)
                new_state_tuple = discretize_state(*new_state)
                new_state_index = get_state_index(new_state_tuple)
                reward = -1 if not (-12 <= new_state[0] <= 12 and -2.4 <= new_state[2] <= 2.4) else 0
                expected_returns.append(reward + gamma * value_function[new_state_index])
            
            policy[state_index] = [10, -10][np.argmax(expected_returns)]
            if old_action != policy[state_index]:
                policy_stable = False

# Run the policy iteration
policy_iteration()


