import gym
import pandas

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
