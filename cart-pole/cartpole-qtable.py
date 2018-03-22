import gym
import numpy as np

theta_ranges = np.linspace(np.radians(-15), np.radians(15), 7)
theta_dot_ranges = np.linspace(np.radians(-30), np.radians(30), 4)


def q_lookup(q_table, observation):
    theta_ix = min(np.searchsorted(theta_ranges[1:], observation[2], side="right"), q_table.shape[0] - 1)
    theta_dot_ix = min(np.searchsorted(theta_dot_ranges[1:], observation[3], side="right"), q_table.shape[1] - 1)
    return q_table[theta_ix, theta_dot_ix]


def train_episode(env, q_table, learning_rate=.1, gamma=.99, epsilon=.1):
    time_steps = 200
    n = 0
    rewards = list()
    observation = env.reset()
    while n < time_steps:
        q_cell = q_lookup(q_table=q_table, observation=observation)
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()  # Draw a random action
        else:
            action = np.argmax(q_cell)
        observation, reward, done, info = env.step(action)
        rewards.append(reward)
        if done and n < time_steps - 1:
            new_q_value = 0
        else:
            new_q_value = reward + gamma * np.max(q_lookup(q_table=q_table, observation=observation))
        q_cell[action] = learning_rate * new_q_value + (1 - learning_rate) * q_cell[action]
        n += 1
        if done:
            break
    return np.sum(rewards)


def show(env, q_table):
    observation = env.reset()
    done = False
    rewards = 0.0
    while not done:
        env.render()
        q_cell = q_lookup(q_table=q_table, observation=observation)
        action = np.argmax(q_cell)
        observation, reward, done, info = env.step(action)
        rewards += reward
    return rewards


def run():
    env = gym.make("CartPole-v0")

    shape = [
        theta_ranges.shape[0] - 1,
        theta_dot_ranges.shape[0] - 1,
        env.action_space.n  # The action space
    ]

    q_table = np.ones(shape=shape) * 0.5
    epsilon = .5  # we will implement epsilon decay
    epsilon_decay_rate = .99
    learning_rate = 1.0
    rewards = []
    for i in range(500):
        reward = train_episode(env, q_table, epsilon=epsilon, learning_rate=learning_rate)
        rewards.append(reward)
        if i % 50 == 0:
            show(env=env, q_table=q_table)
            print("Episode %d, average reward = %f" % (i, np.mean(rewards[-100:])))
        epsilon *= epsilon_decay_rate
        learning_rate *= .995

    reward = show(env=env, q_table=q_table)
    print("Final reward:", reward)
    env.close()

    print("Q-table")
    print(q_table)


if __name__ == '__main__':
    run()
