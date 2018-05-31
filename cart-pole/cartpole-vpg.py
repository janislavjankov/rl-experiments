import tensorflow as tf
import numpy as np
import gym
import argparse


def build_policy(inputs, layers):
    for units, activation in layers:
        inputs = tf.layers.dense(inputs=inputs, units=units, activation=activation)

    # We will be using bernoulli distribution, so we only need one parameter
    logits = tf.layers.dense(inputs=inputs, units=1, activation=None)
    dist = tf.distributions.Bernoulli(logits=logits)
    return dist


def train_agent(env, args):
    states_ = tf.placeholder(tf.float32, shape=[None, env.observation_space.shape[0]], name="states_")
    policy = build_policy(inputs=states_, layers=[(8, tf.nn.tanh)] * 2)
    samples = policy.sample()
    samples_ = tf.placeholder(tf.int32, shape=[None, 1], name="samples_")
    rewards_ = tf.placeholder(tf.float32, shape=[None, 1], name="rewards_")
    log_probs = policy.log_prob(samples_)
    scaled_log_probs = rewards_ * log_probs

    parameters = tf.global_variables()

    # tf.gradients computes sum(dy/dx)
    grads = tf.gradients(scaled_log_probs, parameters)

    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    """
    The optimizer will update the variables by subtracting the gradients as it is intended to find the minimum. We
    instead seek the maximum of the expected reward, hence the negation:
    """
    grads = [-g for g in grads]
    train_op = optimizer.apply_gradients(zip(grads, parameters))

    memory = list()

    episodes = args.episodes
    max_steps = args.max_steps
    update_freq = args.update_freq

    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_actions = list()
        for step in range(max_steps):
            action = sess.run(samples, feed_dict={
                states_: np.reshape(state, newshape=[-1, env.observation_space.shape[0]])
            })
            action = action[0]
            episode_actions.append((state, action))
            state, reward, done, _ = env.step(action=action[0])
            total_reward += reward
            if done:
                break
        print("Episode {}, total reward: {}".format(episode, total_reward))
        memory.extend([(s, a, total_reward) for s, a in episode_actions])
        if episode % update_freq == 0:
            print("updating params")
            states = np.array([s for s, _, _ in memory])
            actions = np.array([a for _, a, _ in memory])
            rewards = np.array([[r] for _, _, r in memory])
            feed_dict = {
                states_: states,
                samples_: actions,
                rewards_: rewards
            }
            sess.run(train_op, feed_dict=feed_dict)
            memory = []

    # Finally render an episode
    state = env.reset()
    for step in range(max_steps):
        env.render()
        action = sess.run(samples, feed_dict={
            states_: np.reshape(state, newshape=[-1, env.observation_space.shape[0]])
        })
        action = action[0]
        state, reward, done, _ = env.step(action=action[0])
        if done:
            break


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", "--learning-rate", dest="learning_rate", type=float, default=1e-2,
                        help="Learning rate, default: %(default)s")
    parser.add_argument("-u", "--update-freq", dest="update_freq", type=int, default=10,
                        help="How often to update the policy, default: %(default)s episodes")
    parser.add_argument("-e", "--episodes", dest="episodes", type=int, default=1000,
                        help="Training episodes, default: %(default)s")
    parser.add_argument("-s", "--max-steps", dest="max_steps", type=int, default=200,
                        help="Max steps per episode, default: %(default)s")

    args = parser.parse_args()

    env = gym.make("CartPole-v0")
    train_agent(env, args)


if __name__ == "__main__":
    run()
