import tensorflow as tf
import gym
import numpy as np


episodes = 1000
memory_size = 5000
batch_size = 32
synch_frequency = 100  # in episodes
gamma = .99  # discount rate

epsilon = 1.0
min_epsilon = 0.05
epsilon_decay = 0.99


def build_net(inputs, layers):
    for units, activation in layers:
        inputs = tf.layers.dense(inputs=inputs, units=units, activation=activation)
    return inputs


def update_memory(memory, item):
    if len(memory) < memory_size:
        memory.append(item)
    else:
        index = np.random.randint(0, memory_size)
        memory[index] = item


def train_step(memory, sess, target_net, x_, actions_, expected_, loss, train_op):
    """
    Selects batch_size samples from the memory and trains the network to match the expected q-values.

    :param memory:
    :param sess:
    :param target_net:
    :param x_: The placeholder for the states
    :param actions_: The placeholder for the actions
    :param expected_: The placeholder for the expected q-values
    :param loss: The tensor computing the loss
    :param train_op: The tensor for the training operation
    :return: The loss value.
    """
    indices = np.random.permutation(len(memory))[:batch_size]
    data = [memory[index] for index in indices]
    inputs = np.array(list(map(lambda x: x[0], data)))
    actions = np.array(list(map(lambda x: x[1], data))).reshape([-1, 1])
    rewards = np.array(list(map(lambda x: x[2], data))).reshape([-1, 1])
    new_states = np.array(list(map(lambda x: x[3], data)))
    terminated = np.array(list(map(lambda x: x[4], data))).reshape([-1, 1])
    next_q = np.max(sess.run(target_net, feed_dict={x_: new_states}), axis=1).reshape([-1, 1])
    expected = np.where(terminated, np.zeros_like(next_q), rewards + gamma * next_q)
    loss_value, _ = sess.run([loss, train_op], feed_dict={
        x_: inputs,
        actions_: actions,
        expected_: expected
    })
    return loss_value


epochs_per_train_step = 100


def run():
    global epsilon

    env = gym.make("CartPole-v0")
    spec = [(8, tf.nn.relu), (env.action_space.n, None)]  # The network specification

    x_ = tf.placeholder(dtype=tf.float32, shape=[None, env.observation_space.shape[0]])
    with tf.variable_scope(name_or_scope="net") as net_scope:
        net = build_net(x_, spec)
    with tf.variable_scope(name_or_scope="target_net") as target_net_scope:
        target_net = build_net(x_, spec)

    # Synch networks
    net_vars = net_scope.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    target_net_vars = target_net_scope.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    memory = list()  # of state, action, reward, state', done
    synch_ops = [tf.assign(t, v) for t, v in zip(target_net_vars, net_vars)]

    # Loss
    actions_ = tf.placeholder(dtype=tf.int32, shape=[None, 1])
    expected_ = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="expected_")
    loss = tf.losses.mean_squared_error(expected_, tf.gather(net, tf.reshape(actions_, shape=[-1]), axis=1))
    optimizer = tf.train.AdamOptimizer(learning_rate=.001)
    train_op = optimizer.minimize(loss)

    init_op = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init_op)
    max_steps = env.spec.max_episode_steps - 1
    all_rewards = []
    for episode in range(episodes):
        state = env.reset()
        reward = 0.0
        for _ in range(max_steps):
            if np.random.uniform() < epsilon:
                action = env.action_space.sample()
            else:
                values = sess.run(net, feed_dict={x_: [state]})
                action = np.argmax(values)
            new_state, r, done, _ = env.step(action)
            reward += r
            mem_tuple = (state, action, r, new_state, done)
            update_memory(memory, mem_tuple)

            if done:
                break
            state = new_state

        all_rewards.append(reward)
        if len(memory) >= batch_size:
            for _ in range(epochs_per_train_step):
                train_step(
                    memory=memory,
                    sess=sess,
                    target_net=target_net,
                    x_=x_,
                    actions_=actions_,
                    expected_=expected_,
                    loss=loss,
                    train_op=train_op
                )
        if episode % synch_frequency == 0:
            print("Episode %d, average reward: %f" % (episode, np.mean(all_rewards[-100:])))
            sess.run(synch_ops)
        if epsilon <= min_epsilon:
            epsilon = min_epsilon
        else:
            epsilon *= epsilon_decay

    reward = 0
    state = env.reset()
    for _ in range(env.spec.max_episode_steps):
        env.render()
        values = sess.run(net, feed_dict={x_: [state]})
        action = np.argmax(values)
        state, r, done, _ = env.step(action)
        reward += r
        if done:
            break
    print("Final reward: %f" % reward)


if __name__ == "__main__":
    run()
