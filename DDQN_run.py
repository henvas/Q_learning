import numpy as np
import tensorflow as tf
#import gym_world as gym
from Worlds.gym_world import World
import threading
import time
import random
from collections import deque
import matplotlib.pyplot as plt


class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=2,
                 action_size=4, hidden_size=10,
                 name='QNetwork'):
        # state inputs to the Q-network
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, state_size], name='inputs')

            # One hot encode the actions to later choose the Q-value for the action
            self.actions_ = tf.placeholder(tf.int32, [None], name='acts')
            one_hot_actions = tf.one_hot(self.actions_, action_size)

            # Target Q values for training
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')

            # ReLU hidden layers
            self.fc1 = tf.contrib.layers.fully_connected(self.inputs_, hidden_size)
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, hidden_size)
            self.fc3 = tf.contrib.layers.fully_connected(self.fc2, hidden_size)

            # Linear output layer
            self.output = tf.contrib.layers.fully_connected(self.fc3, action_size,
                                                            activation_fn=None)

            ### Train with loss (targetQ - Q)^2
            # output has length 4, for four actions. This next line chooses
            # one value from output (per row) according to the one-hot encoded actions.
            self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)

            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Q))
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)


class Memory():
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False)
        return [self.buffer[ii] for ii in idx]


train_episodes = 500          # max number of episodes to learn from
max_steps = 10000                # max steps in an episode
gamma = 0.99                   # future reward discount

# Exploration parameters
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability
decay_rate = 0.00006            # exponential decay rate for exploration prob

# Network parameters
hidden_size = 512               # number of units in each Q-network hidden layer
learning_rate = 0.0001         # Q-network learning rate

# Memory parameters
memory_size = 10000            # memory capacity
batch_size = 32                # experience mini-batch size
pretrain_length = 10000   # number experiences to pretrain the memory

# We update the target network with the DQNetwork every tau step
max_tau = 100


tf.reset_default_graph()
graph1 = tf.get_default_graph()
with graph1.as_default():
    mainQN = QNetwork(name='DQNetwork', hidden_size=hidden_size, learning_rate=learning_rate)
    TargetNetwork = QNetwork(name='TargetNetwork', hidden_size=hidden_size, learning_rate=learning_rate)


def update_target_graph():
    # Get the parameters of our DQNNetwork
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")

    # Get the parameters of our Target_network
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")

    op_holder = []

    # Update our target_network parameters with DQNNetwork parameters
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def populate_memory(env):
    # Take one random step
    state, reward, done = env.step(random.choice(env.ACTIONS))

    memory = Memory(max_size=memory_size)

    # Make a bunch of random actions and store the experiences
    for ii in range(pretrain_length):
        if ii % 1000 == 0:
            print("Pre-training...")

        # Make a random action
        state = env.Player
        action = random.choice(env.ACTIONS)
        next_state, reward, done = env.step(action)

        if done:

            # The simulation fails so no next state
            next_state = np.zeros(state.shape)
            # Add experience to memory
            memory.add((state, action, reward, next_state))

            env.restart_game()
            # Take one random step to get moving?
            #state, reward, done = env.step(random.choice(env.ACTIONS))
        else:
            # Add experience to memory
            memory.add((state, action, reward, next_state))
            state = next_state
    return memory, state


def render_game(game):
    game.render_grid()
    game.board.grid(row=0, column=0)
    base_1 = game.WIDTH + (game.WIDTH * 0.2)
    base_2 = game.WIDTH + (game.WIDTH * 0.8)
    #print("Player: ", game.Player)

    game.me = game.board.create_rectangle(game.Player[0] * base_1, game.Player[1] * base_2,
                                          game.Player[0] * base_1, game.Player[1] * base_2,
                                          fill="orange", width=1, tag="me")

def train(env, memory):
    time.sleep(0.1)
    # Now train with experiences
    saver = tf.train.Saver()
    rewards_list = []
    with tf.Session(graph=graph1) as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())

        tau = 0

        update_target = update_target_graph()
        sess.run(update_target)

        step = 0
        for ep in range(1, train_episodes):
            total_reward = 0
            t = 0
            state = env.Player
            while t < max_steps:
                step += 1
                tau += 1
                # Uncomment this next line to watch the training
                # env.render()

                # Explore or Exploit
                explore_p = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * step)
                if explore_p > np.random.rand():
                    # Make a random action
                    action = random.choice(env.ACTIONS)
                else:
                    # Get action from Q-network
                    feed = {mainQN.inputs_: state.reshape((1, *state.shape))}
                    Qs = sess.run(mainQN.output, feed_dict=feed)
                    action = np.argmax(Qs)

                # Take action, get new state and reward
                next_state, reward, done = env.step(action)
                #print(done)

                total_reward += reward

                if done:
                    # the episode ends so no next state
                    next_state = np.zeros(state.shape)
                    t = max_steps

                    print('Episode: {}'.format(ep),
                          'Total reward: {:.4f}'.format(total_reward),
                          'Training loss: {:.4f}'.format(loss),
                          'Explore P: {:.4f}'.format(explore_p))
                    rewards_list.append((ep, total_reward))

                    # Add experience to memory
                    memory.add((state, action, reward, next_state))

                    # Start new episode
                    env.restart_game()
                    # Uncomment this next line to watch the training
                    render_game(env)
                    # Take one random step
                    #_state, reward, done = env.step(random.choice(env.ACTIONS))

                else:
                    # Add experience to memory
                    memory.add((state, action, reward, next_state))
                    state = next_state
                    t += 1

                # Sample mini-batch from memory
                batch = memory.sample(batch_size)
                states = np.array([each[0] for each in batch])
                acts = np.array([each[1] for each in batch])
                rewards = np.array([each[2] for each in batch])
                next_states = np.array([each[3] for each in batch])

                ### DOUBLE DQN Logic
                # Use DQNNetwork to select the action to take at next_state (a') (action with the highest Q-value)
                # Use TargetNetwork to calculate the Q_val of Q(s',a')

                # Calculate Q_target for all actions that state
                Qs_target_next_state = sess.run(TargetNetwork.output, feed_dict={TargetNetwork.inputs_: next_states})
                # Get Q values for next_state
                Qs_next_state = sess.run(mainQN.output, feed_dict={mainQN.inputs_: next_states})

                target_Qs_batch = []




                # Set target_Qs to 0 for states where episode ends
                episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
                for i in range(0, len(batch)):
                    terminal = episode_ends[i]
                    action = np.argmax(Qs_next_state[i])
                    if terminal:
                        target_Qs_batch.append(rewards[i])
                    else:
                        target = rewards[i] + gamma * Qs_target_next_state[i][action]
                        target_Qs_batch.append(target)

                #target_Qs[episode_ends] = (0, 0, 0, 0)

                #targets = rewards + gamma * np.max(target_Qs, axis=1)

                targets_mb = np.array([each for each in target_Qs_batch])

                loss, _ = sess.run([mainQN.loss, mainQN.opt],
                                   feed_dict={mainQN.inputs_: states,
                                              mainQN.targetQs_: targets_mb,
                                              mainQN.actions_: acts})

                if tau > max_tau:
                    print("Model updated")
                    tau = 0
                    update_target = update_target_graph()
                    sess.run(update_target)

        save_path = saver.save(sess, "checkpoints/model_dqn.ckpt")
        print("Model saved in path: %s" % save_path)
    return rewards_list, saver


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


def plot(rewards_list):
    eps, rews = np.array(rewards_list).T
    smoothed_rews = running_mean(rews, 10)
    plt.plot(eps[-len(smoothed_rews):], smoothed_rews)
    plt.plot(eps, rews, color='grey', alpha=0.3)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()


def run(env, saver):
    time.sleep(0.1)
    test_episodes = 10
    test_max_steps = 60
    with tf.Session(graph=graph1) as sess:
        saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
        for ep in range(1, test_episodes):
            t = 0
            state = env.Player
            while t < test_max_steps:
                # Get action from Q-network
                feed = {mainQN.inputs_: state.reshape((1, *state.shape))}
                Qs = sess.run(mainQN.output, feed_dict=feed)
                action = np.argmax(Qs)
                print(action)

                # Take action, get new state and reward
                next_state, reward, done = env.step(action)

                if done:
                    t = test_max_steps
                    #time.sleep(0.1)
                    env.restart_game()
                    render_game(env)
                    print(env.Player)
                    print(env.OBJECTS)
                    print(env.WALLS)
                    # Take one random step to get the pole and cart moving
                    # _state, reward, done = env.step(random.choice(env.ACTIONS))

                else:
                    state = next_state
                    t += 1
                time.sleep(0.1)


def main():
    env = World(100, 10, 10)

    memory, state = populate_memory(env)
    rewards_list, saver = train(env, memory)
    #plot(rewards_list)

    render_game(env)
    t = threading.Thread(target=run, args=(env, saver))
    #t = threading.Thread(target=train, args=(env, memory))
    t.daemon = True
    t.start()
    env.start_game()
    plot(rewards_list)


if __name__ == "__main__":
    main()
