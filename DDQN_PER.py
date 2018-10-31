import numpy as np
import tensorflow as tf
import Worlds.World as World
import threading
import time
import random
from collections import deque
import matplotlib.pyplot as plt


actions = World.actions     # ["up", "down", "left", "right"]


class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=2,
                 action_size=4, hidden_size=10,
                 name='QNetwork'):
        # state inputs to the Q-network
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, state_size], name='inputs')

            self.ISWeights_ = tf.placeholder(tf.float32, [None, 1], name='IS_weights')

            # One hot encode the actions to later choose the Q-value for the action
            self.actions_ = tf.placeholder(tf.int32, [None], name='acts')
            one_hot_actions = tf.one_hot(self.actions_, action_size)

            # Target Q values for training
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')

            # ReLU hidden layers
            self.fc1 = tf.contrib.layers.fully_connected(self.inputs_, hidden_size)
            #self.fc2 = tf.contrib.layers.fully_connected(self.fc1, hidden_size)
            #self.fc3 = tf.contrib.layers.fully_connected(self.fc2, hidden_size)

            # Linear output layer
            self.output = tf.contrib.layers.fully_connected(self.fc1, action_size,
                                                            activation_fn=None)

            ### Train with loss (targetQ - Q)^2
            # output has length 4, for four actions. This next line chooses
            # one value from output (per row) according to the one-hot encoded actions.
            self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)

            # The loss is modified b.c of PER, for updating SumTree
            self.absolute_errors = tf.abs(self.targetQs_ - self.Q)

            self.loss = tf.reduce_mean(self.ISWeights_ * tf.square(self.targetQs_ - self.Q))
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)


class SumTree(object):
    #Modified version of Morvan Zhou's sumtree: https://github.com/MorvanZhou/

    data_pointer = 0

    # Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    def __init__(self, capacity):
        self.capacity = capacity
        # Generate the tree with all nodes values = 0
        self.tree = np.zeros(2*capacity - 1)
        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)

    # Here we add our priority score in the sumtree leaf and add the experience in data
    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1

        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update(tree_index, priority)

        # Add 1 to data_pointer
        self.data_pointer += 1
        # If we're above the capacity, you go back to first index (we overwrite)
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    # Update the leaf priority score and propagate the change through tree
    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    # Here we get the leaf_index, priority value of that leaf and experience associated with that index
    def get_leaf(self, v):
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break

            else:  # downward search, always search for a higher priority node

                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index

                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]  # Returns the root node


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree

    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1

    PER_b_increment_per_sampling = 0.001

    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        # Making the tree
        self.tree = SumTree(capacity)

    # Store a new experience in our tree
    # Each new experience have a score of max_prority (it will be then improved when we use this exp to train our DDQN)
    def store(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)  # set the max p for new p

    """
    - First, to sample a minibatch of k size, the range [0, priority_total] is / into k ranges.
    - Then a value is uniformly sampled from each range
    - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
    - Then, we calculate IS weights for each minibatch element
    """

    def sample(self, n):
        # Create a sample array that will contains the minibatch
        memory_b = []

        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n  # priority segment

        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1

        # Calculating the max_weight
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * n) ** (-self.PER_b)

        for i in range(n):
            # A value is uniformly sample from each range
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            # Experience that correspond to each value is retrieved
            index, priority, data = self.tree.get_leaf(value)

            # P(j)
            sampling_probabilities = priority / self.tree.total_priority

            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b) / max_weight

            b_idx[i] = index

            experience = [data]

            memory_b.append(experience)

        return b_idx, memory_b, b_ISWeights

    # Update the priorities on the tree
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


'''
class Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False)
        return [self.buffer[ii] for ii in idx]'''


train_episodes = 1000          # max number of episodes to learn from
max_steps = 10000                # max steps in an episode
gamma = 0.99                   # future reward discount

# Exploration parameters
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability
decay_rate = 0.0001            # exponential decay rate for exploration prob

# Network parameters
hidden_size = 32               # number of units in each Q-network hidden layer
learning_rate = 0.0001         # Q-network learning rate

# Memory parameters
memory_size = 10000            # memory capacity
batch_size = 32                # experience mini-batch size
pretrain_length = 10000   # number experiences to pretrain the memory

# We update the target network with the DQNetwork every tau step
max_tau = 500


tf.reset_default_graph()
graph1 = tf.get_default_graph()
# Instantiate both DQN and and target network
with graph1.as_default():
    mainQN = QNetwork(name='DQNetwork', hidden_size=hidden_size, learning_rate=learning_rate)
    TargetNetwork = QNetwork(name='TargetNetwork', hidden_size=hidden_size, learning_rate=learning_rate)


# Thanks for the very good implementation from Arthur Juliani https://github.com/awjuliani
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


def move(act):
    s_1 = World.player
    reward = -World.score
    # Up, down, left, right
    if act == actions[0]:
        World.try_move(0, -1)
    if act == actions[1]:
        World.try_move(0, 1)
    if act == actions[2]:
        World.try_move(-1, 0)
    if act == actions[3]:
        World.try_move(1, 0)
    s_2 = World.player
    reward += World.score
    return s_1, act, reward, s_2


def populate_memory():
    # Take one random step to get moving (not necessary?)
    #_, act, reward, state = move(random.choice(actions))

    memory = Memory(memory_size)
    state = World.player

    # Make a bunch of random actions and store the experiences
    for ii in range(pretrain_length):
        if ii % 1000 == 0:
            print("Pre-training...")

        # Make a random action
        action = random.choice(actions)
        state, action, reward, next_state = move(action)

        if World.has_restarted():

            # The simulation ends so no next state
            next_state = np.zeros(state.shape)
            # Add experience to memory
            memory.store((state, action, reward, next_state))

            World.restart_game(board_rs=False)
            # Take one random step to get moving
            #_, action, reward, state = move(random.choice(actions))

        else:
            # Add experience to memory
            memory.store((state, action, reward, next_state))
            state = next_state
    return memory, state


def train(memory):
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
            state = World.player
            while t < max_steps:
                step += 1
                tau += 1

                # Explore or Exploit
                explore_p = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * step)
                if explore_p > np.random.rand():
                    # Make a random action
                    action = random.choice(actions)
                else:
                    # Get action from Q-network
                    feed = {mainQN.inputs_: state.reshape((1, *state.shape))}
                    Qs = sess.run(mainQN.output, feed_dict=feed)
                    action = np.argmax(Qs)

                # Take action, get new state and reward
                state, action, reward, next_state = move(action)

                total_reward += reward

                if World.has_restarted():
                    # the episode ends so no next state
                    next_state = np.zeros(state.shape)
                    t = max_steps

                    print('Episode: {}'.format(ep),
                          'Total reward: {:.4f}'.format(total_reward),
                          'Training loss: {:.4f}'.format(loss),
                          'Explore P: {:.4f}'.format(explore_p))
                    rewards_list.append((ep, total_reward))

                    # Add experience to memory
                    memory.store((state, action, reward, next_state))

                    # Start new episode
                    World.restart_game(board_rs=False)
                    # Take one random step to get moving?
                    #_, action, reward, state = move(random.choice(actions))

                else:
                    # Add experience to memory
                    memory.store((state, action, reward, next_state))
                    state = next_state
                    t += 1

                # Sample mini-batch from memory
                tree_idx, batch, ISWeights_mb = memory.sample(batch_size)

                states = np.array([each[0][0] for each in batch])
                acts = np.array([each[0][1] for each in batch])
                rewards = np.array([each[0][2] for each in batch])
                next_states = np.array([each[0][3] for each in batch])

                # DOUBLE DQN Logic:
                # Use DQNNetwork to select the action to take at next_state (a') (action with the highest Q-value)
                # Use TargetNetwork to calculate the Q_val of Q(s',a')

                # Calculate Q_target for all actions that state
                Qs_target_next_state = sess.run(TargetNetwork.output, feed_dict={TargetNetwork.inputs_: next_states})
                # Get Q values for next_state
                Qs_next_state = sess.run(mainQN.output, feed_dict={mainQN.inputs_: next_states})

                target_Qs_batch = []




                # Set target_Qs to r for states where episode ends
                episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
                for i in range(0, len(batch)):
                    terminal = episode_ends[i]
                    action = np.argmax(Qs_next_state[i])
                    if terminal:
                        target_Qs_batch.append(rewards[i])
                    else:
                        target = rewards[i] + gamma * Qs_target_next_state[i][action]
                        target_Qs_batch.append(target)


                #targets = rewards + gamma * np.max(target_Qs, axis=1)

                targets_mb = np.array([each for each in target_Qs_batch])

                loss, _, absolute_errors = sess.run([mainQN.loss, mainQN.opt, mainQN.absolute_errors],
                                   feed_dict={mainQN.inputs_: states,
                                              mainQN.targetQs_: targets_mb,
                                              mainQN.actions_: acts,
                                              mainQN.ISWeights_: ISWeights_mb})

                # Update priority
                memory.batch_update(tree_idx, absolute_errors)

                if tau > max_tau:
                    update_target = update_target_graph()
                    sess.run(update_target)
                    tau = 0
                    print("Model updated")

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


def run(saver):
    time.sleep(0.1)
    test_episodes = 10
    test_max_steps = 400
    with tf.Session(graph=graph1) as sess:
        saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
        for ep in range(1, test_episodes):
            t = 0
            state = World.player
            while t < test_max_steps:
                time.sleep(0.1)
                # Get action from Q-network
                feed = {mainQN.inputs_: state.reshape((1, *state.shape))}
                Qs = sess.run(mainQN.output, feed_dict=feed)
                action = np.argmax(Qs)
                print(action)

                # Take action, get new state and reward
                state, action, reward, next_state = move(action)

                if World.has_restarted():
                    t = test_max_steps
                    World.restart_game()
                    time.sleep(0.1)
                    # Take one random step to get moving?
                    #_, action, reward, state = move(random.choice(actions))

                else:
                    state = next_state
                    t += 1


def main():
    memory, state = populate_memory()
    rewards_list, saver = train(memory)
    #plot(rewards_list)

    t = threading.Thread(target=run, args=(saver,))
    #t = threading.Thread(target=train, args=(memory))
    t.daemon = True
    t.start()
    World.start_game()
    plot(rewards_list)


if __name__ == "__main__":
    main()
