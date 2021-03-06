import os
from random import sample as rsample
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD, RMSprop
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
import json


GRID_SIZE = 10


def episode():
    """
    Coroutine function for an episode.
        Action has to be explicitly sent (via "send") to this co-routine.
    """
    x, y, x_basket = (
        np.random.randint(0, GRID_SIZE),  # X of fruit
        0,  # Y of dot
        np.random.randint(1, GRID_SIZE - 1))  # X of basket

    while True:
        # Reset grid
        X = np.zeros((GRID_SIZE, GRID_SIZE))
        # Draw the fruit in the screen
        X[y, x] = 1.
        # Draw the basket
        bar = range(x_basket - 1, x_basket + 2)
        X[-1, bar] = 1.

        # End of game is known when fruit is at penultimate line of grid.
        # End represents either the reward (a win or a loss)
        end = int(y >= GRID_SIZE - 2)
        if end and x not in bar:
            end *= -1

        action = yield np.array(X).reshape(GRID_SIZE, GRID_SIZE, 1), end
        if end:
            break

        x_basket = min(max(x_basket + action, 1), GRID_SIZE - 2)
        y += 1


def experience_replay(batch_size):
    """
    Coroutine function for implementing experience replay.
        Provides a new experience by calling "send", which in turn yields
        a random batch of previous replay experiences.
    """
    memory = []
    while True:
        # experience is a tuple containing (S, action, reward, S_prime)
        experience = yield rsample(memory, batch_size) if batch_size <= len(memory) else None
        memory.append(experience)


def save_img():
    """
    Coroutine to store images in the "images" directory
    """
    if 'images' not in os.listdir('.'):
        os.mkdir('images')
    frame = 0
    while True:
        screen = (yield)
        plt.imshow(screen[0], interpolation='none')
        plt.savefig('images/%03i.png' % frame)
        frame += 1


nb_epochs = 50
batch_size = 128
epsilon = .1
gamma = .7

# Recipe of deep reinforcement learning model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(GRID_SIZE, GRID_SIZE, 1)))
model.add(Activation("relu"))
model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(3))
model.compile(RMSprop(), 'MSE')
model.summary()

#################
# RELOAD A MODEL
#################
# model = model_from_json(open('model.json').read())
# model.load_weights('model.h5')

exp_replay = experience_replay(batch_size)
next(exp_replay)  # Start experience-replay coroutine

for i in range(nb_epochs):
    ep = episode()
    S, reward = next(ep)  # Start coroutine of single entire episode
    loss = 0.
    try:
        while True:
            action = np.random.randint(-1, 2)
            if np.random.random() > epsilon:
                # Get the index of the maximum q-value of the model.
                # Subtract one because actions are either -1, 0, or 1
                action = np.argmax(model.predict(S[np.newaxis]), axis=-1)[0] - 1
                '''
                print(action)
                print(S[np.newaxis])
                print(np.argmax(model.predict(S[np.newaxis]), axis=-1))
                print(model.predict(S[np.newaxis]))'''

            S_prime, reward = ep.send(action)
            experience = (S, action, reward, S_prime)
            S = S_prime

            batch = exp_replay.send(experience)
            if batch:
                inputs = []
                targets = []
                for s, a, r, s_prime in batch:
                    # The targets of unchosen actions are the q-values of the model,
                    # so that the corresponding errors are 0. The targets of chosen actions
                    # are either the rewards, in case a terminal state has been reached,
                    # or future discounted q-values, in case episodes are still running.
                    t = model.predict(s[np.newaxis]).flatten()
                    t[a + 1] = r
                    if not r:
                        t[a + 1] = r + gamma * model.predict(s_prime[np.newaxis]).max(axis=-1)
                    targets.append(t)
                    inputs.append(s)

                loss += model.train_on_batch(np.array(inputs), np.array(targets))

    except StopIteration:
        pass

    # if (i + 1) % 100 == 0:
    print('Epoch %i, loss: %.6f' % (i + 1, loss))

#################
# SAVE THE MODEL
#################
# Save trained model weights and architecture, this will be used by the visualization code
model_name = "model.h5"
print("Saving the model to " + model_name)
json_string = model.to_json()
open('model.json', 'w').write(json_string)
model.save_weights(model_name)

#################
# TEST
#################
img_saver = save_img()
next(img_saver)

epsilon = 0

for _ in range(10):
    g = episode()
    S, _ = next(g)
    S = np.array(S).reshape(1, GRID_SIZE, GRID_SIZE)
    img_saver.send(S)
    try:
        while True:
            S = np.array(S).reshape(GRID_SIZE, GRID_SIZE, 1)
            act = np.argmax(model.predict(S[np.newaxis]), axis=-1)[0] - 1
            S, _ = g.send(act)
            S = np.array(S).reshape(1, GRID_SIZE, GRID_SIZE)
            img_saver.send(S)

    except StopIteration:
        pass

img_saver.close()
