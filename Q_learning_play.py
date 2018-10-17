import World
import threading
import time
import random

epsilon = 0.2
discount = 0.3
actions = World.actions     # ["up", "down", "left", "right"]
states = []                 # states = (x, y) location
Q = {}

# Set state-coordinates
for i in range(World.x):
    for j in range(World.y):
        states.append((i, j))

# Init Q matrix and set colors for cells
for state in states:
    temp = {}
    for action in actions:
        temp[action] = 0.1
        World.set_cell_score(state, action, temp[action])
    Q[state] = temp

for (i, j, c, w) in World.specials:
    for action in actions:
        Q[(i, j)][action] = w
        World.set_cell_score((i, j), action, w)


def move(action):
    s_1 = World.player
    reward = -World.score
    # Up, down, left, right
    if action == actions[0]:
        World.try_move(0, -1)
    if action == actions[1]:
        World.try_move(0, 1)
    if action == actions[2]:
        World.try_move(-1, 0)
    if action == actions[3]:
        World.try_move(1, 0)
    s_2 = World.player
    reward += World.score
    return s_1, action, reward, s_2


def max_Q(s):
    val = None
    move = None
    for a, q in Q[s].items():
        if val is None or q > val:
            val = q
            move = a
    return move, val


def policy(max_act):
    if random.random() > epsilon:
        return max_act
    else:
        random_idx = random.randint(0, len(actions)-2)
        if actions[random_idx] == max_act:
            return actions[len(actions)-1]
        else:
            return actions[random_idx]


def update_Q(s, a, alpha, reward, gamma, s_2, maxQ):
    Q[s][a] = (1 - alpha)*Q[s][a] + alpha*(reward + gamma*maxQ)
    World.set_cell_score(s, a, Q[s][a])


def run():
    global discount
    time.sleep(0.1)
    alpha = 1               # learning rate
    t = 1                   # time
    while True:
        s = World.player

        # choose an action:
        max_act, _ = max_Q(s)
        #max_act = policy(max_act)
        # perform action and get new state and received reward
        (s, a, r, s_2) = move(max_act)

        _, maxQ = max_Q(s_2)
        update_Q(s, a, alpha, r, discount, s_2, maxQ)

        t += 1
        if World.has_restarted():
            World.restart_game()
            time.sleep(0.1)
            t = 1

        # update learning rate
        alpha = pow(t, -0.1)

        time.sleep(0.1)


t = threading.Thread(target=run)
t.daemon = True
t.start()
World.start_game()
