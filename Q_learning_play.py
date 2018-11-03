import Worlds.World as World
import threading
import time
import random

'''
Setting the learning rate to 0 makes the agent not learn anything, 
while a factor of 1 will make the agent only consider the most recent information. 
Equally, setting the discount factor to 0 will make the agent only consider current rewards, 
while a factor of 1 make the agent take long term actions.
'''

epsilon = 1
discount = .8
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
    s_1 = (s_1[0], s_1[1])
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
    s_2 = (s_2[0], s_2[1])
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
    global epsilon
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

rewards_list = []

def run():
    global discount, epsilon
    time.sleep(0.1)
    alpha = .08               # learning rate
    t = 1                   # time
    ep = 0
    p = 0.001
    total_r = 0
    while True:
        #board = World.board_image
        #board.show()
        s = World.player
        s = (s[0], s[1])

        # choose an action:
        max_act, _ = max_Q(s)
        #max_act = policy(max_act)
        # perform action and get new state and received reward
        (s, a, r, s_2) = move(max_act)

        _, maxQ = max_Q(s_2)
        update_Q(s, a, alpha, r, discount, s_2, maxQ)
        t += 1

        total_r += r
        if World.has_restarted():
            print("World restart, score: ", total_r)
            rewards_list.append((ep, total_r))
            total_r = 0
            print(World.player)
            World.restart_game()
            #board = World.board_image
            time.sleep(0.1)
            t = 1
            epsilon *= 0.995
            epsilon = max(0.05, epsilon)
            ep += 1

        if epsilon < 0.3:
            p = 0.1
        # update learning rate
        #alpha = pow(t, -0.1)

        time.sleep(0.01)


t = threading.Thread(target=run)
t.daemon = True
t.start()
World.start_game()
