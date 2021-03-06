__author__ = 'philippe'
from tkinter import *
import numpy as np
import random
master = Tk()

triangle_size = 0.1
cell_score_min = -0.2
cell_score_max = 0.2
Width = 100
(x, y) = (10, 10)
actions = [0, 1, 2, 3]  # ["up", "down", "left", "right"]

board = Canvas(master, width=x*Width, height=y*Width)
player = np.array([0, 0])       # (0,0)
score = 1
restart = False
walk_reward = -0.004     #-0.004
goal = 0
max_step = 500
stepCounter = 0

specials = [(2, 4, "red", -1), (9, 8, "red", -1), (9, 9, "green", 3)]
specials_pos = []
for i, j, c, w in specials:
    specials_pos.append((i, j))
walls = [(3, 7), (7, 2)]


for i in range(0, 10):
    for j in range(0, 10):
        if i == 0:
            if j == 8:
                walls.append((i, j))
        if i == 1:
            if j == 0:
                walls.append((i, j))
            if j > 2 and j < 8:
                walls.append((i, j))
        if i == 2:
            if j == 0 or j == 5:
                walls.append((i, j))
        if i == 3:
            if j == 0 or (j > 1 and j < 6):
                walls.append((i, j))
        if i == 4:
            if j == 0 or j == 7 or j == 9:
                walls.append((i, j))
        if i == 5:
            if j == 9 or (j >= 0 and j < 8):
                walls.append((i, j))
        if i == 6:
            if j == 9:
                walls.append((i, j))
        if i == 7:
            if j == 1 or (j > 2 and j <= 7) or j == 9:
                walls.append((i, j))
        if i == 8:
            if j == 3:
                walls.append((i, j))
        if i == 9:
            if j == 5 or j == 6:
                walls.append((i, j))


cell_scores = {}


def create_triangle(i, j, action):
    if action == actions[0]:
        return board.create_polygon((i+0.5-triangle_size)*Width, (j+triangle_size)*Width,
                                    (i+0.5+triangle_size)*Width, (j+triangle_size)*Width,
                                    (i+0.5)*Width, j*Width,
                                    fill="white", width=1)
    elif action == actions[1]:
        return board.create_polygon((i+0.5-triangle_size)*Width, (j+1-triangle_size)*Width,
                                    (i+0.5+triangle_size)*Width, (j+1-triangle_size)*Width,
                                    (i+0.5)*Width, (j+1)*Width,
                                    fill="white", width=1)
    elif action == actions[2]:
        return board.create_polygon((i+triangle_size)*Width, (j+0.5-triangle_size)*Width,
                                    (i+triangle_size)*Width, (j+0.5+triangle_size)*Width,
                                    i*Width, (j+0.5)*Width,
                                    fill="white", width=1)
    elif action == actions[3]:
        return board.create_polygon((i+1-triangle_size)*Width, (j+0.5-triangle_size)*Width,
                                    (i+1-triangle_size)*Width, (j+0.5+triangle_size)*Width,
                                    (i+1)*Width, (j+0.5)*Width,
                                    fill="white", width=1)


def render_grid():
    global specials, walls, Width, x, y, player
    for i in range(x):
        for j in range(y):
            board.create_rectangle(i*Width, j*Width, (i+1)*Width, (j+1)*Width, fill="white", width=1)
            temp = {}
            for action in actions:
                temp[action] = create_triangle(i, j, action)
            cell_scores[(i,j)] = temp
    for (i, j, c, w) in specials:
        board.create_rectangle(i*Width, j*Width, (i+1)*Width, (j+1)*Width, fill=c, width=1)
    for (i, j) in walls:
        board.create_rectangle(i*Width, j*Width, (i+1)*Width, (j+1)*Width, fill="black", width=1)

render_grid()


def set_cell_score(state, action, val):
    global cell_score_min, cell_score_max
    state = (state[0], state[1])
    triangle = cell_scores[state][action]
    green_dec = int(min(255, max(0, (val - cell_score_min) * 255.0 / (cell_score_max - cell_score_min))))
    green = hex(green_dec)[2:]
    red = hex(255-green_dec)[2:]
    if len(red) == 1:
        red += "0"
    if len(green) == 1:
        green += "0"
    color = "#" + red + green + "00"
    board.itemconfigure(triangle, fill=color)


def try_move(dx, dy):
    global player, x, y, score, walk_reward, me, restart, specials, stepCounter
    if restart == True:
        restart_game()
    stepCounter += 1
    if stepCounter % 100 == 0:
        #print(stepCounter)
        pass
    if stepCounter >= max_step:
        score -= 0
        #print("Max steps overstepped, score: ", score)
        restart = True
        return
    new_x = player[0] + dx
    new_y = player[1] + dy
    score += walk_reward
    if (new_x >= 0) and (new_x < x) and (new_y >= 0) and (new_y < y) and not ((new_x, new_y) in walls):
        board.coords(me, new_x*Width+Width*2/10, new_y*Width+Width*2/10, new_x*Width+Width*8/10, new_y*Width+Width*8/10)
        player = np.array([new_x, new_y])
    counter = 0
    for (i, j, c, w) in specials:
        if new_x == i and new_y == j:
            score -= walk_reward
            score += w
            if score > 0:
                pass
                #print("Success! score: ", score)
            else:
                pass
                #print("Fail! score: ", score)
            restart = True
            return
        counter += 1
    #print "score: ", score


def call_up(event):
    try_move(0, -1)


def call_down(event):
    try_move(0, 1)


def call_left(event):
    try_move(-1, 0)


def call_right(event):
    try_move(1, 0)


def restart_game(board_rs=True):
    global player, score, me, restart, specials, stepCounter, specials_pos
    new_x = random.randint(0, 9)
    new_y = random.randint(0, 9)
    while True:
        if (new_x >= 0) and (new_x < x) and (new_y >= 0) and (new_y < y) and not ((new_x, new_y) in walls) and not ((new_x, new_y) in specials_pos):
            break
        else:
            new_x = random.randint(0, 9)
            new_y = random.randint(0, 9)
    player = np.array([new_x, new_y])
    #player = np.array([0, 0])
    stepCounter = 0
    score = 1
    restart = False
    if board_rs:
        board.coords(me, player[0]*Width+Width*2/10, player[1]*Width+Width*2/10, player[0]*Width+Width*8/10, player[1]*Width+Width*8/10)

def has_restarted():
    return restart

master.bind("<Up>", call_up)
master.bind("<Down>", call_down)
master.bind("<Right>", call_right)
master.bind("<Left>", call_left)

me = board.create_rectangle(player[0]*Width+Width*2/10, player[1]*Width+Width*2/10,
                            player[0]*Width+Width*8/10, player[1]*Width+Width*8/10, fill="orange", width=1, tag="me")

board.grid(row=0, column=0)


def start_game():
    master.mainloop()