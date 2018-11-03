__author__ = 'philippe'
from tkinter import *
import numpy as np
import random
import math
from PIL import ImageGrab
master = Tk()

triangle_size = 0.1
cell_score_min = -0.2
cell_score_max = 0.2
Width = 100
(x, y) = (5, 5)
actions = [0, 1, 2, 3]  # ["up", "down", "left", "right"]

board = Canvas(master, width=x*Width, height=y*Width)
player = np.array([0, y-1])
score = 1
restart = False
walk_reward = -0.04
stepCounter = 0
max_steps = 50

walls = [(1, 1), (1, 2), (2, 1), (2, 2)]
specials = [(4, 1, "red", -1), (4, 0, "green", 2)]
specials_pos = []
for i, j, c, w in specials:
    specials_pos.append((i, j))
cell_scores = {}


def special_pos():
    global specials, specials_pos
    specials_pos = []
    for i, j, c, w in specials:
        specials_pos.append((i, j))


def get(widget):
    x2=master.winfo_rootx()+widget.winfo_x()
    y2=master.winfo_rooty()+widget.winfo_y()
    x1=x2+widget.winfo_width()
    y1=y2+widget.winfo_height()
    #.save('screenshot.png')
    return ImageGrab.grab().crop((x2,y2,x1,y1))


def init_objects():
    specials = []
    walls = []
    n_of_red_squares = 2
    prev_idx = None
    first = True
    total_objects = 0

    for i in range(0, x):
        # idx = np.random.randint(10, size=4)
        idx = np.random.choice(range(x), 3, replace=False)
        counter = 0
        counters = []
        for ii in idx:
            donot = False
            if first:
                if total_objects % 5 == 0:
                    # add red squares
                    specials.append((ii, i, "red", -1))
                    total_objects += 1
                else:
                    # add walls only if we don't trap our player
                    walls.append((ii, i))
                    total_objects += 1
            if prev_idx is not None and not first:
                for iii in prev_idx:
                    if math.fabs(ii - iii) < 2:
                        donot = True
                        if counter not in counters:
                            counters.append(counter)
                if not donot:
                    if total_objects % 5 == 0:
                        # add red squares
                        specials.append((ii, i, "red", -1))
                        total_objects += 1
                    else:
                        # add walls
                        walls.append((ii, i))
                        total_objects += 1
            counter += 1
        first = False
        deleted = False
        for c in counters:
            if deleted:
                c -= 1
                deleted = False
            if c >= len(idx):
                c = len(idx) - 1
            if len(idx) is not 0:
                idx = np.delete(idx, c)
                deleted = True
        prev_idx = idx
    return specials, walls


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
    global specials, walls, Width, x, y, player, my_green
    for i in range(x):
        for j in range(y):
            board.create_rectangle(i*Width, j*Width, (i+1)*Width, (j+1)*Width, fill="white", width=1)
            temp = {}
            for action in actions:
                temp[action] = create_triangle(i, j, action)
            cell_scores[(i,j)] = temp
    for (i, j, c, w) in specials:
        if c == "green":
            my_green = board.create_rectangle(i*Width, j*Width, (i+1)*Width, (j+1)*Width, fill=c, width=1)
        board.create_rectangle(i*Width, j*Width, (i+1)*Width, (j+1)*Width, fill=c, width=1)
    for (i, j) in walls:
        board.create_rectangle(i*Width, j*Width, (i+1)*Width, (j+1)*Width, fill="black", width=1)


render_grid()
board_image = get(board)
board_image.show()

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
    global player, x, y, score, walk_reward, me, restart, stepCounter
    if restart == True:
        restart_game()
    stepCounter += 1
    if stepCounter % 100 == 0:
        # print(stepCounter)
        pass
    if stepCounter >= max_steps:
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
    for (i, j, c, w) in specials:
        if new_x == i and new_y == j:
            #print(i, j)
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
    #print "score: ", score


def call_up(event):
    try_move(0, -1)


def call_down(event):
    try_move(0, 1)


def call_left(event):
    try_move(-1, 0)


def call_right(event):
    try_move(1, 0)


def generate_pos():
    new_x = random.randint(0, x)
    new_y = random.randint(0, y)
    while True:
        if (new_x >= 0) and (new_x < x) and (new_y >= 0) and (new_y < y) and not ((new_x, new_y) in walls) and not (
                (new_x, new_y) in specials_pos):
            break
        else:
            new_x = random.randint(0, x)
            new_y = random.randint(0, y)
    return np.array([new_x, new_y])


def restart_game(board_rs=True):
    global player, score, me, restart, stepCounter, my_green, specials, board_image
    new_x = random.randint(0, x)
    new_y = random.randint(0, y)
    specials = [(4, 1, "red", -1)]
    green = generate_pos()
    specials.append((green[0], green[1], "green", 2))
    special_pos()
    while True:
        if (new_x >= 0) and (new_x < x) and (new_y >= 0) and (new_y < y) and not ((new_x, new_y) in walls) and not (
                (new_x, new_y) in specials_pos):
            break
        else:
            new_x = random.randint(0, x)
            new_y = random.randint(0, y)
    player = generate_pos()
    #player = np.array([0, y-1])
    score = 1
    stepCounter = 0
    restart = False
    #print(specials)
    #print(specials_pos)

    if board_rs:
        board.coords(me, player[0]*Width+Width*2/10, player[1]*Width+Width*2/10, player[0]*Width+Width*8/10, player[1]*Width+Width*8/10)
        board.delete(my_green)
        my_green = board.create_rectangle(green[0] * Width, green[1] * Width, (green[0] + 1) * Width, (green[1] + 1) * Width, fill=c, width=1)
    board_image = get(board)



def has_restarted():
    return restart

master.bind("<Up>", call_up)
master.bind("<Down>", call_down)
master.bind("<Right>", call_right)
master.bind("<Left>", call_left)



me = board.create_rectangle(player[0]*Width+Width*2/10, player[1]*Width+Width*2/10,
                            player[0]*Width+Width*8/10, player[1]*Width+Width*8/10, fill="orange", width=1, tag="me")

board.grid(row=0, column=0)

board_image = get(board)
#board_image.show()



def start_game():
    master.mainloop()