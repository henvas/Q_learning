import numpy as np
from tkinter import *
import math
import random

master = Tk()


class World(object):
    def __init__(self, width, axis_x, axis_y):
        # static
        self.WIDTH = width
        (self.AXIS_X, self.AXIS_Y) = (axis_x, axis_y)
        self.WALK_REWARD = -0.04
        #self.ACTIONS = [[0, 0, -1], [1, 0, 1], [2, -1, 0], [3, 1, 0]]
        self.ACTIONS = [0, 1, 2, 3]
        self.N_OBJECTS = round((axis_x * axis_y) / 4)
        self.stepCounter = 0

        # Init objects, then add special green square
        self.OBJECTS, self.SPECIALS, self.WALLS = self.init_objects()
        x_green, y_green = self.gen_pos()
        self.OBJECTS.append((x_green, y_green, "green", 1))
        self.SPECIALS.append((x_green, y_green, "green", 1))
        x_green, y_green = self.gen_pos()
        self.OBJECTS.append((x_green, y_green, "green", 1))
        self.SPECIALS.append((x_green, y_green, "green", 1))

        # init remaining variables
        self.board = Canvas(master, width=self.AXIS_X * self.WIDTH, height=self.AXIS_Y * self.WIDTH)
        self.score = 1
        self.restart = False
        self.Player = self.gen_pos()
        self.me = self.board.create_rectangle(self.Player[0] * self.WIDTH + self.WIDTH * 2 / 10,
                                              self.Player[1] * self.WIDTH + self.WIDTH * 2 / 10,
                                              self.Player[0] * self.WIDTH + self.WIDTH * 8 / 10,
                                              self.Player[1] * self.WIDTH + self.WIDTH * 8 / 10, fill="orange", width=1,
                                              tag="me")

    # May have swapped x and y axis, check
    def init_objects(self):
        objects = set([])
        specials = set([])
        walls = set([])
        n_of_red_squares = round(self.N_OBJECTS / 5)
        prev_idx = None
        first = True
        total_objects = 0

        for i in range(0, self.AXIS_X):
            #idx = np.random.randint(10, size=4)
            idx = np.random.choice(range(self.AXIS_X), 3, replace=False)
            counter = 0
            counters = []
            for ii in idx:
                donot = False
                if first:
                    if total_objects % 5 == 0:
                        # add red squares
                        objects.add((ii, i, "red", -1))
                        specials.add((ii, i, "red", -1))
                        total_objects += 1
                    else:
                        # add walls only if we don't trap our player
                        walls.add((ii, i))
                        objects.add((ii, i, "black", -1))
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
                            specials.add((ii, i, "red", -1))
                            objects.add((ii, i, "red", -1))
                            total_objects += 1
                        else:
                            # add walls
                            walls.add((ii, i))
                            objects.add((ii, i, "black", -1))
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
        return list(objects), list(specials), list(walls)

    def init_objects_terrible(self):
        objects = set([])
        n_of_red_squares = round(self.N_OBJECTS / 5)

        for i in range(self.N_OBJECTS):
            # First add N amount of walls, then add N amount of OBJECTS to objects
            if i < n_of_red_squares:
                # add red squares
                objects.add((np.random.randint(0, self.AXIS_X), np.random.randint(0, self.AXIS_Y), "red", -1))
            else:
                # add walls
                objects.add((np.random.randint(0, self.AXIS_X), np.random.randint(0, self.AXIS_Y), "black", -1))

        return list(objects)

    def gen_pos(self):
        specials_pos = []
        for i, j, c, w in self.SPECIALS:
            specials_pos.append((i, j))
        new_x = random.randint(0, 9)
        new_y = random.randint(0, 9)
        while True:
            if (new_x >= 0) and (new_x < self.AXIS_X) and (new_y >= 0) and (new_y < self.AXIS_Y) and not ((new_x, new_y) in self.WALLS) and not (
                    (new_x, new_y) in specials_pos):
                break
            else:
                new_x = random.randint(0, self.AXIS_X)
                new_y = random.randint(0, self.AXIS_Y)
        return np.array([new_x, new_y])
        #'''while True:
        #    x = np.random.randint(0, self.AXIS_X)
        #    y = np.random.randint(0, self.AXIS_Y)
        #    check_for_rows = [e for e in self.OBJECTS if x == e[0] and y == e[1]]
        #
        #    if not check_for_rows:
        #        break
        #return np.array([x, y])'''

    def render_grid(self):
        for i in range(self.AXIS_X):
            for j in range(self.AXIS_Y):
                # Check if objects for current coordinates
                objects = [e for e in self.OBJECTS if i == e[0] and j == e[1]]
                if objects and ((i, j) == objects[0][0:2]):
                    self.board.create_rectangle(i * self.WIDTH, j * self.WIDTH, (i + 1) * self.WIDTH,
                                                (j + 1) * self.WIDTH, fill=objects[0][2], width=1)
                else:
                    self.board.create_rectangle(i * self.WIDTH, j * self.WIDTH, (i + 1) * self.WIDTH,
                                                (j + 1) * self.WIDTH, fill="white", width=1)

    def start_game(self):
        master.mainloop()

    def try_move(self, dx, dy):
        self.stepCounter += 1
        max_step = 70
        if self.stepCounter >= max_step:
            self.score -= 0
            #print("Max steps overstepped, score: ", self.score)
            self.restart = True
            return self.restart
        if self.restart:
            self.restart_game()
        new_x = self.Player[0] + dx
        new_y = self.Player[1] + dy
        self.score += self.WALK_REWARD
        #print("x, y = ", (new_x, new_y))
        #print(self.OBJECTS)
        #objects = [e for e in self.OBJECTS if new_x == e[0] and new_y == e[1]]
        if (new_x >= 0) and (new_x < self.AXIS_X) and (new_y >= 0) and (new_y < self.AXIS_Y) and not (
                (new_x, new_y) in self.WALLS):
            self.board.coords(self.me, new_x * self.WIDTH + self.WIDTH * 2 / 10,
                              new_y * self.WIDTH + self.WIDTH * 2 / 10,
                              new_x * self.WIDTH + self.WIDTH * 8 / 10, new_y * self.WIDTH + self.WIDTH * 8 / 10)

            self.Player = np.array([new_x, new_y])

        for (i, j, c, w) in self.SPECIALS:
            if c == "black":
                continue
            if new_x == i and new_y == j and c != "black":
                self.score -= self.WALK_REWARD
                self.score += w
                self.restart = True
                if c == "green":
                    pass
                    #print("Green")
                return self.restart
        #print("self.restart: ", self.restart)
        return self.restart

    def step(self, action):
        reward = -self.score

        actions = [[0, 0, -1], [1, 0, 1], [2, -1, 0], [3, 1, 0]]

        coords = [i for i in actions if i[0] == action]

        x, y = coords[0][1:]
        done = self.try_move(x, y)
        #print("deone", done)

        state = self.Player
        reward += self.score
        return state, reward, done

    def restart_game(self):
        self.score = 1
        self.stepCounter = 0
        self.restart = False
        # Init objects, then add special green square
        self.OBJECTS, self.SPECIALS, self.WALLS = self.init_objects()
        x_green, y_green = self.gen_pos()
        self.OBJECTS.append((x_green, y_green, "green", 1))
        self.SPECIALS.append((x_green, y_green, "green", 1))
        x_green, y_green = self.gen_pos()
        self.OBJECTS.append((x_green, y_green, "green", 1))
        self.SPECIALS.append((x_green, y_green, "green", 1))
        self.Player = self.gen_pos()
        self.board.coords(self.me, self.Player[0] * self.WIDTH + self.WIDTH * 2 / 10,
                          self.Player[1] * self.WIDTH + self.WIDTH * 2 / 10,
                          self.Player[0] * self.WIDTH + self.WIDTH * 8 / 10,
                          self.Player[1] * self.WIDTH + self.WIDTH * 8 / 10)
        #self.render_grid()

    def has_restarted(self):
        return self.restart

