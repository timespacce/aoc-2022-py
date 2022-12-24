import math
import math

import numpy as np


def load_data():
    s = open('data/input_24', 'r')

    def transform(c):
        if c == '.':
            return 0
        elif c == '#':
            return 1
        elif c == '>':
            return 2
        elif c == '<':
            return 3
        elif c == '^':
            return 4
        elif c == 'v':
            return 5

    grid = np.array([[transform(c) for c in list(row.rstrip())] for row in s.readlines()])
    h, w = grid.shape
    s.close()

    blizzards_a, blizzards_b = np.where(grid >= 2)
    blizzards = list(zip(blizzards_a, blizzards_b))
    blizzards = [(y + 1, x + 1, grid[y, x]) for y, x in blizzards]
    h, w = h + 2, w + 2
    grid = np.zeros((h, w))
    for y, x, b in blizzards:
        grid[y, x] = b
    grid[0:2, :] = grid[:, 0:2] = grid[:, -2:] = grid[-2:, :] = 1
    grid[1, 2] = grid[h - 2, w - 3] = 0

    return grid, blizzards


grid, blizzards = load_data()
h, w = grid.shape

blizzard_to_v = {2: (0, 1), 3: (0, -1), 4: (-1, 0), 5: (1, 0)}
tick_to_grid = {0: grid}


def simulate(h, w, blizzards, ticks):
    global blizzard_to_v
    next_blizzards = blizzards.copy()
    for _ in range(ticks):
        for i, (y, x, b) in enumerate(next_blizzards):
            dy, dx = blizzard_to_v[b]
            next_y, next_x = y + dy, x + dx
            if next_y >= h - 2:
                next_y = 2
            elif next_y <= 1:
                next_y = h - 3

            if next_x >= w - 2:
                next_x = 2
            elif next_x <= 1:
                next_x = w - 3
            next_blizzards[i] = (next_y, next_x, b)

    output_grid = np.zeros((h, w))
    for y, x, b in next_blizzards:
        output_grid[y, x] = b
    output_grid[0:2, :] = output_grid[:, 0:2] = output_grid[:, -2:] = output_grid[-2:, :] = 1
    output_grid[1, 2] = output_grid[h - 2, w - 3] = 0
    return output_grid


# simulate(h, w, blizzards, 4)

vs = [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)]


def lcm(a, b):
    return (a * b) // math.gcd(a, b)


def bfs(start, target):
    global iter
    min_steps, iter = 1e10, 0
    minimum_steps_per_position = {}
    queue, target = [start], target
    while queue:
        y, x, o = queue.pop(0)
        next_tick = o + 1
        modulo_tick = next_tick % lcm(h - 4, w - 4)
        #
        if o >= minimum_steps_per_position.get((y, x, modulo_tick), 1e10):
            continue
        minimum_steps_per_position[(y, x, modulo_tick)] = o
        #
        if (y, x) == target:
            min_steps = min(min_steps, o)
            continue
        #
        if modulo_tick not in tick_to_grid:
            tick_to_grid[modulo_tick] = simulate(h, w, blizzards, modulo_tick)
        current_grid = tick_to_grid[modulo_tick]
        #
        for dy, dx in vs:
            next_y, next_x = y + dy, x + dx
            if current_grid[next_y, next_x] == 0:
                queue.append((next_y, next_x, next_tick))

        print(f'{iter:>5} {o:>5} : {len(queue):>5} : {min_steps} {len(tick_to_grid)}')
        iter += 1

    print(f'MIN_STEPS = {min_steps}')
    return min_steps


x1, y1, x2, y2 = 1, 2, h - 2, w - 3
s1 = bfs(start=(x1, y1, 0), target=(x2, y2))
s2 = bfs(start=(x2, y2, s1), target=(x1, y1))  # 274
s3 = bfs(start=(x1, y1, s2), target=(x2, y2))  # 568
