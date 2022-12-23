import re
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    s = open('data/input_22', 'r')
    rows, moves = s.read().split('\n\n')
    rows = [list(row) for row in rows.split('\n')]
    s.close()

    w = max([len(row) for row in rows])
    grid = np.ones((len(rows), w))
    for i, row in enumerate(rows):
        for j in range(w):
            if j >= len(row):
                grid[i, j] = 2
            else:
                grid[i, j] = 2 if row[j] == ' ' else 1 if row[j] == '#' else 0

    x_ranges = {}
    for i in range(len(rows)):
        min_x, max_x = 0, 0
        for j in range(w):
            if grid[i, j] == 0 or grid[i, j] == 1:
                min_x = j
                break
        for j in range(w):
            if grid[i, j] == 0 or grid[i, j] == 1:
                max_x = j
        x_ranges[i] = (min_x, max_x)

    y_ranges = {}
    for j in range(w):
        min_y, max_y = 0, 0
        for i in range(len(rows)):
            if grid[i, j] == 0 or grid[i, j] == 1:
                min_y = i
                break
        for i in range(len(rows)):
            if grid[i, j] == 0 or grid[i, j] == 1:
                max_y = i
        y_ranges[j] = (min_y, max_y)

    moves = re.findall('[A-Z]|[0-9]+', moves.rstrip())
    return grid, x_ranges, y_ranges, moves


def rotate(direction, move):
    dy, dx = direction
    if move == 'R':
        dy, dx = (dx, -dy)
    elif move == 'L':
        dy, dx = (-dx, dy)
    return dy, dx


def part_1():
    grid, x_ranges, y_ranges, moves = load_data()
    current_position, (dy, dx) = (0, x_ranges[0][0]), (0, 1)

    for move in moves:
        if move == 'R':
            dy, dx = (dx, -dy)
        elif move == 'L':
            dy, dx = (-dx, dy)
        else:
            move_count = int(move)
            for i in range(move_count):
                y, x = current_position
                y1, x1 = (y + dy, x + dx)
                ##
                (min_x, max_x), (min_y, max_y) = x_ranges[y], y_ranges[x]
                if x1 > max_x:
                    x1 = min_x
                if x1 < min_x:
                    x1 = max_x
                if y1 > max_y:
                    y1 = min_y
                if y1 < min_y:
                    y1 = max_y
                ##
                if grid[y1, x1] == 1:
                    break
                else:
                    if grid[y1, x1] == 0:
                        current_position = (y1, x1)

    y, x = current_position
    direction_score = 0 if (dy, dx) == (0, 1) else 1 if (dy, dx) == (1, 0) else 2 if (dy, dx) == (0, -1) else 3 if (dy, dx) == (-1, 0) else 0
    score = (y + 1) * 1000 + (x + 1) * 4 + direction_score
    print(f'{y + 1}, {x + 1} :  {score}')
    return


def part_2():
    grid, x_ranges, y_ranges, moves = load_data()
    current_position, (dy, dx) = (0, x_ranges[0][0]), (0, 1)

    h, w = grid.shape
    # b_size = 4
    b_size = 50
    quadrants = np.zeros((h, w), dtype=np.int)
    # quadrant_ranges = [
    #     ((0 * b_size, 1 * b_size), (2 * b_size, 3 * b_size)),
    #     ((1 * b_size, 2 * b_size), (0 * b_size, 1 * b_size)),
    #     ((1 * b_size, 2 * b_size), (1 * b_size, 2 * b_size)),
    #     ((1 * b_size, 2 * b_size), (2 * b_size, 3 * b_size)),
    #     ((2 * b_size, 3 * b_size), (2 * b_size, 3 * b_size)),
    #     ((2 * b_size, 3 * b_size), (3 * b_size, 4 * b_size)),
    # ]
    quadrant_ranges = [
        ((0 * b_size, 1 * b_size), (1 * b_size, 2 * b_size)),
        ((0 * b_size, 1 * b_size), (2 * b_size, 3 * b_size)),

        ((1 * b_size, 2 * b_size), (1 * b_size, 2 * b_size)),

        ((2 * b_size, 3 * b_size), (0 * b_size, 1 * b_size)),
        ((2 * b_size, 3 * b_size), (1 * b_size, 2 * b_size)),

        ((3 * b_size, 4 * b_size), (0 * b_size, 1 * b_size)),
    ]
    for i, ((y1, y2), (x1, x2)) in enumerate(quadrant_ranges):
        quadrants[y1:y2, x1:x2] = (i + 1)

    v_to_code = {(0, 1): 'R', (0, -1): 'L', (1, 0): 'B', (-1, 0): 'U'}
    code_to_v = {'R': (0, 1), 'L': (0, -1), 'B': (1, 0), 'U': (-1, 0)}
    opposite_v = {'R': 'L', 'L': 'R', 'U': 'B', 'B': 'U'}
    path = np.zeros((h, w), 'U1')

    def translate(c1, position, next_quadrant, c2):
        nonlocal quadrant_ranges, v_to_code, code_to_v, b_size
        y, x = position
        (y1, y2), (x1, x2) = quadrant_ranges[next_quadrant - 1]
        y2 -= 1
        x2 -= 1
        norm_x, norm_y = x % b_size, y % b_size
        o_v = code_to_v[opposite_v[c2]]
        if c1 == 'R':
            if c2 == 'R':
                return (y2 - norm_y, x2), o_v
            # if c2 == 'U':
            #     return (y1, x2 - norm_y), o_v
            if c2 == 'B':
                return (y2, x1 + norm_y), o_v
        ##
        if c1 == 'L':
            if c2 == 'U':
                return (y1, x1 + norm_y), o_v
            # if c2 == 'B':
            #     return (y2, x2 - norm_y), o_v
            if c2 == 'L':
                return (y2 - norm_y, x1), o_v
        ##
        if c1 == 'B':
            # if c2 == 'B':
            #     return (y2, x2 - norm_x), o_v
            if c2 == 'R':
                return (y1 + norm_x, x2), o_v
            # if c2 == 'L':
            #     return (y2 - norm_x, x1), o_v
            if c2 == 'U':
                return (y1, x1 + norm_x), o_v
        ##
        if c1 == 'U':
            # if c2 == 'U':
            #     return (y1, x2 - norm_x), o_v
            if c2 == 'L':
                return (y1 + norm_x, x1), o_v
            # if c2 == 'R':
            #     return (y2 - norm_x, x2), o_v
            if c2 == 'B':
                return (y2, x1 + norm_x), o_v
        print(f'No Transition for {c1} in {quadrant} to {c2}')

    # navigation = {
    #     1: {'R': (6, 'R'), 'L': (3, 'U'), 'U': (2, 'U')},
    #     2: {'L': (6, 'B'), 'U': (1, 'U'), 'B': (5, 'B')},
    #     3: {'U': (1, 'L'), 'B': (5, 'L')},
    #     4: {'R': (6, 'U')},
    #     5: {'L': (3, 'B'), 'B': (2, 'B')},
    #     6: {'R': (1, 'R'), 'U': (4, 'R'), 'B': (2, 'L')},
    # }

    navigation = {
        1: {'L': (4, 'L'), 'U': (6, 'L')},
        2: {'R': (5, 'R'), 'U': (6, 'B'), 'B': (3, 'R')},
        3: {'R': (2, 'B'), 'L': (4, 'U')},
        4: {'L': (1, 'L'), 'U': (3, 'L')},
        5: {'R': (2, 'R'), 'B': (6, 'R')},
        6: {'R': (5, 'B'), 'L': (1, 'U'), 'B': (2, 'U')},
    }

    for move in moves:
        if move == 'R' or move == 'L':
            dy, dx = rotate((dy, dx), move)
        else:
            move_count = int(move)
            for i in range(move_count):
                y, x = current_position
                y1, x1 = (y + dy, x + dx)
                ##
                (min_x, max_x), (min_y, max_y) = x_ranges[y], y_ranges[x]
                quadrant, v_code = quadrants[y, x], v_to_code[(dy, dx)]
                path[y, x] = v_code
                next_dy, next_dx = dy, dx
                if x1 > max_x or x1 < min_x or y1 > max_y or y1 < min_y:
                    quadrant_navigation = navigation[quadrant]
                    next_quadrant, next_side = quadrant_navigation[v_code]
                    (y1, x1), (next_dy, next_dx) = translate(v_code, current_position, next_quadrant, next_side)
                ##
                if grid[y1, x1] == 1:
                    # print(f'{y, x} @ Block')
                    break
                else:
                    if grid[y1, x1] == 0:
                        dy, dx = next_dy, next_dx
                        current_position = (y1, x1)

    y, x = current_position
    dy, dx = code_to_v[path[y, x]]
    direction_score = 0 if (dy, dx) == (0, 1) else 1 if (dy, dx) == (1, 0) else 2 if (dy, dx) == (0, -1) else 3 if (dy, dx) == (-1, 0) else 0
    score = (y + 1) * 1000 + (x + 1) * 4 + direction_score
    print(f'{y + 1}, {x + 1}, {dy, dx} :  {score}')
    return


# part_1()

part_2()
