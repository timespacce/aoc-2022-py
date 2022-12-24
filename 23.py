import numpy as np
import matplotlib.pyplot as plt


def load_data():
    s = open('data/input_23', 'r')
    grid = np.array([[1 if c == '#' else 0 for c in list(row.rstrip())] for row in s.readlines()])
    h, w = grid.shape
    print(f'{h}, {w}')
    s.close()

    y, x = np.nonzero(grid)
    elfs = [(ey + 1, ex + 1) for ey, ex in list(zip(y, x))]

    h, w = h + 2, w + 2
    grid = np.zeros((h, w))
    for y, x in elfs:
        grid[y, x] = 1
    return grid, elfs, h, w


grid, elfs, h, w = load_data()
rounds = 1500
us = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
vs = [
    [(-1, 0), (-1, 1), (-1, -1)],  # N NE NW
    [(1, 0), (1, 1), (1, -1)],  # S SE SW
    [(0, -1), (-1, -1), (1, -1)],  # W NW SW
    [(0, 1), (-1, 1), (1, 1)],  # E NE SE
]
vs_order = [0, 1, 2, 3]
for j in range(rounds):
    elf_to_next_coordinate, coordinate_to_visits, proposed_direction = {}, {}, None
    # 1. Get Coordinates
    for y, x in elfs:
        # Check if anything is around the elf.
        free = all([grid[y + dy, x + dx] == 0 for dy, dx in us])
        if free:
            continue
        # Check where to move
        next_y, next_x = None, None
        for i in vs_order:
            v = vs[i]
            direction_is_free = all([grid[y + dy, x + dx] == 0 for dy, dx in v])
            if direction_is_free:
                next_y, next_x = v[0]
                if proposed_direction is None:
                    proposed_direction = i
                break
        if next_y is None or next_x is None:
            # print('No Direction Is Free')
            continue
        next_coordinate = (y + next_y, x + next_x)
        elf_to_next_coordinate[(y, x)] = next_coordinate
        coordinate_to_visits[next_coordinate] = coordinate_to_visits.get(next_coordinate, 0) + 1
    # If no elf translates - this is the round
    if len(elf_to_next_coordinate) == 0:
        print(f'Round : {j}')
        break

    # 2. Translate To Coordinates
    next_elfs = []
    o = 1
    for y, x in elfs:
        if (y, x) in elf_to_next_coordinate:
            next_y, next_x = elf_to_next_coordinate[(y, x)]
            number_of_visits = coordinate_to_visits[(next_y, next_x)]
            if number_of_visits > 1:
                next_elfs.append((y + o, x + o))
            else:
                next_elfs.append((next_y + o, next_x + o))
        else:
            # Elf is free
            next_elfs.append((y + o, x + o))
    elfs = next_elfs
    h, w = h + 2, w + 2
    grid = np.zeros((h, w))
    min_x, max_x, min_y, max_y = 1e10, -1e10, 1e10, -1e10
    for y, x in elfs:
        grid[y, x] = 1
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
    sub_grid = grid[min_y:max_y + 1, min_x:max_x + 1]
    empty_tiles = np.where(sub_grid == 0)
    if j == 9:
        print(f'{len(empty_tiles[0])}')
    proposed_direction = vs_order[0]
    vs_order.pop(0)
    vs_order += [proposed_direction]
    print(f'{j} @ {h} {w} with {len(elf_to_next_coordinate)} : {len(coordinate_to_visits)}')
    continue
