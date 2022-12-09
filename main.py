import string
from functools import reduce

import numpy as np
import re
import time
from collections import Counter
import matplotlib.pyplot as plt


def run():
    # task_1()
    # task_2()
    # task_3()
    # task_4()
    # task_5()
    # task_6()
    # task_7()
    # task_8()
    # task_9()
    # task_10()
    # task_11()
    # task_12()
    # task_13()
    # task_14()
    # task_15()
    # task_16()
    # task_17()
    task_18()
    return


def task_1():
    # READ
    s = open("data/input_1", "r")
    rows = s.readlines()
    s.close()

    # TRANSFORM
    def f(row):
        return row.rstrip()

    rows = [f(row) for row in rows]

    max_calories, calories = 0, 0
    for row in rows:
        if row == "":
            calories = 0
        else:
            calories += float(row)
            max_calories = max(max_calories, calories)
    print(f"MAX_CALORIES = {max_calories}")


def task_2():
    # READ
    s = open("data/input_1", "r")
    rows = s.readlines()
    s.close()

    # TRANSFORM
    def f(row):
        return row.rstrip()

    rows = [f(row) for row in rows]

    stack, calories = [], 0
    for row in rows:
        if row == "":
            stack.append(calories)
            calories = 0
        else:
            calories += float(row)
    stack = np.array(stack)
    sorted_stack = np.sort(stack)[::-1]
    top_3_elves = sorted_stack[:3].sum()
    print(f"TOP_3_ELVES = {top_3_elves}")


def task_3():
    s = open("data/input_2", "r")
    rows = s.readlines()
    s.close()

    mapping = {"A": 1, "B": 2, "C": 3, "X": 1, "Y": 2, "Z": 3}

    def transform(row):
        i, j = row.rstrip().split(' ')
        return i, j

    rows = [transform(row) for row in rows]

    def aggregate(acc, x, y):
        if mapping[y] == mapping[x]:
            return acc + mapping[y] + 3
        elif y == "Z" and x == "B":
            return acc + mapping[y] + 6
        elif y == "Y" and x == "A":
            return acc + mapping[y] + 6
        elif y == "X" and x == "C":
            return acc + mapping[y] + 6
        else:
            return acc + mapping[y]

    score = reduce(lambda acc, ij: aggregate(acc, ij[0], ij[1]), rows, 0)
    print(f"SCORE = {score}")
    return


def task_4():
    s = open("data/input_2", "r")
    rows = s.readlines()
    s.close()

    mapping = {"A": 1, "B": 2, "C": 3, "X": 1, "Y": 2, "Z": 3}

    def transform(row):
        i, j = row.rstrip().split(' ')
        return i, j

    rows = [transform(row) for row in rows]

    def aggregate(acc, x, y):
        if y == "Y":
            return acc + mapping[x] + 3
        elif y == "X":
            return acc + (3 if x == 'A' else 2 if x == 'C' else 1)
        else:
            return acc + 6 + (2 if x == 'A' else 3 if x == 'B' else 1)

    score = reduce(lambda acc, ij: aggregate(acc, ij[0], ij[1]), rows, 0)
    print(f"SCORE = {score}")
    return


def task_5():
    s = open("data/input_3", "r")
    rows = s.readlines()
    s.close()

    def transform(row):
        items = list(row.rstrip())
        items_len = len(items) // 2
        return set(row[:items_len]), set(row[items_len:])

    letters = list(string.ascii_letters)
    scores = dict(zip(letters, np.arange(1, len(letters) + 1)))

    rows = [transform(row) for row in rows]
    total_score = 0
    for a, b in rows:
        intersection = a.intersection(b)
        item = list(intersection)[0]
        total_score += scores[item]
    print(f"TOTAL_SCORE = {total_score}")
    return


def task_6():
    s = open("data/input_3", "r")
    rows = s.readlines()
    s.close()

    def transform(row):
        items = list(row.rstrip())
        return set(items)

    rows = [transform(row) for row in rows]

    letters = list(string.ascii_letters)
    scores = dict(zip(letters, np.arange(1, len(letters) + 1)))

    number_of_groups = len(rows) // 3
    total_score = 0
    for i in range(0, number_of_groups):
        group = rows[i * 3:i * 3 + 3]
        a, b, c = group[0], group[1], group[2]
        item = a.intersection(b).intersection(c)
        item = list(item)[0]
        total_score += scores[item]
    print(f"TOTAL_SCORE = {total_score}")
    return


def task_7():
    s = open("data/input_4", "r")
    rows = s.readlines()
    s.close()

    def transform(row):
        left, right = row.rstrip().split(",")
        (x1, y1), (x2, y2) = left.split("-"), right.split("-")
        return (int(x1), int(y1)), (int(x2), int(y2))

    rows = [transform(row) for row in rows]
    overlap_count = 0
    for (x1, y1), (x2, y2) in rows:
        if (x1 >= x2 and y1 <= y2) or (x2 >= x1 and y2 <= y1):
            overlap_count += 1

    print(f"OVERLAP_COUNT = {overlap_count}")
    return


def task_8():
    s = open("data/input_4", "r")
    rows = s.readlines()
    s.close()

    def transform(row):
        left, right = row.rstrip().split(",")
        (x1, y1), (x2, y2) = left.split("-"), right.split("-")
        return (int(x1), int(y1)), (int(x2), int(y2))

    rows = [transform(row) for row in rows]
    overlap_count = 0
    for (x1, y1), (x2, y2) in rows:
        if (x2 <= x1 <= y2) or (x1 <= x2 <= y1) or (x2 <= y1 <= y2) or (x1 <= y2 <= y1):
            overlap_count += 1

    print(f"OVERLAP_COUNT = {overlap_count}")
    return


def task_9():
    s = open("data/input_5", "r")
    platforms, instructions = s.read().split('\n\n')
    platforms = platforms.split('\n')
    platforms = platforms[::-1]
    platforms, states = platforms[0], platforms[1:]
    instructions = instructions.split("\n")
    s.close()

    def transform_platforms(row):
        platform_numbers = list(map(int, list(row.replace(' ', ''))))
        platform_dictionary = {i: [] for i in platform_numbers}
        return platform_dictionary

    platforms = transform_platforms(platforms)
    for state in states:
        i, j = 0, 1
        while j <= len(platforms):
            token = state[i:i + 3]
            if token != '   ':
                letter = token.replace('[', '').replace(']', '')
                platforms[j].append(letter)
            i += 4
            j += 1

    def transform_instruction(row):
        info = row.rstrip()
        count, source, target = list(map(int, list(re.findall('[0-9]+', info))))
        return count, source, target

    instructions = [transform_instruction(row) for row in instructions]

    for count, source, target in instructions:
        for i in range(0, count):
            element = platforms[source].pop()
            platforms[target].append(element)

    word = ''
    for key, value in platforms.items():
        word += value[-1]
    print(word)
    return


def task_10():
    s = open("data/input_5", "r")
    platforms, instructions = s.read().split('\n\n')
    platforms = platforms.split('\n')
    platforms = platforms[::-1]
    platforms, states = platforms[0], platforms[1:]
    instructions = instructions.split("\n")
    s.close()

    def transform_platforms(row):
        platform_numbers = list(map(int, list(row.replace(' ', ''))))
        platform_dictionary = {i: [] for i in platform_numbers}
        return platform_dictionary

    platforms = transform_platforms(platforms)
    for state in states:
        i, j = 0, 1
        while j <= len(platforms):
            token = state[i:i + 3]
            if token != '   ':
                letter = token.replace('[', '').replace(']', '')
                platforms[j].append(letter)
            i += 4
            j += 1

    def transform_instruction(row):
        info = row.rstrip()
        count, source, target = list(map(int, list(re.findall('[0-9]+', info))))
        return count, source, target

    instructions = [transform_instruction(row) for row in instructions]

    for count, source, target in instructions:
        stack = platforms[source][-count:]
        platforms[source] = platforms[source][:-count]
        platforms[target].extend(stack)

    word = ''
    for key, value in platforms.items():
        word += value[-1]
    print(word)
    return


def task_11():
    s = open("data/input_6", "r")
    sequence = s.readline()
    s.close()

    seq_len = len(sequence)
    message_len = 4
    for i in range(0, seq_len - message_len):
        sub = list(sequence[i:i + message_len])
        if len(np.unique(sub)) == message_len:
            print(f"{i + message_len}")
            break
    return


def task_12():
    s = open("data/input_6", "r")
    sequence = s.readline()
    s.close()

    seq_len = len(sequence)
    message_len = 14
    for i in range(0, seq_len - message_len):
        sub = list(sequence[i:i + message_len])
        if len(np.unique(sub)) == message_len:
            print(f"{i + message_len}")
            break
    return


def task_13():
    s = open("data/input_7", "r")
    rows = s.readlines()
    s.close()

    class Directory:
        name: str
        parent: object = None
        children: [object] = []
        files: [(str, int)] = []

        def __init__(self, name, parent):
            self.name = name
            self.parent = parent
            self.children = []
            self.files = []

    root_node = Directory("/", None)
    current_node = root_node

    for row in rows[1:]:
        c = row.rstrip()
        if c.startswith("$ cd "):
            directonary_name = c.replace("$ cd ", "")
            if directonary_name == "..":
                current_node = current_node.parent
            else:
                contains_child = False
                for child in current_node.children:
                    if child.name == directonary_name:
                        current_node = child
                        contains_child = True
                if not contains_child:
                    print("Error!")
        elif c.startswith("$ ls"):
            continue
        elif c.startswith("dir "):
            child_directory_name = c.replace("dir ", "")
            current_node.children.append(Directory(child_directory_name, current_node))
        else:
            (file_size, file_name) = c.split(" ")
            current_node.files.append((int(file_size), file_name))

    directory_sizes = []

    def calculate_size(directory: Directory) -> int:
        size_of_files = reduce(lambda x, y: x + y[0], directory.files, 0)
        directory_size = size_of_files + reduce(lambda x, y: x + calculate_size(y), directory.children, 0)
        directory_sizes.append(directory_size)
        return directory_size

    calculate_size(root_node)
    total_size = reduce(lambda x, y: x + (0 if y > 100000 else y), directory_sizes, 0)
    print(f"{total_size}")
    return


def task_14():
    s = open("data/input_7", "r")
    rows = s.readlines()
    s.close()

    class Directory:
        name: str
        parent: object = None
        children: [object] = []
        files: [(str, int)] = []

        def __init__(self, name, parent):
            self.name = name
            self.parent = parent
            self.children = []
            self.files = []

    root_node = Directory("/", None)
    current_node = root_node

    for row in rows[1:]:
        c = row.rstrip()
        if c.startswith("$ cd "):
            directonary_name = c.replace("$ cd ", "")
            if directonary_name == "..":
                current_node = current_node.parent
            else:
                contains_child = False
                for child in current_node.children:
                    if child.name == directonary_name:
                        current_node = child
                        contains_child = True
                if not contains_child:
                    print("Error!")
        elif c.startswith("$ ls"):
            continue
        elif c.startswith("dir "):
            child_directory_name = c.replace("dir ", "")
            current_node.children.append(Directory(child_directory_name, current_node))
        else:
            (file_size, file_name) = c.split(" ")
            current_node.files.append((int(file_size), file_name))

    directory_sizes = []

    def calculate_size(directory: Directory) -> int:
        size_of_files = reduce(lambda x, y: x + y[0], directory.files, 0)
        directory_size = size_of_files + reduce(lambda x, y: x + calculate_size(y), directory.children, 0)
        directory_sizes.append(directory_size)
        return directory_size

    calculate_size(root_node)
    directory_sizes = sorted(directory_sizes)
    outer_directory = directory_sizes[-1]
    unused_space = 70000000 - outer_directory
    for directory_size in directory_sizes:
        if unused_space + directory_size >= 30000000:
            print(directory_size)
            break
    return


def task_15():
    s = open("data/input_8", "r")
    rows = s.readlines()
    s.close()

    grid = np.array([list(map(int, list(row.rstrip()))) for row in rows])
    h, w = grid.shape

    count = 2 * h + 2 * w - 4
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            x = grid[i, j]
            left, right, top, bottom = grid[i, :j], grid[i, (j + 1):], grid[:i, j], grid[(i + 1):, j]
            left_check, right_check, top_check, bottom_check = x > left, x > right, x > top, x > bottom
            if left_check.all() or right_check.all() or top_check.all() or bottom_check.all():
                count += 1
    print(f"COUNT : {count}")
    return


def task_16():
    s = open("data/input_8", "r")
    rows = s.readlines()
    s.close()

    grid = np.array([list(map(int, list(row.rstrip()))) for row in rows])
    h, w = grid.shape

    max_scenic_score = 0
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            x = grid[i, j]
            left, right, top, bottom = grid[i, :j][::-1], grid[i, (j + 1):], grid[:i, j][::-1], grid[(i + 1):, j]
            scenic_score = 1
            for direction in [left, right, top, bottom]:
                count = 0
                for e in direction:
                    if e >= x:
                        count += 1
                        break
                    else:
                        count += 1
                scenic_score *= count
            max_scenic_score = max(max_scenic_score, scenic_score)
    print(f"COUNT : {max_scenic_score}")
    return


def task_17():
    s = open("data/input_9", "r")
    rows = s.readlines()
    s.close()

    def transform(row):
        u, c = row.strip().split(' ')
        return u, int(c)

    instructions = [transform(row) for row in rows]
    h, w = 2000, 2000
    grid = np.zeros((h, w))

    directions = {'L': np.array([0, -1]), 'R': np.array([0, 1]), 'U': np.array([-1, 0]), 'D': np.array([1, 0]), }

    h_xy, t_xy = np.array([h // 2, w // 2]), np.array([h // 2, w // 2])
    grid[h // 2, w // 2] = 1

    def distance(a, b):
        return b[0] - a[0], b[1] - a[1]

    for (u, c) in instructions:
        v = directions[u]
        for i in range(c):
            h_xy_previous = h_xy.copy()
            h_xy += v
            dx, dy = distance(t_xy, h_xy)
            if abs(dx) <= 1 and abs(dy) <= 1:
                continue
            else:
                t_xy = h_xy_previous
                grid[t_xy[0], t_xy[1]] += 1

    visited_positions = np.sum(grid >= 1)
    print(f"VISITED_POSITIONS is {visited_positions}")
    return


def task_18():
    s = open("data/input_9", "r")
    rows = s.readlines()
    s.close()

    directions = {'L': np.array([0, -1]), 'R': np.array([0, 1]), 'U': np.array([-1, 0]), 'D': np.array([1, 0]), }

    def transform(row):
        u, c = row.strip().split(' ')
        return directions[u], int(c)

    instructions = [transform(row) for row in rows]
    h, w = 50, 50
    grid = np.zeros((h, w))

    h_xy = np.array([h // 2, w // 2])
    grid[h // 2, w // 2] = 1

    tail = np.zeros((10, 2), dtype=np.int)
    tail[:, ] = h_xy

    for (v, c) in instructions:
        for i in range(c):
            tail[-1] += v
            for i in reversed(range(len(tail) - 1)):
                t_x, t_y = tail[i]
                h_x, h_y = tail[i + 1]
                if h_x - t_x > 1:
                    tail[i] += np.array([0, 1])
                    continue
                elif h_x - t_x < 1:
                    tail[i] += np.array([0, -1])
                    continue
                if h_y - t_y > 1:
                    tail[i] += np.array([1, 0])
                    continue
                elif h_y - t_y < 1:
                    tail[i] += np.array([-1, 0])
                    continue
        continue

    visited_positions = np.sum(grid >= 1)
    print(f"VISITED_POSITIONS is {visited_positions}")
    return


if __name__ == "__main__":
    begin = time.time()
    run()
    runtime = time.time() - begin
    print(f"AOC in {runtime} s")
