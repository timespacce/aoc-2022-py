import ast
import copy
import functools
import string
from enum import Enum
from functools import reduce, cmp_to_key

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
    # task_18()
    # task_19()
    # task_20()
    # task_21()
    # task_22()
    # task_23()
    # task_24()
    # task_25()
    # task_26()
    # task_27()
    # task_28()
    # task_29()
    # task_30()
    # task_31()
    # task_32()
    # task_33()
    # task_34()
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
    h, w = 1000, 1000
    grid = np.ones((h, w)) * -100

    h_xy = np.array([h // 2, w // 2])
    grid[h // 2, w // 2] = 1

    knots = np.zeros((10, 2), dtype=np.int)
    knots[:, ] = h_xy

    def distance(a, b):
        d_height, d_width = b[0] - a[0], b[1] - a[1]
        move = abs(d_height) >= 2 or abs(d_width) >= 2
        if d_height == 2:
            d_height -= 1
        if d_height == -2:
            d_height += 1
        if d_width == 2:
            d_width -= 1
        if d_width == -2:
            d_width += 1
        return d_height, d_width, move

    visited_positions = set()
    visited_positions.add((h // 2, w // 2))

    for (v, c) in instructions:
        for i in range(c):
            knots[-1] += v
            y, x = knots[-1]
            grid[y, x] = 9

            for i in reversed(range(len(knots) - 1)):
                # 0 - HEIGHT; 1 - WIDTH
                previous, current = knots[i], knots[i + 1]
                d_height, d_width, move = distance(previous, current)
                if move:
                    knots[i] += np.array([d_height, d_width])
                y, x = knots[i]
                grid[y, x] = i
                if i == 0:
                    visited_positions.add((y, x))
        a = 1

    print(f"VISITED_POSITIONS is {len(visited_positions)}")
    return


def task_19():
    s = open('data/input_10', 'r')
    rows = s.readlines()
    s.close()

    def transform(row):
        data = row.rstrip()
        if data.startswith('addx '):
            return 'addx', int(data.replace('addx', ''))
        else:
            return 'noop', 0

    instructions = [transform(row) for row in rows]
    flow = {}
    cycle = 0
    for instruction, x in instructions:
        if instruction == 'addx':
            cycle += 2
            flow[cycle] = x
        else:
            cycle += 1

    value = 1
    sum_of_strengths = 0
    checkpoints = [20, 60, 100, 140, 180, 220]
    for i in range(300):
        if i in checkpoints:
            strength = value * i
            sum_of_strengths += strength
            print(f"{i:>5} {value:>5} {strength:>5}")
        if i in flow:
            value += flow[i]
    print(f"STRENGTH is {sum_of_strengths}")
    return


def task_20():
    s = open('data/input_10', 'r')
    rows = s.readlines()
    s.close()

    def transform(row):
        data = row.rstrip()
        if data.startswith('addx '):
            return 'addx', int(data.replace('addx', ''))
        else:
            return 'noop', 0

    instructions = [transform(row) for row in rows]
    flow = {}
    cycle = 0
    for instruction, x in instructions:
        if instruction == 'addx':
            cycle += 2
            flow[cycle] = x
        else:
            cycle += 1

    value = 1
    sum_of_strengths = 0
    checkpoints = [20, 60, 100, 140, 180, 220]

    screen = np.ones([6, 40])

    sprite_position = 0

    for i in range(240):
        y, x = i // 40, i % 40

        if sprite_position <= x <= sprite_position + 2:
            screen[y, x] = 1
        else:
            screen[y, x] = 0

        if i in flow:
            value += flow[i]
            sprite_position = value

    for screen_row in screen:
        row = ' '.join(list(map(lambda x: '#' if x == 1 else '.', screen_row)))
        print(row)
    return


def task_21():
    s = open('data/input_11', 'r')
    data = s.read()
    s.close()

    class Monkey:
        id: int
        items: [int]
        expression: str
        divisible: int
        true_monkey: int
        false_monkey: int
        runs: int

        def __init__(self, data):
            rows = data.split('\n')
            self.items = list(map(int, rows[1].replace('  Starting items: ', '').split(', ')))
            self.expression = rows[2].replace('  Operation: new = ', '').split(' ')
            self.divisible = int(rows[3].replace('  Test: divisible by ', ''))
            self.true_monkey = int(rows[4].replace('    If true: throw to monkey ', ''))
            self.false_monkey = int(rows[5].replace('    If false: throw to monkey ', ''))
            self.runs = 0
            return

        def evaluate(self, value):
            self.runs += 1
            x, f, y = self.expression
            x = value if x == 'old' else int(x)
            y = value if y == 'old' else int(y)
            if f == '*':
                return x * y
            if f == '+':
                return x + y
            if f == '-':
                return x - y
            if f == '/':
                return x / y

    monkeys = data.split('\n\n')
    monkeys = [Monkey(data) for data in monkeys]

    for i in range(20):
        for monkey in monkeys:
            num_items = len(monkey.items)
            for j in range(num_items):
                item = monkey.items.pop(0)
                level = int(np.floor(monkey.evaluate(item) / 3))
                if level % monkey.divisible == 0:
                    monkeys[monkey.true_monkey].items.append(level)
                else:
                    monkeys[monkey.false_monkey].items.append(level)
        continue

    sorted_monkeys = sorted(monkeys, key=lambda x: x.runs, reverse=True)
    print(f"BUSINESS is {sorted_monkeys[0].runs * sorted_monkeys[1].runs}")
    return


def task_22():
    s = open('data/input_11', 'r')
    data = s.read()
    s.close()

    class Monkey:
        id: int
        items: [int]
        expression: str
        divisible: int
        true_monkey: int
        false_monkey: int
        runs: int

        def __init__(self, data):
            rows = data.split('\n')
            self.items = list(map(int, rows[1].replace('  Starting items: ', '').split(', ')))
            self.expression = rows[2].replace('  Operation: new = ', '').split(' ')
            self.divisible = int(rows[3].replace('  Test: divisible by ', ''))
            self.true_monkey = int(rows[4].replace('    If true: throw to monkey ', ''))
            self.false_monkey = int(rows[5].replace('    If false: throw to monkey ', ''))
            self.runs = 0
            return

        def evaluate(self, value):
            self.runs += 1
            x, f, y = self.expression
            x = value if x == 'old' else int(x)
            y = value if y == 'old' else int(y)
            if f == '*':
                return x * y
            if f == '+':
                return x + y
            if f == '-':
                return x - y
            if f == '/':
                return x / y

    monkeys = data.split('\n\n')
    monkeys = [Monkey(data) for data in monkeys]

    factor = 1
    while True:
        match = reduce(lambda acc, y: acc and factor % y.divisible == 0, monkeys, True)
        if match:
            break
        factor += 1

    for i in range(10000):
        for monkey in monkeys:
            num_items = len(monkey.items)
            for j in range(num_items):
                item = monkey.items.pop(0)
                level = int(monkey.evaluate(item)) % factor
                if level % monkey.divisible == 0:
                    monkeys[monkey.true_monkey].items.append(level)
                else:
                    monkeys[monkey.false_monkey].items.append(level)
        continue

    sorted_monkeys = sorted(monkeys, key=lambda x: x.runs, reverse=True)
    print(f"BUSINESS is {sorted_monkeys[0].runs * sorted_monkeys[1].runs}")
    return


def task_23():
    s = open("data/input_12", "r")
    rows = s.readlines()
    s.close()

    grid = np.array([list(map(ord, list(row.rstrip()))) for row in rows])
    h, w = grid.shape
    scores = np.ones((h, w)) * 1_000_000

    y1, x1 = (None, None)
    y2, x2 = (None, None)
    for y in range(h):
        for x in range(w):
            if grid[y, x] == ord('E'):
                y2, x2 = y, x
                grid[y, x] = ord('z')
            if grid[y, x] == ord('S'):
                y1, x1 = y, x
                grid[y, x] = ord('a')

    scores[y1, x1] = 0
    queue = [(np.array([y1, x1]), ord('a'))]

    directions = [np.array([0, 1]), np.array([0, -1]), np.array([-1, 0]), np.array([1, 0])]

    while len(queue) > 0:
        current_yx, current_elevation = queue.pop(0)
        current_score = scores[current_yx[0], current_yx[1]]
        for v in directions:
            y, x = current_yx + v
            if y < 0 or y >= h or x < 0 or x >= w:
                continue
            next_elevation = grid[y, x]
            if next_elevation > current_elevation + 1:
                continue
            if scores[y, x] <= current_score + 1:
                continue
            scores[y, x] = current_score + 1
            queue.append((np.array([y, x]), next_elevation))
            continue
    print(f"SCORE is {scores[y2, x2]}")
    return


def task_24():
    s = open("data/input_12", "r")
    rows = s.readlines()
    s.close()

    grid = np.array([list(map(ord, list(row.rstrip()))) for row in rows])
    h, w = grid.shape

    sources = []
    y2, x2 = (None, None)
    for y in range(h):
        for x in range(w):
            if grid[y, x] == ord('E'):
                y2, x2 = y, x
                grid[y, x] = ord('z')
            if grid[y, x] == ord('a') or grid[y, x] == ord('S') and (y == 0 or y == h - 1 or x == 0 or x == w - 1):
                sources.append(np.array([y, x]))

    directions = [np.array([0, 1]), np.array([0, -1]), np.array([-1, 0]), np.array([1, 0])]

    all_scores = []
    for y1, x1 in sources:
        scores = np.ones((h, w)) * 1_000_000
        scores[y1, x1] = 0
        queue = [(np.array([y1, x1]), ord('a'))]
        while len(queue) > 0:
            current_yx, current_elevation = queue.pop(0)
            current_score = scores[current_yx[0], current_yx[1]]
            for v in directions:
                y, x = current_yx + v
                if y < 0 or y >= h or x < 0 or x >= w:
                    continue
                next_elevation = grid[y, x]
                if next_elevation > current_elevation + 1:
                    continue
                if scores[y, x] <= current_score + 1:
                    continue
                scores[y, x] = current_score + 1
                queue.append((np.array([y, x]), next_elevation))
                continue
        print(f"SCORE from {y1},{x1} is {scores[y2, x2]}")
        all_scores.append(scores[y2, x2])
    all_scores = sorted(all_scores)
    print(f"BEST SCORE is {all_scores[0]}")
    return


def task_25():
    s = open('data/input_13', 'r')
    pairs = s.read().split('\n\n')
    s.close()
    pairs = [[ast.literal_eval(packet.rstrip()) for packet in pair.split('\n')] for pair in pairs]

    class OrderStatus(Enum):
        CONTINUE = 0,
        ORDERED = 1,
        NOT_ORDERED = 2,

    def check_pair(x, y) -> OrderStatus:
        if isinstance(x, list) and isinstance(y, list):
            iter_len = min(len(x), len(y))
            for i in range(iter_len):
                status = check_pair(x[i], y[i])
                if status == OrderStatus.ORDERED or status == OrderStatus.NOT_ORDERED:
                    return status
            if len(x) < len(y):
                return OrderStatus.ORDERED
            elif len(x) > len(y):
                return OrderStatus.NOT_ORDERED
            else:
                return OrderStatus.CONTINUE
        elif isinstance(x, int) and isinstance(y, int):
            if x == y:
                return OrderStatus.CONTINUE
            elif x < y:
                return OrderStatus.ORDERED
            else:
                return OrderStatus.NOT_ORDERED
        elif isinstance(x, int) and isinstance(y, list):
            return check_pair([x], y)
        elif isinstance(x, list) and isinstance(y, int):
            return check_pair(x, [y])
        else:
            print("Not supported pair")

    ordered_sum = 0
    for i, (x, y) in enumerate(pairs):
        order_status = check_pair(x, y)
        print(f"{i + 1} : {order_status}")
        ordered_sum += (i + 1) if order_status == OrderStatus.ORDERED else 0
    print(f"ORDERED_SUM = {ordered_sum}")
    return


def task_26():
    s = open('data/input_13', 'r')
    pairs = s.read().split('\n\n')
    s.close()
    pairs = [ast.literal_eval(packet.rstrip()) for pair in pairs for packet in pair.split('\n')]
    pairs.append([[2]])
    pairs.append([[6]])

    class OrderStatus(Enum):
        CONTINUE = 0,
        ORDERED = 1,
        NOT_ORDERED = 2,

    def check_pair(x, y) -> OrderStatus:
        if isinstance(x, list) and isinstance(y, list):
            iter_len = min(len(x), len(y))
            for i in range(iter_len):
                status = check_pair(x[i], y[i])
                if status == OrderStatus.ORDERED or status == OrderStatus.NOT_ORDERED:
                    return status
            if len(x) < len(y):
                return OrderStatus.ORDERED
            elif len(x) > len(y):
                return OrderStatus.NOT_ORDERED
            else:
                return OrderStatus.CONTINUE
        elif isinstance(x, int) and isinstance(y, int):
            if x == y:
                return OrderStatus.CONTINUE
            elif x < y:
                return OrderStatus.ORDERED
            else:
                return OrderStatus.NOT_ORDERED
        elif isinstance(x, int) and isinstance(y, list):
            return check_pair([x], y)
        elif isinstance(x, list) and isinstance(y, int):
            return check_pair(x, [y])
        else:
            print("Not supported pair")

    def compare(x, y) -> int:
        order_status = check_pair(x, y)
        return 1 if order_status == OrderStatus.ORDERED else -1

    sorted_pairs = sorted(pairs, key=cmp_to_key(lambda x, y: compare(y, x)))

    key = 1
    for i, pair in enumerate(sorted_pairs):
        if pair == [[2]] or pair == [[6]]:
            key *= (i + 1)
    print(f"DECODER_KEY = {key}")
    return


def task_27():
    s = open('data/input_14', 'r')
    rows = s.readlines()
    s.close()

    rows = [[[int(y) for y in c.split(',')] for c in row.rstrip().split(' -> ')] for row in rows]
    h, w = 1000, 1000
    grid = np.zeros((h, w))
    min_x, max_x, min_y, max_y = 1e10, -1e10, 1e10, -1e10

    def update_min_and_max(x, y):
        nonlocal grid, min_x, max_x, min_y, max_y
        min_x, max_x = min(min_x, x), max(max_x, x)
        min_y, max_y = min(min_y, y), max(max_y, y)
        grid[y, x] = 1

    for row in rows:
        for i in range(1, len(row)):
            c1, c2 = row[i - 1], row[i]
            dx, dy = c2[0] - c1[0], c2[1] - c1[1]
            if dx > 0:
                for x in range(abs(dx) + 1):
                    c1_x, c1_y = c1[0] + x, c1[1]
                    update_min_and_max(c1_x, c1_y)
            elif dx < 0:
                for x in range(abs(dx) + 1):
                    c1_x, c1_y = c1[0] - x, c1[1]
                    update_min_and_max(c1_x, c1_y)
            if dy > 0:
                for y in range(abs(dy) + 1):
                    c1_x, c1_y = c1[0], c1[1] + y
                    update_min_and_max(c1_x, c1_y)
            elif dy < 0:
                for y in range(abs(dy) + 1):
                    c1_x, c1_y = c1[0], c1[1] - y
                    update_min_and_max(c1_x, c1_y)
            continue
    print(f"MIN_X = {min_x} MAX_X = {max_x} MIN_Y = {min_y} MAX_Y = {max_y}")
    units_of_sands = 0
    while True:
        pivot = np.array([0, 500])
        ready = False
        while True:
            if pivot[1] < min_x or pivot[1] > max_x or pivot[0] > max_y:
                ready = True
                break

            (y1, x1), (y2, x2), (y3, x3) = pivot + np.array([1, 0]), pivot + np.array([1, -1]), pivot + np.array([1, 1])

            if grid[y1, x1] == 0:
                pivot = np.array([y1, x1])
            elif grid[y2, x2] == 0:
                pivot = np.array([y2, x2])
            elif grid[y3, x3] == 0:
                pivot = np.array([y3, x3])
            else:
                units_of_sands += 1
                grid[pivot[0], pivot[1]] = 2
                break
        if ready:
            break

    print(f"UNITS_OF_SAND = {units_of_sands}")
    return


def task_28():
    s = open('data/input_14', 'r')
    rows = s.readlines()
    s.close()

    rows = [[[int(y) for y in c.split(',')] for c in row.rstrip().split(' -> ')] for row in rows]
    h, w = 1000, 1000
    grid = np.zeros((h, w))
    min_x, max_x, min_y, max_y = 1e10, -1e10, 1e10, -1e10

    def update_min_and_max(x, y):
        nonlocal grid, min_x, max_x, min_y, max_y
        min_x, max_x = min(min_x, x), max(max_x, x)
        min_y, max_y = min(min_y, y), max(max_y, y)
        grid[y, x] = 1

    for row in rows:
        for i in range(1, len(row)):
            c1, c2 = row[i - 1], row[i]
            dx, dy = c2[0] - c1[0], c2[1] - c1[1]
            if dx > 0:
                for x in range(abs(dx) + 1):
                    c1_x, c1_y = c1[0] + x, c1[1]
                    update_min_and_max(c1_x, c1_y)
            elif dx < 0:
                for x in range(abs(dx) + 1):
                    c1_x, c1_y = c1[0] - x, c1[1]
                    update_min_and_max(c1_x, c1_y)
            if dy > 0:
                for y in range(abs(dy) + 1):
                    c1_x, c1_y = c1[0], c1[1] + y
                    update_min_and_max(c1_x, c1_y)
            elif dy < 0:
                for y in range(abs(dy) + 1):
                    c1_x, c1_y = c1[0], c1[1] - y
                    update_min_and_max(c1_x, c1_y)
            continue
    print(f"MIN_X = {min_x} MAX_X = {max_x} MIN_Y = {min_y} MAX_Y = {max_y}")
    grid[max_y + 2, :] = 1
    units_of_sands = 0
    while True:
        pivot = np.array([0, 500])
        ready = False
        while True:
            if grid[0, 500] == 2:
                ready = True
                break

            (y1, x1), (y2, x2), (y3, x3) = pivot + np.array([1, 0]), pivot + np.array([1, -1]), pivot + np.array([1, 1])

            if grid[y1, x1] == 0:
                pivot = np.array([y1, x1])
            elif grid[y2, x2] == 0:
                pivot = np.array([y2, x2])
            elif grid[y3, x3] == 0:
                pivot = np.array([y3, x3])
            else:
                units_of_sands += 1
                grid[pivot[0], pivot[1]] = 2
                break
        if ready:
            break

    print(f"UNITS_OF_SAND = {units_of_sands}")
    return


def task_29():
    s = open('data/input_15', 'r')
    rows = s.readlines()
    s.close()

    class SensorBeacon:
        x1: int
        y1: int
        x2: int
        y2: int
        manhattan_distance: int

        def __init__(self, row):
            data = row.rstrip().replace('Sensor at x=', '').replace(' y=', '').replace(' closest beacon is at x=', '')
            c1, c2 = data.split(":")
            self.x1, self.y1 = list(map(int, c1.split(',')))
            self.x2, self.y2 = list(map(int, c2.split(',')))
            self.manhattan_distance = abs(self.x2 - self.x1) + abs(self.y2 - self.y1)

    def transform(row):
        return SensorBeacon(row)

    rows = [transform(row) for row in rows]
    min_x, max_x = 1e10, -1e10
    for sb in rows:
        min_x = min(min_x, sb.x1)
        min_x = min(min_x, sb.x2)
        max_x = max(max_x, sb.x1)
        max_x = max(max_x, sb.x2)

    target_y = 2000000
    offset = 4
    print(f"POINTS = {len(rows)} with {min_x * offset} : {max_x * offset}")
    beacon_positions = set()
    for x in range(min_x * offset, max_x * offset):
        for sb in rows:
            manhattan_distance = abs(x - sb.x1) + abs(target_y - sb.y1)
            if x == sb.x2 and target_y == sb.y2:
                continue

            if manhattan_distance <= sb.manhattan_distance:
                beacon_positions.add((x, target_y))

    beacon_positions = sorted(beacon_positions)
    # for beacon_position in beacon_positions:
    #     print(beacon_position)

    print(f"NUMBER_OF_BEACON_POSITIONS = {len(beacon_positions)}")
    return


def task_30():
    s = open('data/input_15', 'r')
    rows = s.readlines()
    s.close()

    class SensorBeacon:
        x1: int
        y1: int
        x2: int
        y2: int
        manhattan_distance: int

        def __init__(self, row):
            data = row.rstrip().replace('Sensor at x=', '').replace(' y=', '').replace(' closest beacon is at x=', '')
            c1, c2 = data.split(":")
            self.x1, self.y1 = list(map(int, c1.split(',')))
            self.x2, self.y2 = list(map(int, c2.split(',')))
            self.manhattan_distance = abs(self.x2 - self.x1) + abs(self.y2 - self.y1)

        def in_range(self, y):
            if y < self.y1 - self.manhattan_distance or y > self.y1 + self.manhattan_distance:
                return False
            return True

        def x_range(self, y):
            radius = self.manhattan_distance - abs(self.y1 - y)
            return self.x1 - radius, self.x1 + radius

    def transform(row):
        return SensorBeacon(row)

    rows = [transform(row) for row in rows]
    offset = 4
    print(f"POINTS = {len(rows)}")
    beacon_positions = set()
    max_range = 4_000_000
    for y in range(max_range):
        x_ranges = [sb.x_range(y) for sb in rows if sb.in_range(y)]
        x_ranges_sorted = sorted(x_ranges, key=lambda x_range: x_range[0])
        max_x = -1e10
        for i in range(1, len(x_ranges_sorted)):
            (x1, x2), (y1, y2) = x_ranges_sorted[i - 1], x_ranges_sorted[i]
            max_x = max(max_x, x2)
            if max_x + 1 < y1:
                beacon_positions.add((y1 - 1, y))
        if y % 1000:
            print(f'{y / max_range * 100:.2f}%')

    beacon_positions = sorted(beacon_positions)
    for (x, y) in beacon_positions:
        print(f"{x},{y} : frequency = {x * 4_000_000 + y}")
    return


def task_31():
    s = open('data/input_16', 'r')
    rows = s.readlines()
    s.close()

    valves = {}

    class Valve:
        idx: int
        name: str
        flow_rate: int
        targets: [object]

        def __init__(self, idx, name, flow_rate, targets):
            self.idx = idx
            self.name = name
            self.flow_rate = flow_rate
            self.targets = targets

    def transform(idx, row):
        data = row.rstrip()
        content = data.replace('Valve ', '').replace(' has flow rate', '').replace(' tunnels lead to valves ', '').replace(' tunnel leads to valve ',
                                                                                                                           '')
        prefix, suffix = content.split(';')
        name, flow_rate = prefix.split('=')
        flow_rate = int(flow_rate)
        targets = list(suffix.split(', '))
        valve = Valve(idx, name, flow_rate, targets)
        valves[valve.name] = valve
        return valve

    rows = [transform(idx, row) for idx, row in enumerate(rows)]

    # @functools.lru_cache(maxsize=None)
    # def search(valve: Valve, open: bool, timeout: int, activated: str) -> int:
    #     if timeout <= 0:
    #         return 0
    #
    #     current_total_pressure = 0
    #     if open and valve.name not in activated:
    #         activated += valve.name
    #         timeout -= 1
    #         current_total_pressure = valve.flow_rate * timeout
    #
    #     max_flow = 0
    #     for target in valve.targets:
    #         valve_unit = valves[target]
    #         left = 0 if valve_unit.name in activated else search(valve_unit, True, timeout - 1, activated)
    #         right = search(valve_unit, False, timeout - 1, activated)
    #         max_flow = max(max_flow, max(left, right))
    #
    #     return current_total_pressure + max_flow

    queue = [(rows[0].name, 29 - 1, '', 0)]
    node_time_score = {(rows[0].name, 29 - 1): -1}
    max_flow = 0
    while len(queue) > 0:
        (valve, timeout, previous, flow) = queue.pop(0)
        if node_time_score.get((valve, timeout), -1) >= flow:
            continue
        node_time_score[(valve, timeout)] = flow
        if timeout <= 0:
            max_flow = max(max_flow, flow)
            continue
        for target in valves[valve].targets:
            valve_unit = valves[target]
            queue.append((valve_unit.name, timeout - 1, previous, flow))
            if valve_unit.name not in previous and valve_unit.flow_rate > 0:
                queue.append((valve_unit.name, timeout - 2, previous + valve_unit.name, flow + timeout * valve_unit.flow_rate))
    print(f"TOTAL PRESSURE IS {max_flow}")
    return


def task_32():
    s = open('data/input_16', 'r')
    rows = s.readlines()
    s.close()

    valves = {}

    class Valve:
        idx: int
        name: str
        flow_rate: int
        targets: [object]
        targets_ids: [int]

        def __init__(self, idx, name, flow_rate, targets):
            self.idx = idx
            self.name = name
            self.flow_rate = flow_rate
            self.targets = targets

    def transform(idx, row):
        data = row.rstrip()
        content = data.replace('Valve ', '').replace(' has flow rate', '').replace(' tunnels lead to valves ', '') \
            .replace(' tunnel leads to valve ', '')
        prefix, suffix = content.split(';')
        name, flow_rate = prefix.split('=')
        flow_rate = int(flow_rate)
        targets = list(suffix.split(', '))
        valve = Valve(idx, name, flow_rate, targets)
        valves[valve.name] = valve
        return valve

    rows = [transform(idx, row) for idx, row in enumerate(rows)]
    for row in rows:
        row.targets_ids = [(valves[target].idx, valves[target].flow_rate) for target in row.targets]
    required_valves = [row.idx for row in rows if row.flow_rate > 0]
    required_idx = reduce(lambda x, y: x | (1 << y), required_valves, 0)
    start, steps, max_flow, queue, checked_states = rows[0].idx, 26, 0, [], {}
    queue.append((start, start, steps, steps, 0, 0))

    def in_activated(idx: int, activated: int):
        return (1 << idx) & activated == (1 << idx)

    iter, min_timeout = 0, steps
    while queue:
        (a_idx, b_idx, timeout_a, timeout_b, activated, flow) = queue.pop(0)
        max_flow = max(max_flow, flow)

        iter += 1
        min_timeout = min(min_timeout, timeout_a)
        min_timeout = min(min_timeout, timeout_b)
        if iter % 10000 == 0:
            activated_valves = [rows[value].name for value in required_valves if in_activated(value, activated)]
            print(f"{iter} {max_flow} {min_timeout} {activated_valves} {len(queue)}")

        if required_idx & activated == required_idx:
            continue

        continue_flow = False
        for i in range(timeout_a, steps):
            for j in range(timeout_b, steps):
                if checked_states.get((a_idx, b_idx, i, j), -1) >= flow:
                    continue_flow = True
        if continue_flow:
            continue
        checked_states[(a_idx, b_idx, timeout_a, timeout_b)] = flow

        for a, flow_rate_a in rows[a_idx].targets_ids:
            for b, flow_rate_b in rows[b_idx].targets_ids:
                queue.append((a, b, timeout_a - 1, timeout_b - 1, activated, flow))
                if not in_activated(a, activated) and flow_rate_a > 0:
                    queue.append((a, b, timeout_a - 2, timeout_b - 1, activated | (1 << a), flow + (timeout_a - 2) * flow_rate_a))
                if not in_activated(b, activated) and flow_rate_b > 0:
                    queue.append((a, b, timeout_a - 1, timeout_b - 2, activated | (1 << b), flow + (timeout_b - 2) * flow_rate_b))
                if not in_activated(a, activated) and flow_rate_a > 0 and not in_activated(b, activated) and flow_rate_b > 0 and a != b:
                    queue.append((a, b, timeout_a - 2, timeout_b - 2, activated | (1 << a) | (1 << b),
                                  flow + (timeout_a - 2) * flow_rate_a + (timeout_b - 2) * flow_rate_b))

    print(f"TOTAL PRESSURE IS {max_flow}")
    return


def task_33():
    s = open('data/input_17', 'r')
    rows = s.read()
    s.close()

    transform = lambda x: np.array([0, -1]) if x == '<' else np.array([0, 1])
    moves = [transform(row) for row in list(rows.rstrip())]

    class Stone:
        position: np.ndarray

        def __init__(self, position):
            self.position = position

    stones = [
        # height - width
        Stone(np.array([[0, 0], [0, 1], [0, 2], [0, 3]])),
        Stone(np.array([[0, 1], [1, 0], [1, 1], [1, 2], [2, 1]])),
        Stone(np.array([[0, 2], [1, 2], [2, 0], [2, 1], [2, 2]])),
        Stone(np.array([[0, 0], [1, 0], [2, 0], [3, 0]])),
        Stone(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
    ]

    offsets = [0, 2, 2, 3, 1]

    height, width, max_stones = 5000 * 5, 7, 2022 * 5
    chamber = np.zeros((height, 7))
    current_stone, current_move, current_y = 0, 0, height - 1

    def correct_x_move(position: np.ndarray) -> bool:
        nonlocal width
        for y, x in position:
            if x < 0 or x >= width:
                return False
        return True

    def non_overlapping_move(position: np.ndarray) -> bool:
        nonlocal chamber
        for y, x in position:
            if y > height - 1 or chamber[y, x] == 1:
                return False
        return True

    cycles = {i: [] for i in range(5)}

    def mark_move(position: np.ndarray):
        nonlocal chamber, current_y, height, cycles
        previous_height = height - current_y - 1
        for y, x in position:
            chamber[y, x] = 1
            current_y = min(current_y, y - 1)
        ##
        y = height - current_y - 1
        # print(f"{current_stone_idx} : {y - previous_height}")
        height_diff = y - previous_height
        cycles[current_stone_idx].append(height_diff)

    while True:
        current_stone_idx = current_stone % len(stones)
        stone = copy.deepcopy(stones[current_stone_idx])
        stone.position += np.array([0, 2]) + np.array([current_y - offsets[current_stone_idx] - 3, 0])
        current_stone += 1
        while True:
            next_move = moves[current_move % len(moves)]
            current_move += 1
            next_position = stone.position + next_move
            if not correct_x_move(next_position):
                next_position = stone.position
            if not non_overlapping_move(next_position):
                next_position = stone.position
            stone.position = next_position

            next_position = stone.position + np.array([1, 0])
            if not non_overlapping_move(next_position):
                mark_move(stone.position)
                break
            stone.position = next_position

        if current_stone >= max_stones:
            break
        # print(f"stone={current_stone}")
    print(f"MAX_HEIGHT = {height - current_y - 1}")
    prefix_count, suffix_count, full_prefix_sum, full_cycle_sum = 0, 0, 0, 0
    sequences = {}
    for key, cycle in cycles.items():
        cycle_len = len(cycle)
        highest_width = 0
        sequence, offset, prefix_sum, suffix_sum = None, 0, 0, 0
        print(f"{cycle}")
        for width in range(1, cycle_len // 2):
            for i in range(width, cycle_len - width):
                a, b = cycle[i - width:i], cycle[i:i + width]
                if a == b:
                    highest_width = max(highest_width, width)
                    sequence = a
                    offset = i - width
                    prefix_sum, suffix_sum = sum(cycle[:offset]), sum(sequence)
        print(f"{offset} {highest_width} : {prefix_sum} {suffix_sum} {sequence}")
        prefix_count += offset
        suffix_count += highest_width
        full_prefix_sum += prefix_sum
        full_cycle_sum += suffix_sum
        sequences[key] = sequence

    all_cycles = np.int64(1000000000000)
    full_cycles_count = np.int64(all_cycles - prefix_count) // np.int64(suffix_count)
    print(
        f"prefix_count = {prefix_count} suffix_count = {suffix_count} full_cycles_count = {full_cycles_count} : prefix_sum = {full_prefix_sum} full_cycle_sum = {full_cycle_sum}")
    full_cycles_sum = np.int64(full_cycles_count) * np.int64(full_cycle_sum)
    print(f"full_cycles_sum = {full_cycles_sum}")
    offset_cycles_count = np.int64(all_cycles - (full_cycles_count * suffix_count + prefix_count))
    print(f"offset_cycles_count = {offset_cycles_count}")
    offset_cycles_sum = 0
    for i in range(offset_cycles_count):
        offset_cycles_sum += sequences[i % len(cycles)][i // len(cycles)]
    total_sum = full_prefix_sum + full_cycles_sum + offset_cycles_sum
    print(f"total_sum = {total_sum}")
    return


def task_34():
    prefix_heights = np.array([
        [1, 1, 1],
        [3, 3, 3],
        [2, 2, 2],
        [1, 2, 0],
        [2, 0, 2, ]
    ])
    cycle_heights = np.array([
        [1, 1, 1, 1, 0, 1, 1],
        [3, 2, 3, 2, 2, 2, 3],
        [3, 3, 2, 3, 1, 1, 2],
        [4, 0, 2, 4, 2, 2, 0],
        [0, 1, 0, 0, 0, 0, 0, ]
    ])

    prefix_heights = np.array([
        [1, 1, 1],
        [3, 3, 3],
        [2, 2, 2],
        [1, 2, 0],
        [2, 0, 2, ]
    ])
    cycle_heights = np.array([
        [1, 1, 1, 1, 0, 1, 1],
        [3, 2, 3, 2, 2, 2, 3],
        [3, 3, 2, 3, 1, 1, 2],
        [4, 0, 2, 4, 2, 2, 0],
        [0, 1, 0, 0, 0, 0, 0, ]
    ])

    all_stones = 1000000000000
    all_stones -= 15
    height = prefix_heights.sum()
    height += (all_stones / (5 * 7)) * cycle_heights.sum()
    print(height)
    return


if __name__ == "__main__":
    begin = time.time()
    run()
    runtime = time.time() - begin
    print(f"AOC in {runtime} s")
