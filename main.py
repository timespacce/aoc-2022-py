import numpy as np
import re
import time


def run():
    task_1()
    task_2()
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


if __name__ == "__main__":
    begin = time.time()
    run()
    runtime = time.time() - begin
    print(f"AOC in {runtime} s")
