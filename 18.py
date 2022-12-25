import numpy as np

s = open('data/input_18', 'r')
rows = s.readlines()
s.close()

rows = [np.array(list(map(int, row.rstrip().split(',')))) for row in rows]
directions = [np.array([-1, 0, 0]), np.array([1, 0, 0]), np.array([0, -1, 0]), np.array([0, 1, 0]), np.array([0, 0, -1]), np.array([0, 0, 1]), ]
total_surface = 0
for c1 in rows:
    local_surface = 0
    for u in directions:
        c1_delta = c1 + u
        covered = False
        for c2 in rows:
            covered |= all(c1_delta == c2)
        if not covered:
            local_surface += 1
    total_surface += local_surface
print(f"TOTAL_SURFACE = {total_surface}")

s = open('data/input_18', 'r')
rows = s.readlines()
rows = [tuple(map(int, row.rstrip().split(','))) for row in rows]
s.close()

directions = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
total_surface, free_cubes = 0, set()
for x, y, z in rows:
    for dx, dy, dz in directions:
        if (x + dx, y + dy, z + dz) not in rows:
            free_cubes.add((x + dx, y + dy, z + dz))
            total_surface += 1
print(f"TOTAL_SURFACE = {total_surface}")

min_r, max_r = 1e10, -1e10
for x, y, z in rows:
    min_r = min(min_r, x)
    min_r = min(min_r, y)
    min_r = min(min_r, z)

    max_r = max(max_r, x)
    max_r = max(max_r, y)
    max_r = max(max_r, z)

min_r -= 1
max_r += 1

start = (min_r, min_r, min_r)
queue, bounding_box = [], set()
queue.append(start)
bounding_box.add(start)
while queue:
    x, y, z = queue.pop(0)
    for dx, dy, dz in directions:
        o_x, o_y, o_z = (x + dx, y + dy, z + dz)
        if o_x < min_r or o_x > max_r or o_y < min_r or o_y > max_r or o_z < min_r or o_z > max_r:
            continue
        o = (o_x, o_y, o_z)
        if o in bounding_box:
            continue
        if o in rows:
            continue
        queue.append(o)
        bounding_box.add(o)

total_surface = 0
for x, y, z in rows:
    for dx, dy, dz in directions:
        o_x, o_y, o_z = (x + dx, y + dy, z + dz)
        if o_x < min_r or o_x > max_r or o_y < min_r or o_y > max_r or o_z < min_r or o_z > max_r:
            continue
        o = (o_x, o_y, o_z)
        if o in bounding_box:
            total_surface += 1

print(f"TOTAL_SURFACE = {total_surface}")
