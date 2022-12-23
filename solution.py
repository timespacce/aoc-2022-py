import itertools


def findFace(pos, w=50):
    x, y = getXY(pos)
    if x < w:
        if y < 3 * w:
            return 5
        return 6
    if x < 2 * w:
        if y < w:
            return 1
        if y < 2 * w:
            return 3
        return 4
    return 2


def getXY(pos):
    return int(pos.real), int(pos.imag)


def findNext(grid, facing, start):
    x, y = getXY(start + facing)
    y %= len(grid)
    x %= len(grid[y])
    while grid[y][x] == ' ':
        x, y = getXY(x + y * 1j + facing)
        y %= len(grid)
        x %= len(grid[y])
    if grid[y][x] == '.':
        return x + y * 1j
    return start


def findNextCube(grid, facing, pos, w=50):
    xOld, yOld = getXY(pos)
    xNew, yNew = getXY(pos + facing)
    xRel = xOld % w
    yRel = yOld % w
    curFace = findFace(pos)
    if xNew // w > xOld // w:  # exiting face to the right
        if curFace == 1 or curFace == 5:  # just go right
            return (xNew + yNew * 1j, facing)
        if curFace == 2:  # enter 4 from the right, upside down
            xNew = w * 2 - 1
            yNew = w * 3 - yRel - 1
            facing = -1
            return (xNew + yNew * 1j, facing)
        if curFace == 3:  # enter 2 from below
            xNew = 2 * w + yRel
            yNew = w - 1
            facing = -1j
            return (xNew + yNew * 1j, facing)
        if curFace == 4:  # enter 2 from the right, upside down
            xNew = w * 3 - 1
            yNew = w - yRel - 1
            facing = -1
            return (xNew + yNew * 1j, facing)
        if curFace == 6:  # enter 4 from below
            xNew = w + yRel
            yNew = w * 3 - 1
            facing = -1j
            return (xNew + yNew * 1j, facing)
    if xNew // w < xOld // w:  # exiting face to the left
        if curFace == 1:  # enter 5 from the right, upside down
            xNew = 0
            yNew = w * 3 - yRel - 1
            facing = 1
            return (xNew + yNew * 1j, facing)
        if curFace == 2 or curFace == 4:  # just go left
            return (xNew + yNew * 1j, facing)
        if curFace == 3:  # enter 5 from above
            xNew = yRel
            yNew = w * 2
            facing = 1j
            return (xNew + yNew * 1j, facing)
        if curFace == 5:  # enter 1 from the right, upside down
            xNew = w
            yNew = w - yRel - 1
            facing = 1
            return (xNew + yNew * 1j, facing)
        if curFace == 6:  # enter 1 from above
            xNew = w + yRel
            yNew = 0
            facing = 1j
            return (xNew + yNew * 1j, facing)
    if yNew // w > yOld // w:  # exiting face down (positive direction)
        if curFace == 1 or curFace == 3 or curFace == 5:  # just go down
            return (xNew + yNew * 1j, facing)
        if curFace == 2:  # enter 3 from the right
            xNew = w * 2 - 1
            yNew = w + xRel
            facing = -1
            return (xNew + yNew * 1j, facing)
        if curFace == 4:  # enter 5 from the right
            xNew = w - 1
            yNew = w * 3 + xRel
            facing = -1
            return (xNew + yNew * 1j, facing)
        if curFace == 6:  # enter 2 from above
            xNew = w * 2 + xRel
            yNew = 0
            return (xNew + yNew * 1j, facing)
    if yNew // w < yOld // w:  # exiting face up (negative direction)
        if curFace == 1:  # enter 6 from the left
            xNew = 0
            yNew = w * 3 + xRel
            facing = 1
            return (xNew + yNew * 1j, facing)
        if curFace == 2:  # enter 6 from below
            xNew = xRel
            yNew = w * 4 - 1
            return (xNew + yNew * 1j, facing)
        if curFace == 3 or curFace == 4 or curFace == 6:  # just go up
            return (xNew + yNew * 1j, facing)
        if curFace == 5:  # enter 3 from the left
            xNew = w
            yNew = w + xRel
            facing = 1
            return (xNew + yNew * 1j, facing)
    return (xNew + yNew * 1j, facing)


with open('data/input_22') as f:
    items = [l.strip('\n') for l in f.readlines()]

grid = items.copy()
way = grid.pop()
grid.pop()

w = max(len(l) for l in grid)
for i in range(len(grid)):
    grid[i] = grid[i].ljust(w)

visited = set()
facing = 1
if grid[0][0] != '.':
    pos = findNext(grid, facing, 0)
else:
    pos = 0

while way != '':
    if way[0].isnumeric():
        c = ''.join(itertools.takewhile(lambda x: x.isnumeric(), way))
        way = way[len(c):]
        for _ in range(int(c)):
            visited.add(pos)
            pos = findNext(grid, facing, pos)
    elif way[0] == 'R':
        facing *= 1j
        way = way[1:]
    elif way[0] == 'L':
        facing *= -1j
        way = way[1:]

x, y = getXY(pos)
facingScore = {1: 0, 1j: 1, -1: 2, -1j: 3}
score = 1000 * (y + 1) + 4 * (x + 1) + facingScore[facing]
print(score)

way = items[-1]
facing = 1
pos = findNext(grid, facing, 0)
visited.clear()
visited.add(pos)

while way != '':
    if way[0].isnumeric():
        c = ''.join(itertools.takewhile(lambda x: x.isnumeric(), way))
        way = way[len(c):]
        for _ in range(int(c)):
            newPos, newFacing = findNextCube(grid, facing, pos)
            x, y = getXY(newPos)
            if grid[y][x] == '.':
                pos = newPos
                facing = newFacing
                visited.add(pos)
    elif way[0] == 'R':
        facing *= 1j
        way = way[1:]
    elif way[0] == 'L':
        facing *= -1j
        way = way[1:]

x, y = getXY(pos)
score = 1000 * (y + 1) + 4 * (x + 1) + facingScore[facing]
print(score)