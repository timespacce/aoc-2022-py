def part_1():
    s = open('data/input_21', 'r')
    rows = s.readlines()
    s.close()

    nodes = {}

    class Node:
        id: str
        number: int
        left: object
        right: object
        expression: str
        is_leaf: bool

        def __init__(self, data):
            content = data.rstrip()
            self.id, suffix = content.split(': ')
            contents = suffix.split(' ')
            if len(contents) == 1:
                self.number = int(suffix)
                self.is_leaf = True
            else:
                self.left, self.expression, self.right = contents
                self.is_leaf = False
            return

        def evaluate(self):
            if self.is_leaf:
                return int(self.number)
            else:
                left_value, right_value = self.left.evaluate(), self.right.evaluate()
                if self.expression == '+':
                    return left_value + right_value
                elif self.expression == '-':
                    return left_value - right_value
                elif self.expression == '*':
                    return left_value * right_value
                elif self.expression == '/':
                    return left_value / right_value
            return

    def transform(row):
        node = Node(row)
        nodes[node.id] = node
        return node

    rows = [transform(row) for row in rows]

    for row in rows:
        if not row.is_leaf:
            row.left = nodes[row.left]
            row.right = nodes[row.right]

    root_value = nodes['root'].evaluate()
    print(f"root_value={root_value}")
    return


def part_2():
    s = open('data/input_21', 'r')
    rows = s.readlines()
    s.close()

    nodes = {}

    class Node:
        id: str
        number: int
        left: object
        right: object
        expression: str
        is_leaf: bool

        def __init__(self, data):
            content = data.rstrip()
            self.id, suffix = content.split(': ')
            contents = suffix.split(' ')
            if len(contents) == 1:
                self.number = int(suffix)
                self.is_leaf = True
            else:
                self.left, self.expression, self.right = contents
                self.is_leaf = False
            return

        def evaluate(self):
            if self.is_leaf:
                return int(self.number)
            else:
                left_value, right_value = self.left.evaluate(), self.right.evaluate()
                if self.id == 'root':
                    diff = abs(left_value - right_value)
                    # print(f"{left_value} {right_value} {diff}")
                    return left_value == right_value, diff
                elif self.expression == '+':
                    return left_value + right_value
                elif self.expression == '-':
                    return left_value - right_value
                elif self.expression == '*':
                    return left_value * right_value
                elif self.expression == '/':
                    return left_value / right_value
            return

    def transform(row):
        node = Node(row)
        nodes[node.id] = node
        return node

    rows = [transform(row) for row in rows]

    for row in rows:
        if not row.is_leaf:
            row.left = nodes[row.left]
            row.right = nodes[row.right]
            continue

    left_range, middle, right_range = 0, 82225382988628 // 2, 82225382988628
    while True:
        nodes['humn'].number = left_range
        left_value, diff_left = nodes['root'].evaluate()

        nodes['humn'].number = middle
        middle_value, diff_middle = nodes['root'].evaluate()

        nodes['humn'].number = right_range
        right_value, diff_right = nodes['root'].evaluate()

        print(f"{diff_left:>30.2f} {diff_middle:>30.2f} {diff_right:>30.2f}")

        if diff_left == 0 or diff_middle == 0 or diff_right == 0:
            print(f'{left_range} {middle} {right_range}')
            break

        if diff_left <= diff_middle:
            left_range, middle, right_range = left_range, left_range + (middle - left_range) // 2, middle
        elif diff_right <= diff_middle:
            left_range, middle, right_range = middle, middle + (right_range - middle) // 2, right_range
        else:
            left_range, middle, right_range = left_range + (middle - left_range) // 2, middle, right_range - (left_range - middle) // 2

    return


# part_1()
part_2()
