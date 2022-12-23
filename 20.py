def mix(sequence, indices):
    seq_len = len(sequence)
    for i in range(seq_len):
        current_idx = indices.index(i)
        pivot = sequence[current_idx]
        current_j = current_idx
        if pivot > 0:
            moves = abs(pivot) % (seq_len - 1)
            for _ in range(moves):
                if current_j == seq_len - 1:
                    sequence = [sequence[0]] + sequence[:-1]
                    indices = [indices[0]] + indices[:-1]
                    current_j = 0
                sequence[current_j] = sequence[current_j + 1]
                indices[current_j] = indices[current_j + 1]
                current_j += 1
            if current_j == seq_len - 1:
                sequence = [sequence[0]] + sequence[:-1]
                indices = [indices[0]] + indices[:-1]
                current_j = 0
            sequence[current_j] = pivot
            indices[current_j] = i
        ##
        if pivot < 0:
            moves = abs(pivot) % (seq_len - 1)
            for _ in range(moves):
                if current_j == 0:
                    sequence = sequence[1:] + [sequence[-1]]
                    indices = indices[1:] + [indices[-1]]
                    current_j = seq_len - 1
                sequence[current_j] = sequence[current_j - 1]
                indices[current_j] = indices[current_j - 1]
                current_j -= 1
            if current_j == 0:
                sequence = sequence[1:] + [sequence[-1]]
                indices = indices[1:] + [indices[-1]]
                current_j = seq_len - 1
            sequence[current_j] = pivot
            indices[current_j] = i
        ##
        print(f"{(i / seq_len) * 100:.2f} %")
    return sequence, indices, seq_len


def part_1():
    s = open('data/input_20', 'r')
    sequence = [int(row.rstrip()) for row in s.readlines()]
    indices = list(range(len(sequence)))
    s.close()
    sequence, indices, seq_len = mix(sequence, indices)
    zero_index = sequence.index(0)
    a, b, c = sequence[(zero_index + 1000) % seq_len], sequence[(zero_index + 2000) % seq_len], sequence[(zero_index + 3000) % seq_len]
    print(f"{a} {b} {c} : {a + b + c}")


def part_2():
    s = open('data/input_20', 'r')
    sequence = [int(row.rstrip()) for row in s.readlines()]
    indices = list(range(len(sequence)))
    seq_len = len(sequence)
    s.close()
    key, mix_count = 811589153, 10
    sequence = [x * key for x in sequence]
    for _ in range(mix_count):
        sequence, indices, _ = mix(sequence, indices)
    print(sequence)
    zero_index = sequence.index(0)
    a, b, c = sequence[(zero_index + 1000) % seq_len], sequence[(zero_index + 2000) % seq_len], sequence[(zero_index + 3000) % seq_len]
    print(f"{a} {b} {c} : {a + b + c}")
    return


# part_1()
part_2()
