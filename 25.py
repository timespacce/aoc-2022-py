import itertools

s = open('data/input_25', 'r')
rows = [row.rstrip() for row in s.readlines()]
s.close()

mapping = {
    '2': 2,
    '1': 1,
    '0': 0,
    '-': -1,
    '=': -2,
}


def to_decimal(row):
    return sum([mapping[c] * (5 ** i) for i, c in enumerate(reversed(list(row)))])


def to_snafu(decimal):
    snafu = ''
    while to_decimal(snafu) < decimal:
        snafu += '2'
    for i in range(len(snafu)):
        for c in ['=', '-', '0', '1']:
            next_snafu = snafu[:i] + c + snafu[i + 1:]
            if to_decimal(next_snafu) >= decimal:
                snafu = next_snafu
                break

    return snafu


numbers_sum = 0
for row in rows:
    snafu = to_decimal(row)
    numbers_sum += snafu
print(f"{numbers_sum}")

snafu_sum = to_snafu(numbers_sum)
print(f'{snafu_sum}')
