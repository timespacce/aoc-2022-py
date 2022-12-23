from functools import reduce


def load_data():
    s = open('data/input_19', 'r')
    blueprints = s.readlines()
    s.close()

    def transform(data):
        r = data.rstrip().split(':')[1].split('.')
        ore = r[0].rstrip().replace(' Each ore robot costs ', '').replace(' ore', '')
        clay = r[1].rstrip().replace(' Each clay robot costs ', '').replace(' ore', '')
        obsidian_1, obsidian_2 = r[2].rstrip().replace(' Each obsidian robot costs ', '').replace(' ore and ', ' ').replace(' clay', '').split(' ')
        geode_1, geode_2 = r[3].rstrip().replace(' Each geode robot costs ', '').replace(' ore and ', ' ').replace(' obsidian', '').split(' ')
        return [
            (int(ore), 0, 0),
            (int(clay), 0, 0),
            (int(obsidian_1), int(obsidian_2), 0),
            (int(geode_1), 0, int(geode_2))
        ]

    blueprints = [transform(blueprint) for blueprint in blueprints]

    return blueprints


blueprints = load_data()


def check_blueprint(minutes, blueprint):
    timeout, start_resources, start_robots, max_geode, max_geode_robots, iter = minutes, (0, 0, 0, 0), (1, 0, 0, 0), 0, 0, 0
    queue, visited = [(timeout, start_resources, start_robots)], set()

    max_ore_cost = max([ore for (ore, clay, obsidian) in blueprint])

    def next_robots(robots, update):
        return robots[0] + update[0], robots[1] + update[1], robots[2] + update[2], robots[3] + update[3]

    def next_resources(resources, update, cost):
        return resources[0] + update[0] - cost[0], resources[1] + update[1] - cost[1], resources[2] + update[2] - cost[2], resources[3] + update[3] - \
               cost[3]

    while queue:
        (timeout, r, robots) = queue.pop(0)
        max_geode = max(max_geode, r[3])
        max_geode_robots = max(max_geode_robots, robots[3])

        iter += 1
        if iter % 10000 == 0:
            print(f"{iter} {max_geode} {timeout} : {r} : {robots}")

        if timeout <= 0:
            continue

        if max_geode_robots > robots[3] + 1 or robots[3] * 1 + r[3] < max_geode:
            continue

        (ore_for_ore, _, _), (ore_for_clay, _, _), (ore_for_obsidian, clay_for_obsidian, _), (ore_for_geode, _, obsidian_for_geode) = blueprint
        r = (min(r[0], timeout * max_ore_cost - robots[0] * (timeout - 1)),
             min(r[1], timeout * clay_for_obsidian - robots[1] * (timeout - 1)),
             min(r[2], timeout * obsidian_for_geode - robots[2] * (timeout - 1)),
             r[3])

        if (timeout, r, robots) in visited:
            continue
        visited.add((timeout, r, robots))

        if ore_for_ore <= r[0] and robots[0] < max_ore_cost:
            state = (timeout - 1, next_resources(r, robots, (ore_for_ore, 0, 0, 0)), next_robots(robots, (1, 0, 0, 0)))
            queue.append(state)
        if ore_for_clay <= r[0] and robots[1] < clay_for_obsidian:
            state = (timeout - 1, next_resources(r, robots, (ore_for_clay, 0, 0, 0)), next_robots(robots, (0, 1, 0, 0)))
            queue.append(state)
        if ore_for_obsidian <= r[0] and clay_for_obsidian <= r[1] and robots[2] < obsidian_for_geode:
            state = (timeout - 1, next_resources(r, robots, (ore_for_obsidian, clay_for_obsidian, 0, 0)), next_robots(robots, (0, 0, 1, 0)))
            queue.append(state)
        if ore_for_geode <= r[0] and obsidian_for_geode <= r[2]:
            state = (timeout - 1, next_resources(r, robots, (ore_for_geode, 0, obsidian_for_geode, 0)), next_robots(robots, (0, 0, 0, 1)))
            queue.append(state)

        state = (timeout - 1, next_resources(r, robots, (0, 0, 0, 0)), robots)
        queue.append(state)

    return max_geode


def part1():
    minutes = 24
    quality_level_sum = 0
    for i, blueprint in enumerate(blueprints):
        blueprint_geode = check_blueprint(minutes, blueprint)
        quality_level = (i + 1) * blueprint_geode
        print(f'{i + 1} : {blueprint_geode} : {quality_level}')
        quality_level_sum += quality_level
    print(quality_level_sum)


def part2():
    minutes = 32
    geodes = []
    for i, blueprint in enumerate(blueprints[:3]):
        blueprint_geode = check_blueprint(minutes, blueprint)
        geodes.append(blueprint_geode)
        print(f'{i + 1} : {blueprint_geode}')
    print(reduce(lambda x, y: x * y, geodes, 1))


part2()
