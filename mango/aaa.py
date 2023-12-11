from random import randint


dimensions = 100

checks = {True: 0, False: 0}

for _ in range(100):
    p1 = [randint(-100, 100) for _ in range(dimensions)]
    p2 = [randint(-100, 100) for _ in range(dimensions)]
    p3 = [randint(-100, 100) for _ in range(dimensions)]

    while p2 == p1:
        p2 = [randint(-3, 3) for _ in range(dimensions)]

    while p3 == p1 or p3 == p2:
        p3 = [randint(-3, 3) for _ in range(dimensions)]

    pa = [p3[i] - p1[i] for i in range(dimensions)]
    ba = [p2[i] - p1[i] for i in range(dimensions)]

    t = sum([pa[i] * ba[i] for i in range(dimensions)]) / sum(
        [ba[i] * ba[i] for i in range(dimensions)]
    )

    d = [pa[i] - t * ba[i] for i in range(dimensions)]

    check = sum([d[i] * ba[i] for i in range(dimensions)])

    distance = sum([d[i] * d[i] for i in range(dimensions)]) ** 0.5

    checks[abs(check) <= 0.00000000001] += 1


print(checks)
