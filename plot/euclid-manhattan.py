import math
import matplotlib.pyplot as plt
import numpy as np


def euclidean_dist(x1, y1, x2, y2):
    dX = x2 - x1
    dY = y2 - y1
    return math.sqrt(dX ** 2 + dY ** 2)


def manhattan_dist(x1, y1, x2, y2):
    dX = x2 - x1
    dY = y2 - y1
    return dX + dY


start = 0
ends = np.linspace(0, 100, 10)


euclid = list(map(lambda end: euclidean_dist(start, start, end, end), ends))
manhattan = list(map(lambda end: manhattan_dist(start, start, end, end), ends))

print(euclid)

plt.plot(ends, euclid, )
plt.plot(ends, manhattan)
plt.show()