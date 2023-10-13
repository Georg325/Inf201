# opp 5
# Georg Vang, Thomas Drageset

import numpy as np
from copy import deepcopy

n = [64, 128, 128, 128, 10]


def sigma(y):
    if y < 0:
        return 0
    else:
        return y


sigma_vec = np.vectorize(sigma)


def layer(W, b, x):
    return sigma_vec(W @ x + b)


x = np.random.normal(size=64)
y = deepcopy(x)

for id_n in range(len(n) - 1):
    W = np.random.normal(size=[n[id_n + 1], n[id_n]])
    print(f'W_{id_n + 1} shape: {W.shape}')
    b = np.random.normal(size=n[id_n + 1])
    y = layer(W, b, y)

print("Output:")
print(np.around(y, 2))
