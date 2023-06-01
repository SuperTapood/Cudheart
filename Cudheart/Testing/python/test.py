import numpy as np
from util import *


a = np.arange(25).reshape((5, 5))
b = np.arange(75).reshape((3, 5, 5))
c = np.repeat(a, repeats=3, axis=0)

# print(np.sum(b, axis=0))
# print(b[0, :, :])
# print(np.add.reduce(b, 0))

print(b)