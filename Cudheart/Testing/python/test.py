import numpy as np

from util import *


test_name = "Statistics::histogram2d(Vector<int>, Vector<int>, Vector<int>, Vector<int>)"
a = np.array([1.0, 0.0, 1.0])
b = np.array([0.0, 1.0, 2.0, 3.0])
aa = np.array([2.0, 7, 3.0])
bb = np.array([3.0, 6.0, 9.0, 10.0])
res, a, b = np.histogram2d(a, aa, bins=(b, bb))
out = [[0.0, 0.0, 0.0], [4.0, 2.0, 0.0], [2.0, 1.0, 0.0]]

print(np.histogram(a, b))
print(np.histogram(aa, bb))

print(res)
print(a)
print(b)
