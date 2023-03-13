from util import *


test_name = "Statistics::cov(Matrix<int>, true)"
a = np.array([
 [0, 1, 2],
 [3, 4, 5],
 [6, 7, 8]
])
res = np.cov(a, None, True)
out = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

print(res)
