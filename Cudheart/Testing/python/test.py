from util import *

test_name = "MatrixOps::vander(Vector<int>, int, bool)"
vec = np.arange(0, 6, 1, dtype=int)
res = np.vander(vec, 4, True)

out = [[1, 1, 1, 1], [1, 1, 5, 4], [3, 2, 1, 0], [25, 16, 9, 4], [1, 0, 125, 64], [27, 8, 1, 0]]
print(vec)
print(res)
print(out)
print(check(test_name, res, out))