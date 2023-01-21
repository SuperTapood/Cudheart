from util import *

test_name = "Searching::argwhere(Matrix<int>)"
vec = np.arange(0, 6, 1, dtype=int)
mat = vec.reshape((2, 3))
res = np.argwhere(mat)

out = [[0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
print(res)
print(out)
print(check(test_name, res, out))