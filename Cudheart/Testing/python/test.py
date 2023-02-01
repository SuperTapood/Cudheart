from util import *

test_name = "Linalg::solve(Matrix<int>, Vector<int>)"
mat = np.array([
 [1, 2, 3, 4],
 [5, 6, 7, 8],
 [9, 10, 11, 12],
 [13, 14, 15, 16]
])
vec = np.array([1, 2, 3, 4])
res = np.linalg.solve(mat, vec)
out = [1, 0, 0, 0]


print(res)
print(out)
print(check(test_name, res, out))