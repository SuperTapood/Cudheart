from util import np, check

a = np.array([0, 0, 0, 1, 3, 2, 0, 2, 1, 0])
res = np.unique(a, True, True, True)
comp = [[0, 1, 2, 3], [0, 3, 5, 4], [0, 0, 0, 1, 3, 2, 0, 2, 1, 0], [5, 2, 2, 1]]
res = res[3]
check(res, np.array(comp[3]))
