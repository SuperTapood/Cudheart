from util import *

test_name = "Sorting::argsort(Vector<int>, Kind::Quicksort)"
vec = np.array([1, 25, 5820, 5, 58, 22, 48, 14, 32, 22, 99, 90900, 15, 11])
res = np.argsort(vec, kind='quicksort')

out = [0, 3, 13, 7, 12, 9, 5, 1, 8, 6, 4, 10, 2, 11]

print(np.sort(vec))
print(res)
print(out)
print(check(test_name, res, out))