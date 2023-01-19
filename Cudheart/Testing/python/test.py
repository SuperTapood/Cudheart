from util import *

test_name = "IO::save<int>(string, Vector<StringType*>*, char)"
res = np.fromfile('savedArray.txt', sep='x', dtype=int)

out = [11, 21, 31, 41]
print(check(test_name, res, out))