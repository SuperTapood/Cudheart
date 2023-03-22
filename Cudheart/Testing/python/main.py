# this file is generated automatically to simplify unit testing :)
from util import *

test_name = "Vector<int>(int[], int)"
res = np.array([5, 7, 451, 14, 25, 250, 52205, 255, 897])
out = [5, 7, 451, 14, 25, 250, 52205, 255, 897]
add2queue(test_name, res, out)


test_name = "Vector<int>(int{})"
res = np.array([5, 7, 451, 14, 25, 250, 52205, 255, 897])
out = [5, 7, 451, 14, 25, 250, 52205, 255, 897]
add2queue(test_name, res, out)


test_name = "Vector<int>->reshape((2, 3))"
a = np.array([5, 7, 451, 14, 25, 250])
res = a.reshape((2, 3))
out = [[5, 7, 451], [14, 25, 250]]
add2queue(test_name, res, out)


test_name = "Matrix(int, int)"
vec = np.array([25, 25, 45, 588, 655555, 55, 568, 58])
res = vec.reshape((4, 2))
out = [[25, 25], [45, 588], [655555, 55], [568, 58]]
add2queue(test_name, res, out)


test_name = "Matrix(int{}, int)"
vec = np.array([25, 25, 45, 588, 655555, 55, 568, 58, 999])
res = vec.reshape((3, 3))
out = [[25, 25, 45], [588, 655555, 55], [568, 58, 999]]
add2queue(test_name, res, out)


test_name = "Matrix<int>->reshape((6))"
res = np.array([5, 7, 451, 14, 25, 250])
out = [5, 7, 451, 14, 25, 250]
add2queue(test_name, res, out)


test_name = "Matrix<int>->reverseRows()"
mat = np.array([5, 7, 451, 14, 25, 250, 52205, 255, 897])
mat = mat.reshape((3, 3))
res = np.flip(mat, 1)
out = [[451, 7, 5], [250, 25, 14], [897, 255, 52205]]
add2queue(test_name, res, out)


test_name = "Matrix<int>->reverseCols()"
mat = np.array([5, 7, 451, 14, 25, 250, 52205, 255, 897])
mat = mat.reshape((3, 3))
res = np.flip(mat, 0)
out = [[52205, 255, 897], [14, 25, 250], [5, 7, 451]]
add2queue(test_name, res, out)


test_name = "Matrix<int>->transpose()"
mat = np.array([5, 7, 451, 14, 25, 250, 52205, 255, 897])
mat = mat.reshape((3, 3))
res = mat.T
out = [[5, 14, 52205], [7, 25, 255], [451, 250, 897]]
add2queue(test_name, res, out)


test_name = "Matrix<int>->rot90(k=1)"
mat = np.array([5, 7, 451, 14, 25, 250, 52205, 255, 897])
mat = mat.reshape((3, 3))
res = np.rot90(mat, k = 1)
out = [[451, 250, 897], [7, 25, 255], [5, 14, 52205]]
add2queue(test_name, res, out)


test_name = "Matrix<int>->rot90(k=2)"
mat = np.array([5, 7, 451, 14, 25, 250, 52205, 255, 897])
mat = mat.reshape((3, 3))
res = np.rot90(mat, k = 2)
out = [[897, 255, 52205], [250, 25, 14], [451, 7, 5]]
add2queue(test_name, res, out)


test_name = "Matrix<int>->rot90(k=3)"
mat = np.array([5, 7, 451, 14, 25, 250, 52205, 255, 897])
mat = mat.reshape((3, 3))
res = np.rot90(mat, k = 3)
out = [[52205, 14, 5], [255, 25, 7], [897, 250, 451]]
add2queue(test_name, res, out)


test_name = "Matrix<int>->rot90(k=4)"
mat = np.array([5, 7, 451, 14, 25, 250, 52205, 255, 897])
mat = mat.reshape((3, 3))
res = np.rot90(mat, k = 4)
out = [[5, 7, 451], [14, 25, 250], [52205, 255, 897]]
add2queue(test_name, res, out)


test_name = "Matrix<int>->augment(Vector<int>)"
vec = np.array([5, 7, 451, 14, 25, 250, 52205, 255, 897])
mat = np.array([
 [5, 5],
 [7, 7],
 [451, 451],
 [14, 14],
 [25, 25],
 [250, 250],
 [52205, 52205],
 [255, 255],
 [897, 897]
])
mat2 = np.array([5, 7, 451, 14, 25, 250, 52205, 255, 897])
mat2 = mat2.reshape((9, 1))
res = np.concatenate((mat, mat2), axis=1)
out = [[5, 5, 5], [7, 7, 7], [451, 451, 451], [14, 14, 14], [25, 25, 25], [250, 250, 250], [52205, 52205, 52205], [255, 255, 255], [897, 897, 897]]
add2queue(test_name, res, out)


test_name = "IO::fromString<int>(string, char, int)"
res = np.array([1, 2, 3, 4])
out = [1, 2, 3, 4]
add2queue(test_name, res, out)


test_name = "IO::fromString<int>(string)"
res = np.array([1, 2, 3, 4])
out = [1, 2, 3, 4]
add2queue(test_name, res, out)


test_name = "IO::fromString<int>(string, int)"
res = np.array([1, 2, 3])
out = [1, 2, 3]
add2queue(test_name, res, out)


test_name = "IO::fromFile<int>(string, char, int)"
res = np.fromfile('file.txt', sep=' ', dtype=int)
res = res[0:3]
out = [11, 21, 31]
add2queue(test_name, res, out)


test_name = "IO::fromFile<int>(string, char)"
res = np.fromfile('file.txt', sep=' ', dtype=int)
out = [11, 21, 31, 41]
add2queue(test_name, res, out)


test_name = "IO::fromFile<int>(string, int)"
res = np.fromfile('file.txt', sep=' ', dtype=int)
res = res[0:3]
out = [11, 21, 31]
add2queue(test_name, res, out)


test_name = "IO::fromFile<int>(string)"
res = np.fromfile('file.txt', sep=' ', dtype=int)
out = [11, 21, 31, 41]
add2queue(test_name, res, out)


test_name = "IO::fromFunction(int func(int), int)"
func = lambda x: 10 * x
res = np.fromfunction(func, (17,), dtype=int)
out = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
add2queue(test_name, res, out)


test_name = "VectorOps::emptyLike<int>"
res = np.empty((5,))
res = res.shape
out = (5,)
add2queue(test_name, res, out)


test_name = "VectorOps::arange<int>(int)"
res = np.arange(0, 5, 1, dtype=int)
out = [0, 1, 2, 3, 4]
add2queue(test_name, res, out)


test_name = "VectorOps::arange<int>(int, int)"
res = np.arange(0, 5, 2, dtype=int)
out = [0, 2, 4]
add2queue(test_name, res, out)


test_name = "VectorOps::arange<int>(int, int, int)"
res = np.arange(3, 7, 1, dtype=int)
out = [3, 4, 5, 6]
add2queue(test_name, res, out)


test_name = "VectorOps::full<int>(int, int)"
res = np.full((5,), 5)
out = [5, 5, 5, 5, 5]
add2queue(test_name, res, out)


test_name = "VectorOps::fullLike(int, int)"
res = np.full((5,), 5)
out = [5, 5, 5, 5, 5]
add2queue(test_name, res, out)


test_name = "VectorOps::linspace(float, float)"
res = np.linspace(5, 10, 50, True, dtype=float)
out = [5.0, 5.1020408163265305, 5.204081632653061, 5.3061224489795915, 5.408163265306122, 5.5102040816326534, 5.6122448979591839, 5.7142857142857144, 5.8163265306122449, 5.9183673469387754, 6.0204081632653059, 6.1224489795918373, 6.2244897959183678, 6.3265306122448983, 6.4285714285714288, 6.5306122448979593, 6.6326530612244898, 6.7346938775510203, 6.8367346938775508, 6.9387755102040813, 7.0408163265306118, 7.1428571428571423, 7.2448979591836737, 7.3469387755102042, 7.4489795918367347, 7.5510204081632653, 7.6530612244897958, 7.7551020408163271, 7.8571428571428577, 7.9591836734693882, 8.0612244897959187, 8.1632653061224492, 8.2653061224489797, 8.3673469387755102, 8.4693877551020407, 8.5714285714285712, 8.6734693877551017, 8.7755102040816322, 8.8775510204081627, 8.979591836734695, 9.0816326530612237, 9.183673469387756, 9.2857142857142847, 9.387755102040817, 9.4897959183673475, 9.591836734693878, 9.6938775510204085, 9.795918367346939, 9.8979591836734695, 10.0]
add2queue(test_name, res, out)


test_name = "VectorOps::linspace(float, float, float)"
res = np.linspace(7, 12, 2, True, dtype=float)
out = [7.0, 12.0]
add2queue(test_name, res, out)


test_name = "VectorOps::linspace(float, float, float, bool)"
res = np.linspace(5, 10, 10, False, dtype=float)
out = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5]
add2queue(test_name, res, out)


test_name = "VectorOps::ones<int>"
res = np.full((5,), 1)
out = [1, 1, 1, 1, 1]
add2queue(test_name, res, out)


test_name = "VectorOps::onesLike<int>"
res = np.full((5,), 1)
out = [1, 1, 1, 1, 1]
add2queue(test_name, res, out)


test_name = "VectorOps::zeros<int>"
res = np.full((5,), 0)
out = [0, 0, 0, 0, 0]
add2queue(test_name, res, out)


test_name = "VectorOps::zerosLike<int>"
res = np.full((5,), 0)
out = [0, 0, 0, 0, 0]
add2queue(test_name, res, out)


test_name = "VectorOps::logspace(float, float)"
res = np.logspace(5, 10, 50, True, dtype=float, base=10)
out = [100000.0, 126485.5078125, 159985.84375, 202358.890625, 255954.671875, 323745.9375, 409491.6875, 517947.625, 655128.6875, 828642.875, 1048113.125, 1325711.125, 1676832.5, 2120950.25, 2682697.5, 3393223.5, 4291936.0, 5428677.0, 6866489.5, 8685114.0, 10985411.0, 13894968.0, 17575102.0, 22229980.0, 28117674.0, 35564820.0, 44984344.0, 56898676.0, 71968576.0, 91029824.0, 115139656.0, 145634816.0, 184207152.0, 232995088.0, 294705344.0, 372759136.0, 471486816.0, 596363136.0, 754312128.0, 954096576.0, 1206792576.0, 1526419328.0, 1930701312.0, 2442054656, 3088841984, 3906941696, 4941720576, 6250553344, 7906035200, 10000000000]
add2queue(test_name, res, out)


test_name = "VectorOps::logspace(float, float, int)"
res = np.logspace(7, 12, 2, True, dtype=float, base=10)
out = [10000000.0, 999999995904]
add2queue(test_name, res, out)


test_name = "VectorOps::logspace(float, float, int, bool)"
res = np.logspace(5, 10, 10, False, dtype=float, base=10)
out = [100000.0, 316227.78125, 1000000.0, 3162277.75, 10000000.0, 31622776.0, 100000000.0, 316227776.0, 1000000000.0, 3162277632]
add2queue(test_name, res, out)


test_name = "VectorOps::logspace(float, float, int, bool, float)"
res = np.logspace(5, 10, 10, False, dtype=float, base=2)
out = [32.0, 45.254833221435547, 64.0, 90.509666442871094, 128.0, 181.01933288574219, 256.0, 362.03866577148438, 512.0, 724.07733154296875]
add2queue(test_name, res, out)


test_name = "VectorOps::geomspace(float, float)"
res = np.geomspace(5, 10, 50, True, dtype=float)
out = [5.0, 5.0712318420410156, 5.1434793472290039, 5.2167549133300781, 5.2910747528076172, 5.3664536476135254, 5.4429059028625488, 5.5204477310180664, 5.5990939140319824, 5.678861141204834, 5.7597641944885254, 5.8418197631835938, 5.9250450134277344, 6.0094552040100098, 6.0950679779052734, 6.1819014549255371, 6.2699708938598633, 6.3592948913574219, 6.4498920440673828, 6.5417799949645996, 6.6349763870239258, 6.7295017242431641, 6.8253722190856934, 6.9226088523864746, 7.0212306976318359, 7.1212577819824219, 7.2227106094360352, 7.3256077766418457, 7.4299721717834473, 7.5358219146728516, 7.6431798934936523, 7.7520670890808105, 7.8625068664550781, 7.9745187759399414, 8.0881261825561523, 8.2033538818359375, 8.3202219009399414, 8.4387540817260742, 8.5589761734008789, 8.6809110641479492, 8.8045825958251953, 8.9300165176391602, 9.0572366714477539, 9.1862697601318359, 9.3171405792236328, 9.4498748779296875, 9.584503173828125, 9.7210483551025391, 9.8595380783081055, 10.0]
add2queue(test_name, res, out)


test_name = "VectorOps::geomspace(float, float, int)"
res = np.geomspace(7, 12, 2, True, dtype=float)
out = [6.9999995231628418, 11.999999046325684]
add2queue(test_name, res, out)


test_name = "VectorOps::geomspace(float, float, int, bool)"
res = np.geomspace(5, 10, 10, False, dtype=float)
out = [5.0, 5.3588676452636719, 5.7434921264648438, 6.1557216644287109, 6.5975399017333984, 7.071068286895752, 7.578582763671875, 8.122523307800293, 8.70550537109375, 9.3303308486938477]
add2queue(test_name, res, out)


test_name = "MatrixOps::emptyLike<int>"
res = [5, 5]
out = (5, 5)
add2queue(test_name, res, out)


test_name = "MatrixOps::arange<int>(int)"
res = np.arange(0, 6, 1, dtype=int)
res = res.reshape((2, 3))
out = [[0, 1, 2], [3, 4, 5]]
add2queue(test_name, res, out)


test_name = "MatrixOps::arange<int>(int, int)"
res = np.arange(0, 5, 2, dtype=int)
res = res.reshape((3, 1))
out = [[0], [2], [4]]
add2queue(test_name, res, out)


test_name = "MatrixOps::arange<int>(int, int)"
res = np.arange(3, 7, 1, dtype=int)
res = res.reshape((2, 2))
out = [[3, 4], [5, 6]]
add2queue(test_name, res, out)


test_name = "MatrixOps::full<int>(int, int)"
res = np.full((5, 5), 5)
out = [[5, 5, 5, 5, 5], [5, 5, 5, 5, 5], [5, 5, 5, 5, 5], [5, 5, 5, 5, 5], [5, 5, 5, 5, 5]]
add2queue(test_name, res, out)


test_name = "MatrixOps::fullLike(int, int)"
res = np.full((5, 5), 5)
out = [[5, 5, 5, 5, 5], [5, 5, 5, 5, 5], [5, 5, 5, 5, 5], [5, 5, 5, 5, 5], [5, 5, 5, 5, 5]]
add2queue(test_name, res, out)


test_name = "MatrixOps::linspace(float, float)"
res = np.linspace(5, 10, 50, True, dtype=float)
res = res.reshape((5, 10))
out = [[5.0, 5.1020407676696777, 5.2040815353393555, 5.3061223030090332, 5.4081630706787109, 5.5102043151855469, 5.6122450828552246, 5.7142858505249023, 5.8163266181945801, 5.9183673858642578], [6.0204081535339355, 6.1224489212036133, 6.224489688873291, 6.3265304565429688, 6.4285717010498047, 6.5306124687194824, 6.6326532363891602, 6.7346940040588379, 6.8367347717285156, 6.9387755393981934], [7.0408163070678711, 7.142857551574707, 7.2448978424072266, 7.3469390869140625, 7.448979377746582, 7.551020622253418, 7.6530613899230957, 7.7551021575927734, 7.8571429252624512, 7.9591836929321289], [8.0612249374389648, 8.1632652282714844, 8.2653064727783203, 8.3673467636108398, 8.4693880081176758, 8.5714282989501953, 8.6734695434570312, 8.7755107879638672, 8.8775510787963867, 8.9795923233032227], [9.0816326141357422, 9.1836738586425781, 9.2857151031494141, 9.3877553939819336, 9.4897956848144531, 9.5918369293212891, 9.693878173828125, 9.7959184646606445, 9.8979587554931641, 10.0]]
add2queue(test_name, res, out)


test_name = "MatrixOps::linspace(float, float, float)"
res = np.linspace(7, 12, 2, True, dtype=float)
res = res.reshape((2, 1))
out = [[7.0], [12.0]]
add2queue(test_name, res, out)


test_name = "MatrixOps::linspace(float, float, float, bool)"
res = np.linspace(5, 10, 10, False, dtype=float)
res = res.reshape((5, 2))
out = [[5.0, 5.5], [6.0, 6.5], [7.0, 7.5], [8.0, 8.5], [9.0, 9.5]]
add2queue(test_name, res, out)


test_name = "MatrixOps::ones<int>"
res = np.full((5, 5), 1)
out = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
add2queue(test_name, res, out)


test_name = "MatrixOps::onesLike<int>"
res = np.full((5, 5), 1)
out = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
add2queue(test_name, res, out)


test_name = "MatrixOps::zeros<int>"
res = np.full((5, 5), 0)
out = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
add2queue(test_name, res, out)


test_name = "MatrixOps::zerosLike<int>"
res = np.full((5, 5), 0)
out = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
add2queue(test_name, res, out)


test_name = "MatrixOps::logspace(float, float, int, int)"
res = np.logspace(5, 10, 50, True, dtype=float, base=10)
res = res.reshape((5, 10))
out = [[100000.0, 126485.5078125, 159985.84375, 202358.890625, 255954.671875, 323745.9375, 409491.6875, 517947.625, 655128.6875, 828642.875], [1048113.125, 1325711.125, 1676832.5, 2120950.25, 2682697.5, 3393223.5, 4291936.0, 5428677.0, 6866489.5, 8685114.0], [10985411.0, 13894968.0, 17575102.0, 22229980.0, 28117674.0, 35564820.0, 44984344.0, 56898676.0, 71968576.0, 91029824.0], [115139656.0, 145634816.0, 184207152.0, 232995088.0, 294705344.0, 372759136.0, 471486816.0, 596363136.0, 754312128.0, 954096576.0], [1206792576.0, 1526419328.0, 1930701312.0, 2442054656, 3088841984, 3906941696, 4941720576, 6250553344, 7906035200, 10000000000]]
add2queue(test_name, res, out)


test_name = "MatrixOps::logspace(float, float, float, int, int)"
res = np.logspace(7, 12, 2, True, dtype=float, base=10)
res = res.reshape((1, 2))
out = [[10000000.0, 999999995904]]
add2queue(test_name, res, out)


test_name = "MatrixOps::logspace(float, float, float, bool, int, int)"
res = np.logspace(5, 10, 10, False, dtype=float, base=10)
res = res.reshape((5, 2))
out = [[100000.0, 316227.78125], [1000000.0, 3162277.75], [10000000.0, 31622776.0], [100000000.0, 316227776.0], [1000000000.0, 3162277632]]
add2queue(test_name, res, out)


test_name = "MatrixOps::logspace(float, float, float, bool, float, int, int)"
res = np.logspace(5, 10, 10, False, dtype=float, base=2)
res = res.reshape((5, 2))
out = [[32.0, 45.254833221435547], [64.0, 90.509666442871094], [128.0, 181.01933288574219], [256.0, 362.03866577148438], [512.0, 724.07733154296875]]
add2queue(test_name, res, out)


test_name = "MatrixOps::geomspace(float, float, int, int)"
res = np.geomspace(5, 10, 50, True, dtype=float)
res = res.reshape((25, 2))
out = [[5.0, 5.0712318420410156], [5.1434793472290039, 5.2167549133300781], [5.2910747528076172, 5.3664536476135254], [5.4429059028625488, 5.5204477310180664], [5.5990939140319824, 5.678861141204834], [5.7597641944885254, 5.8418197631835938], [5.9250450134277344, 6.0094552040100098], [6.0950679779052734, 6.1819014549255371], [6.2699708938598633, 6.3592948913574219], [6.4498920440673828, 6.5417799949645996], [6.6349763870239258, 6.7295017242431641], [6.8253722190856934, 6.9226088523864746], [7.0212306976318359, 7.1212577819824219], [7.2227106094360352, 7.3256077766418457], [7.4299721717834473, 7.5358219146728516], [7.6431798934936523, 7.7520670890808105], [7.8625068664550781, 7.9745187759399414], [8.0881261825561523, 8.2033538818359375], [8.3202219009399414, 8.4387540817260742], [8.5589761734008789, 8.6809110641479492], [8.8045825958251953, 8.9300165176391602], [9.0572366714477539, 9.1862697601318359], [9.3171405792236328, 9.4498748779296875], [9.584503173828125, 9.7210483551025391], [9.8595380783081055, 10.0]]
add2queue(test_name, res, out)


test_name = "MatrixOps::geomspace(float, float, float, int, int)"
res = np.geomspace(7, 12, 2, True, dtype=float)
res = res.reshape((2, 1))
out = [[6.9999995231628418], [11.999999046325684]]
add2queue(test_name, res, out)


test_name = "MatrixOps::geomspace(float, float, float, bool, int, int)"
res = np.geomspace(5, 10, 10, False, dtype=float)
res = res.reshape((2, 5))
out = [[5.0, 5.3588676452636719, 5.7434921264648438, 6.1557216644287109, 6.5975399017333984], [7.071068286895752, 7.578582763671875, 8.122523307800293, 8.70550537109375, 9.3303308486938477]]
add2queue(test_name, res, out)


test_name = "MatrixOps::eye(int, int, int)"
res = np.eye(7, 6, 2, dtype=int)
out = [[0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
add2queue(test_name, res, out)


test_name = "MatrixOps::eye(int, int)"
res = np.eye(7, 7, 2, dtype=int)
out = [[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]
add2queue(test_name, res, out)


test_name = "MatrixOps::eye(int)"
res = np.eye(4, 4, 0, dtype=int)
out = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
add2queue(test_name, res, out)


test_name = "MatrixOps::meshgrid(Vector<int>, Vector<int>)"
a = np.arange(0, 10, 1, dtype=int)
b = np.arange(0, 10, 1, dtype=int)
res = np.meshgrid(a, b)
comp = np.array([[
 [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
 [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
 [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
 [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
 [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
 [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
 [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
 [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
 [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
 [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
], [
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
 [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
 [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
 [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
 [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
 [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0],
 [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0],
 [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
 [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
]])
out = comp
add2queue(test_name, res, out)


test_name = "MatrixOps::meshgrid(Matrix<int>, Matrix<int>)"
a = np.arange(0, 10, 1, dtype=int)
b = np.arange(0, 10, 1, dtype=int)
a = a.reshape((5, 2))
b = b.reshape((5, 2))
res = np.meshgrid(a, b)
comp = np.array([[
 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
], [
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
 [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
 [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
 [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
 [6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
 [7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
 [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
 [9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
]])
out = comp
add2queue(test_name, res, out)


test_name = "MatrixOps::diag(Matrix<int>, int)"
mat = np.arange(0, 9, 1, dtype=int)
mat = mat.reshape((3, 3))
res = np.diag(mat, 1)
out = [1, 5]
add2queue(test_name, res, out)


test_name = "MatrixOps::diag(Matrix<int>)"
mat = np.arange(0, 9, 1, dtype=int)
mat = mat.reshape((3, 3))
res = np.diag(mat, 0)
out = [0, 4, 8]
add2queue(test_name, res, out)


test_name = "MatrixOps::diagflat(Vector<int>, int)"
mat = np.arange(0, 9, 1, dtype=int)
res = np.diagflat(mat, 1)
out = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 3, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 4, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 5, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 6, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 7, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 8], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
add2queue(test_name, res, out)


test_name = "MatrixOps::diagflat(Vector<int>)"
mat = np.arange(0, 9, 1, dtype=int)
res = np.diagflat(mat, 0)
out = [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0, 0, 0, 0], [0, 0, 0, 3, 0, 0, 0, 0, 0], [0, 0, 0, 0, 4, 0, 0, 0, 0], [0, 0, 0, 0, 0, 5, 0, 0, 0], [0, 0, 0, 0, 0, 0, 6, 0, 0], [0, 0, 0, 0, 0, 0, 0, 7, 0], [0, 0, 0, 0, 0, 0, 0, 0, 8]]
add2queue(test_name, res, out)


test_name = "MatrixOps::tri(int, int, int)"
res = np.tri(2, 2, 1, dtype=int)
out = [[1, 1], [1, 1]]
add2queue(test_name, res, out)


test_name = "MatrixOps::tri(int, int)"
res = np.tri(5, 5, 2, dtype=int)
out = [[1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
add2queue(test_name, res, out)


test_name = "MatrixOps::tri(int)"
res = np.tri(7, 7, 0, dtype=int)
out = [[1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1]]
add2queue(test_name, res, out)


test_name = "MatrixOps::tril(Matrix<int>, int)"
mat = np.arange(0, 9, 1, dtype=int)
mat = mat.reshape((3, 3))
res = np.tril(mat, 1)
out = [[0, 1, 0], [3, 4, 5], [6, 7, 8]]
add2queue(test_name, res, out)


test_name = "MatrixOps::tril(Matrix<int>)"
mat = np.arange(0, 9, 1, dtype=int)
mat = mat.reshape((3, 3))
res = np.tril(mat, 0)
out = [[0, 0, 0], [3, 4, 0], [6, 7, 8]]
add2queue(test_name, res, out)


test_name = "MatrixOps::triu(Matrix<int>, int)"
mat = np.arange(0, 9, 1, dtype=int)
mat = mat.reshape((3, 3))
res = np.triu(mat, 1)
out = [[0, 1, 2], [0, 0, 5], [0, 0, 0]]
add2queue(test_name, res, out)


test_name = "MatrixOps::triu(Matrix<int>)"
mat = np.arange(0, 9, 1, dtype=int)
mat = mat.reshape((3, 3))
res = np.triu(mat, 0)
out = [[0, 1, 2], [0, 4, 5], [0, 0, 8]]
add2queue(test_name, res, out)


test_name = "MatrixOps::vander(Vector<int>, int, bool)"
vec = np.arange(0, 6, 1, dtype=int)
res = np.vander(vec, 4, True)
out = [[1, 0, 0, 0], [1, 1, 1, 1], [1, 2, 4, 8], [1, 3, 9, 27], [1, 4, 16, 64], [1, 5, 25, 125]]
add2queue(test_name, res, out)


test_name = "MatrixOps::vander(Vector<int>, int)"
vec = np.arange(0, 6, 1, dtype=int)
res = np.vander(vec, 7, False)
out = [[0, 0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1, 1], [64, 32, 16, 8, 4, 2, 1], [729, 243, 81, 27, 9, 3, 1], [4096, 1024, 256, 64, 16, 4, 1], [15625, 3125, 625, 125, 25, 5, 1]]
add2queue(test_name, res, out)


test_name = "MatrixOps::vander(Vector<int>)"
vec = np.arange(0, 6, 1, dtype=int)
res = np.vander(vec, None, False)
out = [[0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1], [32, 16, 8, 4, 2, 1], [243, 81, 27, 9, 3, 1], [1024, 256, 64, 16, 4, 1], [3125, 625, 125, 25, 5, 1]]
add2queue(test_name, res, out)


test_name = "ArrayOps::append(Vector<int>*, int)"
vec = np.arange(1, 4, 1, dtype=int)
res = np.append(vec, 4, axis = None)
out = [1, 2, 3, 4]
add2queue(test_name, res, out)


test_name = "ArrayOps::append(Matrix<int>*, Vector<int>*)"
vec = np.arange(1, 4, 1, dtype=int)
mat = np.arange(0, 9, 1, dtype=int)
res = np.append(mat, vec, axis = None)
res = res.reshape((4, 3))
out = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [1, 2, 3]]
add2queue(test_name, res, out)


test_name = "ArrayOps::concatenate(Vector<float>, Vector<float>)"
a = np.arange(1, 4, 1, dtype=int)
b = np.arange(1, 4, 1, dtype=int)
res = np.concatenate((a, b))
out = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
add2queue(test_name, res, out)


test_name = "ArrayOps::concatenate(Matrix<int>, Matrix<int>)"
a = np.arange(0, 4, 1, dtype=int)
a = a.reshape((2, 2))
b = np.arange(0, 4, 1, dtype=int)
b = b.reshape((2, 2))
res = np.concatenate((a, b))
out = [[0, 1], [2, 3], [0, 1], [2, 3]]
add2queue(test_name, res, out)


test_name = "ArrayOps::split(Vector<int>, int)"
a = np.arange(0, 15, 1, dtype=int)
res = np.split(a, 5)
comp = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]])
out = comp
add2queue(test_name, res, out)


test_name = "ArrayOps::split(Matrix<int>, int)"
a = np.arange(0, 15, 1, dtype=int)
res = np.split(a, 5)
comp = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]])
out = comp
add2queue(test_name, res, out)


test_name = "ArrayOps::tile(Vector<int>, int)"
a = np.arange(0, 15, 1, dtype=int)
res = np.tile(a, 5)
out = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
add2queue(test_name, res, out)


test_name = "ArrayOps::tile(Vector<int>, int, int)"
a = np.arange(0, 15, 1, dtype=int)
res = np.tile(a, (2, 2))
out = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
add2queue(test_name, res, out)


test_name = "ArrayOps::tile(Matrix<int>, int)"
a = np.arange(0, 15, 1, dtype=int)
a = a.reshape((3, 5))
res = np.tile(a, 5)
out = [[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 10, 11, 12, 13, 14, 10, 11, 12, 13, 14, 10, 11, 12, 13, 14, 10, 11, 12, 13, 14]]
add2queue(test_name, res, out)


test_name = "ArrayOps::tile(Matrix<int>, int, int)"
a = np.arange(0, 15, 1, dtype=int)
a = a.reshape((3, 5))
res = np.tile(a, (2, 2))
out = [[0, 1, 2, 3, 4, 0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 10, 11, 12, 13, 14], [0, 1, 2, 3, 4, 0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 10, 11, 12, 13, 14]]
add2queue(test_name, res, out)


test_name = "ArrayOps::remove(Vector<int>, int)"
a = np.arange(0, 20, 1, dtype=int)
res = np.delete(a, 3)
out = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
add2queue(test_name, res, out)


test_name = "ArrayOps::remove(Matrix<int>, int, axis=-1)"
a = np.arange(0, 20, 1, dtype=int)
res = np.delete(a, 3)
out = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
add2queue(test_name, res, out)


test_name = "ArrayOps::remove(Matrix<int>, int, axis=0)"
a = np.arange(0, 20, 1, dtype=int)
a = a.reshape((4, 5))
res = np.delete(a, 3, axis=0)
out = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]
add2queue(test_name, res, out)


test_name = "ArrayOps::remove(Matrix<int>, int, axis=1)"
a = np.arange(0, 20, 1, dtype=int)
a = a.reshape((4, 5))
res = np.delete(a, 3, axis=1)
out = [[0, 1, 2, 4], [5, 6, 7, 9], [10, 11, 12, 14], [15, 16, 17, 19]]
add2queue(test_name, res, out)


test_name = "ArrayOps::trimZeros(Vector<int>, 'fb')"
a = np.array([0, 0, 0, 1, 2, 3, 0, 2, 1, 0])
res = np.trim_zeros(a, 'fb')
out = [1, 2, 3, 0, 2, 1]
add2queue(test_name, res, out)


test_name = "ArrayOps::trimZeros(Vector<int>, 'f')"
a = np.array([0, 0, 0, 1, 2, 3, 0, 2, 1, 0])
res = np.trim_zeros(a, 'f')
out = [1, 2, 3, 0, 2, 1, 0]
add2queue(test_name, res, out)


test_name = "ArrayOps::trimZeros(Vector<int>, 'b')"
a = np.array([0, 0, 0, 1, 2, 3, 0, 2, 1, 0])
res = np.trim_zeros(a, 'b')
out = [0, 0, 0, 1, 2, 3, 0, 2, 1]
add2queue(test_name, res, out)


test_name = "ArrayOps::unique(a, false, false, false)"
a = np.array([0, 0, 0, 1, 3, 2, 0, 2, 1, 0])
res = np.unique(a, True, True, True)
comp = [[0, 1, 2, 3], [0, 3, 5, 4], [0, 0, 0, 1, 3, 2, 0, 2, 1, 0], [5, 2, 2, 1]]
res = res[0]
out = comp[0]
add2queue(test_name, res, out)


test_name = "ArrayOps::unique(a, true, false, false)"
a = np.array([0, 0, 0, 1, 3, 2, 0, 2, 1, 0])
res = np.unique(a, True, True, True)
comp = [[0, 1, 2, 3], [0, 3, 5, 4], [0, 0, 0, 1, 3, 2, 0, 2, 1, 0], [5, 2, 2, 1]]
res = res[1]
out = comp[1]
add2queue(test_name, res, out)


test_name = "ArrayOps::unique(a, false, true, false)"
a = np.array([0, 0, 0, 1, 3, 2, 0, 2, 1, 0])
res = np.unique(a, True, True, True)
comp = [[0, 1, 2, 3], [0, 3, 5, 4], [0, 0, 0, 1, 3, 2, 0, 2, 1, 0], [5, 2, 2, 1]]
res = res[2]
out = comp[2]
add2queue(test_name, res, out)


test_name = "ArrayOps::unique(a, false, false, true)"
a = np.array([0, 0, 0, 1, 3, 2, 0, 2, 1, 0])
res = np.unique(a, True, True, True)
comp = [[0, 1, 2, 3], [0, 3, 5, 4], [0, 0, 0, 1, 3, 2, 0, 2, 1, 0], [5, 2, 2, 1]]
res = res[3]
out = comp[3]
add2queue(test_name, res, out)


test_name = "Searching::argmax(Vector<int>)"
vec = np.array([5, 2, 25, 25, 22, 21, 4, 7, 11])
res = np.argmax(vec)
out = 2
add2queue(test_name, res, out)


test_name = "Searching::argmin(Vector<int>)"
vec = np.array([5, 2, 25, 25, 22, 21, 4, 7, 11])
res = np.argmin(vec)
out = 1
add2queue(test_name, res, out)


test_name = "Searching::nonzero(Vector<int>)"
vec = np.array([5, 0, 25, 0, 22, 0, 4, 0, 11])
res = np.nonzero(vec)
out = [0, 2, 4, 6, 8]
add2queue(test_name, res, out)


test_name = "Searching::nonzero(Matrix<int>)"
vec = np.array([5, 0, 25, 0, 22, 0, 4, 0, 11])
mat = vec.reshape((3, 3))
res = np.nonzero(mat)
out = [[0, 0, 1, 2, 2], [0, 2, 1, 0, 2]]
add2queue(test_name, res, out)


test_name = "Searching::argwhere(Vector<int>)"
vec = np.arange(0, 6, 1, dtype=int)
res = np.argwhere(vec)
out = [[1], [2], [3], [4], [5]]
add2queue(test_name, res, out)


test_name = "Searching::argwhere(Matrix<int>)"
vec = np.arange(0, 6, 1, dtype=int)
mat = vec.reshape((2, 3))
res = np.argwhere(mat)
out = [[0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
add2queue(test_name, res, out)


test_name = "Searching::flatnonzero(Vector<int>)"
vec = np.array([5, 0, 25, 0, 22, 0, 4, 0, 11])
res = np.flatnonzero(vec)
out = [0, 2, 4, 6, 8]
add2queue(test_name, res, out)


test_name = "Searching::flatnonzero(Matrix<int>)"
vec = np.array([5, 0, 25, 0, 22, 0, 4, 0, 11])
mat = vec.reshape((3, 3))
res = np.flatnonzero(mat)
out = [0, 2, 4, 6, 8]
add2queue(test_name, res, out)


test_name = "Searching::where(Vector<bool>, Vector<int>, Vector<int>)"
cond = np.array([true, false, true, true, false, false, true, false])
a = np.arange(0, 8, 1, dtype=int)
b = np.arange(14, 22, 1, dtype=int)
res = np.where(cond, a, b)
out = [0, 15, 2, 3, 18, 19, 6, 21]
add2queue(test_name, res, out)


test_name = "Searching::searchsorted(Vector<int>, int, 'left', Vector<int>)"
vec = np.arange(1, 6, 1, dtype=int)
v = np.array([-10, 10, 2, 3])
sorter = np.arange(0, 5, 1, dtype=int)
res = np.searchsorted(vec, v[0], 'left', sorter)
out = 0
add2queue(test_name, res, out)


test_name = "Searching::searchsorted(Vector<int>, int, 'right', Vector<int>)"
vec = np.arange(1, 6, 1, dtype=int)
v = np.array([-10, 10, 2, 3])
sorter = np.arange(0, 5, 1, dtype=int)
res = np.searchsorted(vec, v[0], 'right', sorter)
out = 0
add2queue(test_name, res, out)


test_name = "Searching::searchsorted(Vector<int>, int, 'left')"
vec = np.arange(1, 6, 1, dtype=int)
v = np.array([-10, 10, 2, 3])
sorter = np.arange(0, 5, 1, dtype=int)
res = np.searchsorted(vec, v[0], 'left', None)
out = 0
add2queue(test_name, res, out)


test_name = "Searching::searchsorted(Vector<int>, int, 'left')"
vec = np.arange(1, 6, 1, dtype=int)
v = np.array([-10, 10, 2, 3])
sorter = np.arange(0, 5, 1, dtype=int)
res = np.searchsorted(vec, v[0], 'left', None)
out = 0
add2queue(test_name, res, out)


test_name = "Searching::searchsorted(Vector<int>, int, Vector<int>)"
vec = np.arange(1, 6, 1, dtype=int)
v = np.array([-10, 10, 2, 3])
sorter = np.arange(0, 5, 1, dtype=int)
res = np.searchsorted(vec, v[0], 'left', sorter)
out = 0
add2queue(test_name, res, out)


test_name = "Searching::searchsorted(Vector<int>, int)"
vec = np.arange(1, 6, 1, dtype=int)
v = np.array([-10, 10, 2, 3])
sorter = np.arange(0, 5, 1, dtype=int)
res = np.searchsorted(vec, v[0], 'left', None)
out = 0
add2queue(test_name, res, out)


test_name = "Searching::searchsorted(Vector<int>, Vector<int>, 'left', Vector<int>)"
vec = np.arange(1, 6, 1, dtype=int)
v = np.array([-10, 10, 2, 3])
sorter = np.arange(0, 5, 1, dtype=int)
res = np.searchsorted(vec, v, 'left', sorter)
out = [0, 5, 1, 2]
add2queue(test_name, res, out)


test_name = "Searching::searchsorted(Vector<int>, Vector<int>, 'right', Vector<int>)"
vec = np.arange(1, 6, 1, dtype=int)
v = np.array([-10, 10, 2, 3])
sorter = np.arange(0, 5, 1, dtype=int)
res = np.searchsorted(vec, v, 'right', sorter)
out = [0, 5, 2, 3]
add2queue(test_name, res, out)


test_name = "Searching::searchsorted(Vector<int>, Vector<int>, 'left')"
vec = np.arange(1, 6, 1, dtype=int)
v = np.array([-10, 10, 2, 3])
sorter = np.arange(0, 5, 1, dtype=int)
res = np.searchsorted(vec, v, 'left', None)
out = [0, 5, 1, 2]
add2queue(test_name, res, out)


test_name = "Searching::searchsorted(Vector<int>, Vector<int>, 'right')"
vec = np.arange(1, 6, 1, dtype=int)
v = np.array([-10, 10, 2, 3])
sorter = np.arange(0, 5, 1, dtype=int)
res = np.searchsorted(vec, v, 'right', None)
out = [0, 5, 2, 3]
add2queue(test_name, res, out)


test_name = "Searching::searchsorted(Vector<int>, Vector<int>, Vector<int>)"
vec = np.arange(1, 6, 1, dtype=int)
v = np.array([-10, 10, 2, 3])
sorter = np.arange(0, 5, 1, dtype=int)
res = np.searchsorted(vec, v, 'left', sorter)
out = [0, 5, 1, 2]
add2queue(test_name, res, out)


test_name = "Searching::searchsorted(Vector<int>, Vector<int>)"
vec = np.arange(1, 6, 1, dtype=int)
v = np.array([-10, 10, 2, 3])
sorter = np.arange(0, 5, 1, dtype=int)
res = np.searchsorted(vec, v, 'left', None)
out = [0, 5, 1, 2]
add2queue(test_name, res, out)


test_name = "Searching::extract(Vector<int>, Vector<int>)"
cond = np.array([true, false, false, true, false, true, true, false, true])
vec = np.arange(0, 9, 1, dtype=int)
res = np.extract(cond, vec)
out = [0, 3, 5, 6, 8]
add2queue(test_name, res, out)


test_name = "Searching::count_nonzero(Vector<int>)"
v = np.array([0, 1, 7, 0, 3, 0, 2, 19])
res = np.count_nonzero(v)
out = 5
add2queue(test_name, res, out)


test_name = "Sorting::quicksort(Vector<int>)"
vec = np.array([1, 25, 5820, 5, 58, 22, 48, 14, 32, 22, 99, 90900, 15, 11])
res = np.sort(vec, kind='quicksort')
out = [1, 5, 11, 14, 15, 22, 22, 25, 32, 48, 58, 99, 5820, 90900]
add2queue(test_name, res, out)


test_name = "Sorting::mergesort(Vector<int>)"
vec = np.array([1, 25, 5820, 5, 58, 22, 48, 14, 32, 22, 99, 90900, 15, 11])
res = np.sort(vec, kind='mergesort')
out = [1, 5, 11, 14, 15, 22, 22, 25, 32, 48, 58, 99, 5820, 90900]
add2queue(test_name, res, out)


test_name = "Sorting::heapsort(Vector<int>)"
vec = np.array([1, 25, 5820, 5, 58, 22, 48, 14, 32, 22, 99, 90900, 15, 11])
res = np.sort(vec, kind='heapsort')
out = [1, 5, 11, 14, 15, 22, 22, 25, 32, 48, 58, 99, 5820, 90900]
add2queue(test_name, res, out)


test_name = "Sorting::sort(Vector<int>, Kind::Quicksort)"
vec = np.array([1, 25, 5820, 5, 58, 22, 48, 14, 32, 22, 99, 90900, 15, 11])
res = np.sort(vec, kind='quicksort')
out = [1, 5, 11, 14, 15, 22, 22, 25, 32, 48, 58, 99, 5820, 90900]
add2queue(test_name, res, out)


test_name = "Sorting::sort(Vector<int>, Kind::Mergesort)"
vec = np.array([1, 25, 5820, 5, 58, 22, 48, 14, 32, 22, 99, 90900, 15, 11])
res = np.sort(vec, kind='mergesort')
out = [1, 5, 11, 14, 15, 22, 22, 25, 32, 48, 58, 99, 5820, 90900]
add2queue(test_name, res, out)


test_name = "Sorting::sort(Vector<int>, Kind::Heapsort)"
vec = np.array([1, 25, 5820, 5, 58, 22, 48, 14, 32, 22, 99, 90900, 15, 11])
res = np.sort(vec, kind='heapsort')
out = [1, 5, 11, 14, 15, 22, 22, 25, 32, 48, 58, 99, 5820, 90900]
add2queue(test_name, res, out)


test_name = "Sorting::sort(Vector<int>)"
vec = np.array([1, 25, 5820, 5, 58, 22, 48, 14, 32, 22, 99, 90900, 15, 11])
res = np.sort(vec, kind='quicksort')
out = [1, 5, 11, 14, 15, 22, 22, 25, 32, 48, 58, 99, 5820, 90900]
add2queue(test_name, res, out)


test_name = "Sorting::argsort(Vector<int>, Kind::Quicksort)"
vec = np.array([1, 25, 5820, 5, 58, 22, 48, 14, 32, 22, 99, 90900, 15, 11])
res = np.argsort(vec, kind='quicksort')
out = [0, 3, 13, 7, 12, 5, 9, 1, 8, 6, 4, 10, 2, 11]
add2queue(test_name, res, out)


test_name = "Sorting::argsort(Vector<int>, Kind::Mergesort)"
vec = np.array([1, 25, 5820, 5, 58, 22, 48, 14, 32, 22, 99, 90900, 15, 11])
res = np.argsort(vec, kind='mergesort')
out = [0, 3, 13, 7, 12, 5, 9, 1, 8, 6, 4, 10, 2, 11]
add2queue(test_name, res, out)


test_name = "Sorting::argsort(Vector<int>, Kind::Heapsort)"
vec = np.array([1, 25, 5820, 5, 58, 22, 48, 14, 32, 22, 99, 90900, 15, 11])
res = np.argsort(vec, kind='heapsort')
out = [0, 3, 13, 7, 12, 5, 9, 1, 8, 6, 4, 10, 2, 11]
add2queue(test_name, res, out)


test_name = "Sorting::argsort(Vector<int>)"
vec = np.array([1, 25, 5820, 5, 58, 22, 48, 14, 32, 22, 99, 90900, 15, 11])
res = np.argsort(vec, kind='quicksort')
out = [0, 3, 13, 7, 12, 5, 9, 1, 8, 6, 4, 10, 2, 11]
add2queue(test_name, res, out)


test_name = "Logic::all(Vector<bool>)"
vec = np.array([true, true, false, true, true])
res = np.all(vec)
out = 0
add2queue(test_name, res, out)


test_name = "Logic::any(Vector<bool>)"
vec = np.array([true, true, false, true, true])
res = np.any(vec)
out = 1
add2queue(test_name, res, out)


test_name = "Logic::logicalAnd(Vector<bool>)"
a = np.array([true, true, false, true, true])
b = np.array([false, true, true, false, true])
res = np.logical_and(a, b)
out = [0, 1, 0, 0, 1]
add2queue(test_name, res, out)


test_name = "Logic::logicalOr(Vector<bool>)"
a = np.array([true, true, false, true, true])
b = np.array([false, true, true, false, true])
res = np.logical_or(a, b)
out = [1, 1, 1, 1, 1]
add2queue(test_name, res, out)


test_name = "Logic::logicalNot(Vector<bool>)"
a = np.array([true, true, false, true, true])
res = np.logical_not(a)
out = [0, 0, 1, 0, 0]
add2queue(test_name, res, out)


test_name = "Logic::logicalXor(Vector<bool>)"
a = np.array([true, true, false, true, true])
b = np.array([false, true, true, false, true])
res = np.logical_xor(a, b)
out = [1, 0, 1, 1, 0]
add2queue(test_name, res, out)


test_name = "Logic::allclose(Vector<int>)"
a = np.array([255, 278, 27872, 2, 278])
b = np.array([2782, 278278, 2222, 2, 278])
res = np.allclose(a, b)
out = 0
add2queue(test_name, res, out)


test_name = "Logic::equals(Vector<bool>)"
a = np.array([true, true, false, true, true])
b = np.array([false, true, true, false, true])
res = np.equal(a, b)
out = [0, 1, 0, 0, 1]
add2queue(test_name, res, out)


test_name = "Logic::greater(Vector<int>, Vector<int>)"
a = np.array([255, 278, 27872, 2, 278])
b = np.array([2782, 278278, 2222, 2, 278])
res = np.greater(a, b)
out = [0, 0, 1, 0, 0]
add2queue(test_name, res, out)


test_name = "Logic::greater(Vector<int>, int)"
a = np.array([255, 278, 27872, 2, 278])
b = np.array([2782, 278278, 2222, 2, 278])
res = np.greater(a, b[2])
out = [0, 0, 1, 0, 0]
add2queue(test_name, res, out)


test_name = "Logic::greaterEquals(Vector<int>, Vector<int>)"
a = np.array([255, 278, 27872, 2, 278])
b = np.array([2782, 278278, 2222, 2, 278])
res = np.greater_equal(a, b)
out = [0, 0, 1, 1, 1]
add2queue(test_name, res, out)


test_name = "Logic::greaterEquals(Vector<int>, int)"
a = np.array([255, 278, 27872, 2, 278])
b = np.array([2782, 278278, 2222, 2, 278])
res = np.greater_equal(a, b[2])
out = [0, 0, 1, 0, 0]
add2queue(test_name, res, out)


test_name = "Logic::less(Vector<int>, Vector<int>)"
a = np.array([255, 278, 27872, 2, 278])
b = np.array([2782, 278278, 2222, 2, 278])
res = np.less(a, b)
out = [1, 1, 0, 0, 0]
add2queue(test_name, res, out)


test_name = "Logic::less(Vector<int>, int)"
a = np.array([255, 278, 27872, 2, 278])
b = np.array([2782, 278278, 2222, 2, 278])
res = np.less(a, b[2])
out = [1, 1, 0, 1, 1]
add2queue(test_name, res, out)


test_name = "Logic::equals(Vector<int>, Vector<int>)"
a = np.array([255, 278, 27872, 2, 278])
b = np.array([2782, 278278, 2222, 2, 278])
res = np.less_equal(a, b)
out = [1, 1, 0, 1, 1]
add2queue(test_name, res, out)


test_name = "Logic::equals(Vector<int>, int)"
a = np.array([255, 278, 27872, 2, 278])
b = np.array([2782, 278278, 2222, 2, 278])
res = np.less_equal(a, b[2])
out = [1, 1, 0, 1, 1]
add2queue(test_name, res, out)


test_name = "Logic::maximum(Vector<int>, Vector<int>)"
a = np.array([255, 278, 27872, 2, 278])
b = np.array([2782, 278278, 2222, 2, 278])
res = np.maximum(a, b)
out = [2782, 278278, 27872, 2, 278]
add2queue(test_name, res, out)


test_name = "Logic::amax(Vector<int>, Vector<int>)"
a = np.array([255, 278, 27872, 2, 278])
res = np.amax(a)
out = 27872
add2queue(test_name, res, out)


test_name = "Logic::minimum(Vector<int>, Vector<int>)"
a = np.array([255, 278, 27872, 2, 278])
b = np.array([2782, 278278, 2222, 2, 278])
res = np.minimum(a, b)
out = [255, 278, 2222, 2, 278]
add2queue(test_name, res, out)


test_name = "Logic::amin(Vector<int>, Vector<int>)"
a = np.array([255, 278, 27872, 2, 278])
res = np.amin(a)
out = 2
add2queue(test_name, res, out)


test_name = "BaseMath::cubeRoot(Vector<int>)"
vec = np.array([1, 8, 27])
res = np.cbrt(vec)
out = [1, 2, 3]
add2queue(test_name, res, out)


test_name = "BaseMath::square(Vector<int>)"
vec = np.array([1, 8, 27])
res = np.square(vec)
out = [1, 64, 729]
add2queue(test_name, res, out)


test_name = "BaseMath::squareRoot(Vector<double>)"
vec = np.array([1, 8, 27])
res = np.sqrt(vec)
out = [1.0, 2.8284271247461903, 5.196152422706632]
add2queue(test_name, res, out)


test_name = "BaseMath::power(Vector<double>, Vector<double>)"
base = np.array([1, 8, 27])
power = np.array([2, 6, 5])
res = np.power(base, power)
out = [1.0, 262144.0, 14348907.0]
add2queue(test_name, res, out)


test_name = "BaseMath::around(Vector<double>, int)"
vec = np.array([1.584, 8.45475, 27.5])
res = np.around(vec, decimals=3)
out = [1.5840000000000001, 8.4550000000000001, 27.5]
add2queue(test_name, res, out)


test_name = "BaseMath::around(Vector<double>)"
vec = np.array([1.584, 8.45475, 27.5])
res = np.around(vec, decimals=0)
out = [2.0, 8.0, 28.0]
add2queue(test_name, res, out)


test_name = "BaseMath::rint(Vector<double>)"
vec = np.array([1.584, 8.45475, 27.5])
res = np.rint(vec)
out = [2.0, 8.0, 28.0]
add2queue(test_name, res, out)


test_name = "BaseMath::fix(Vector<double>)"
vec = np.array([1.584, 8.45475, 27.5, -54.9, -2.2])
res = np.fix(vec)
out = [1.0, 8.0, 27.0, -54.0, -2.0]
add2queue(test_name, res, out)


test_name = "BaseMath::floor(Vector<double>)"
vec = np.array([1.584, 8.45475, 27.5, -54.9, -2.2])
res = np.floor(vec)
out = [1.0, 8.0, 27.0, -55.0, -3.0]
add2queue(test_name, res, out)


test_name = "BaseMath::trunc(Vector<double>)"
vec = np.array([1.584, 8.45475, 27.5, -54.9, -2.2])
res = np.trunc(vec)
out = [1.0, 8.0, 27.0, -54.0, -2.0]
add2queue(test_name, res, out)


test_name = "BaseMath::ceil(Vector<double>)"
vec = np.array([1.584, 8.45475, 27.5, -54.9, -2.2])
res = np.ceil(vec)
out = [2.0, 9.0, 28.0, -54.0, -2.0]
add2queue(test_name, res, out)


test_name = "BaseMath::prod(Vector<double>)"
vec = np.array([1.584, 8.45475, 27.5, -54.9, -2.2])
res = np.prod(vec)
out = 44481.934550
add2queue(test_name, res, out)


test_name = "BaseMath::sum(Vector<double>)"
vec = np.array([1.584, 8.45475, 27.5, -54.9, -2.2])
res = np.sum(vec)
out = -19.561250
add2queue(test_name, res, out)


test_name = "BaseMath::cumProd(Vector<double>)"
vec = np.array([1.584, 8.45475, 27.5, -54.9, -2.2])
res = np.cumprod(vec)
out = [1.5840000000000001, 13.392324000000002, 368.28891000000004, -20219.061159000001, 44481.934549800004]
add2queue(test_name, res, out)


test_name = "BaseMath::cumSum(Vector<double>)"
vec = np.array([1.584, 8.45475, 27.5, -54.9, -2.2])
res = np.cumsum(vec)
out = [1.5840000000000001, 10.03875, 37.53875, -17.361249999999998, -19.561249999999998]
add2queue(test_name, res, out)


test_name = "BaseMath::signBit(Vector<double>)"
vec = np.array([1.584, 8.45475, 27.5, -54.9, -2.2])
res = np.signbit(vec)
out = [0, 0, 0, 1, 1]
add2queue(test_name, res, out)


test_name = "BaseMath::copySign(Vector<double>, Vector<double>)"
a = np.array([1.584, 8.45475, 27.5, -54.9, -2.2])
b = np.array([-1, -1, 1, -1, 1])
res = np.copysign(a, b)
out = [-1.5840000000000001, -8.4547500000000007, 27.5, -54.899999999999999, 2.2000000000000002]
add2queue(test_name, res, out)


test_name = "BaseMath::abs(Vector<double>)"
vec = np.array([1.584, 8.45475, 27.5, -54.9, -2.2])
res = np.absolute(vec)
out = [1.5840000000000001, 8.4547500000000007, 27.5, 54.899999999999999, 2.2000000000000002]
add2queue(test_name, res, out)


test_name = "BaseMath::lcm(Vector<int>, Vector<int>)"
a = np.array([14, 102, 55])
b = np.array([10, 51, 11])
res = np.lcm(a, b)
out = [70, 102, 55]
add2queue(test_name, res, out)


test_name = "BaseMath::gcd(Vector<int>, Vector<int>)"
a = np.array([14, 102, 55])
b = np.array([10, 51, 11])
res = np.gcd(a, b)
out = [2, 51, 11]
add2queue(test_name, res, out)


test_name = "BaseMath::add(Vector<double>, Vector<double>)"
a = np.array([1.584, 8.45475, 27.5, -54.9, -2.2])
b = np.array([-1, -1, 1, -1, 1])
res = np.add(a, b)
out = [0.58400000000000007, 7.4547500000000007, 28.5, -55.899999999999999, -1.2000000000000002]
add2queue(test_name, res, out)


test_name = "BaseMath::subtract(Vector<double>, Vector<double>)"
a = np.array([1.584, 8.45475, 27.5, -54.9, -2.2])
b = np.array([-1, -1, 1, -1, 1])
res = np.subtract(a, b)
out = [2.5840000000000001, 9.4547500000000007, 26.5, -53.899999999999999, -3.2000000000000002]
add2queue(test_name, res, out)


test_name = "BaseMath::multiply(Vector<double>, Vector<double>)"
a = np.array([1.584, 8.45475, 27.5, -54.9, -2.2])
b = np.array([-1, -1, 1, -1, 1])
res = np.multiply(a, b)
out = [-1.5840000000000001, -8.4547500000000007, 27.5, 54.899999999999999, -2.2000000000000002]
add2queue(test_name, res, out)


test_name = "BaseMath::divide(Vector<double>, Vector<double>)"
a = np.array([1.584, 8.45475, 27.5, -54.9, -2.2])
b = np.array([-1, -1, 1, -1, 1])
res = np.divide(b, a)
out = [-0.63131313131313127, -0.11827670835920635, 0.036363636363636362, 0.018214936247723135, -0.45454545454545453]
add2queue(test_name, res, out)


test_name = "BaseMath::floorDivide(Vector<double>, Vector<double>)"
a = np.array([1.584, 8.45475, 27.5, -54.9, -2.2])
b = np.array([-1, -1, 1, -1, 1])
res = np.floor_divide(b, a)
out = [-1, -1, 0, 0, -1]
add2queue(test_name, res, out)


test_name = "BaseMath::mod(Vector<int>, Vector<int>)"
a = np.array([14, 102, 55])
b = np.array([10, 51, 11])
res = np.mod(a, b)
out = [4, 0, 0]
add2queue(test_name, res, out)


test_name = "BaseMath::divMod(Vector<int>, Vector<int>)"
a = np.array([14, 102, 55])
b = np.array([10, 51, 11])
res = np.divmod(a, b)
res = np.array([[1, 2, 5], [4, 0, 0]]
)
out = res
add2queue(test_name, res, out)


test_name = "BaseMath::reciprocal(Vector<int>,)"
a = np.array([14, 102, 55])
res = np.reciprocal(a)
out = [0, 0, 0]
add2queue(test_name, res, out)


test_name = "BaseMath::positive(Vector<double>)"
vec = np.array([1.584, 8.45475, 27.5, -54.9, -2.2])
res = np.positive(vec)
out = [1.5840000000000001, 8.4547500000000007, 27.5, -54.899999999999999, -2.2000000000000002]
add2queue(test_name, res, out)


test_name = "BaseMath::negative(Vector<double>)"
vec = np.array([1.584, 8.45475, 27.5, -54.9, -2.2])
res = np.negative(vec)
out = [-1.5840000000000001, -8.4547500000000007, -27.5, 54.899999999999999, 2.2000000000000002]
add2queue(test_name, res, out)


test_name = "BaseMath::sign(Vector<double>)"
vec = np.array([1.584, 8.45475, 27.5, -54.9, -2.2])
res = np.sign(vec)
out = [1.0, 1.0, 1.0, -1.0, -1.0]
add2queue(test_name, res, out)


test_name = "BaseMath::heaviside(Vector<double>, Vector<double>)"
a = np.array([1.584, 8.45475, 27.5, -54.9, -2.2])
b = np.array([-1, 0, 1, -1, 0])
res = np.heaviside(b, a)
out = [0.0, 8.4547500000000007, 1.0, 0.0, -2.2000000000000002]
add2queue(test_name, res, out)


test_name = "Bitwise::BitwiseAnd(Vector<int>, Vector<int>)"
a = np.array([15, 45, 22, 91, 37])
b = np.array([31, 69, 420, 11, 13])
res = np.bitwise_and(a, b)
out = [15, 5, 4, 11, 5]
add2queue(test_name, res, out)


test_name = "Bitwise::bitwiseOr(Vector<int>, Vector<int>)"
a = np.array([15, 45, 22, 91, 37])
b = np.array([31, 69, 420, 11, 13])
res = np.bitwise_or(a, b)
out = [31, 109, 438, 91, 45]
add2queue(test_name, res, out)


test_name = "Bitwise::bitwiseXor(Vector<int>, Vector<int>)"
a = np.array([15, 45, 22, 91, 37])
b = np.array([31, 69, 420, 11, 13])
res = np.bitwise_xor(a, b)
out = [16, 104, 434, 80, 40]
add2queue(test_name, res, out)


test_name = "Bitwise::bitwiseLeftShift(Vector<int>, Vector<int>)"
a = np.array([15, 45, 22, 91, 37])
b = np.array([2, 3, 5, 4, 3])
res = np.left_shift(a, b)
out = [60, 360, 704, 1456, 296]
add2queue(test_name, res, out)


test_name = "Bitwise::bitwiseRightShift(Vector<int>, Vector<int>)"
a = np.array([15, 45, 22, 91, 37])
b = np.array([2, 3, 5, 4, 3])
res = np.right_shift(a, b)
out = [3, 5, 0, 5, 4]
add2queue(test_name, res, out)


test_name = "Bitwise::BitwiseAnd(Vector<int>, Vector<int>)"
a = np.array([15, 45, 22, 91, 37])
res = np.bitwise_not(a)
out = [-16, -46, -23, -92, -38]
add2queue(test_name, res, out)


test_name = "ComplexType(int, int)"
res = 5+5j
out = 5+5j
add2queue(test_name, res, out)


test_name = "ComplexType(int)"
res = 5
out = 5+0j
add2queue(test_name, res, out)


test_name = "Complex::angle(Vector<ComplexType*>, bool)"
a = np.array([1+1j, 2+2j, 3+3j, 4+4j, 5+5j])
res = np.angle(a, true)
out = [45.0, 45.0, 45.0, 45.0, 45.0]
add2queue(test_name, res, out)


test_name = "Complex::angle(Vector<ComplexType*>)"
a = np.array([1+1j, 2+2j, 3+3j, 4+4j, 5+5j])
res = np.angle(a, false)
out = [0.78539816339744828, 0.78539816339744828, 0.78539816339744828, 0.78539816339744828, 0.78539816339744828]
add2queue(test_name, res, out)


test_name = "Complex::real(Vector<ComplexType*>)"
a = np.array([1+1j, 2+2j, 3+3j, 4+4j, 5+5j])
res = np.real(a)
out = [1.0, 2.0, 3.0, 4.0, 5.0]
add2queue(test_name, res, out)


test_name = "Complex::imag(Vector<ComplexType*>)"
a = np.array([1+1j, 2+2j, 3+3j, 4+4j, 5+5j])
res = np.imag(a)
out = [1.0, 2.0, 3.0, 4.0, 5.0]
add2queue(test_name, res, out)


test_name = "Complex::conj(Vector<ComplexType*>)"
a = np.array([1+1j, 2+2j, 3+3j, 4+4j, 5+5j])
res = np.conj(a)
out = [1+-1j, 2+-2j, 3+-3j, 4+-4j, 5+-5j]
add2queue(test_name, res, out)


test_name = "Complex::complexAbs(Vector<ComplexType*>)"
a = np.array([1+1j, 2+2j, 3+3j, 4+4j, 5+5j])
res = np.absolute(a)
out = [1.4142135623730951, 2.8284271247461903, 4.2426406871192848, 5.6568542494923806, 7.0710678118654755]
add2queue(test_name, res, out)


test_name = "Complex::complexSign(Vector<ComplexType*>)"
a = np.array([1+1j, 2+2j, 3+3j, 4+4j, 5+5j])
res = np.sign(a)
out = [1+0j, 1+0j, 1+0j, 1+0j, 1+0j]
add2queue(test_name, res, out)


test_name = "Exps::ln(Vector<double>)"
a = np.array([1, 2.71828, 15, 50, 666, 69, 420])
res = np.log(a)
out = [0.0, 0.99999932734728203, 2.7080502011022101, 3.912023005428146, 6.5012896705403893, 4.2341065045972597, 6.0402547112774139]
add2queue(test_name, res, out)


test_name = "Exps::loga2(Vector<double>)"
a = np.array([1, 2.71828, 15, 50, 666, 69, 420])
res = np.log2(a)
out = [0.0, 1.442694070456223, 3.9068905956085187, 5.6438561897747244, 9.3793783670712614, 6.1085244567781691, 8.7142455176661233]
add2queue(test_name, res, out)


test_name = "Exps::logan(Vector<double>, Vector<double>)"
b = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
res = np.log(a) / np.log(b)
out = [-0.0, inf, 3.9068905956085187, 3.5608767950073115, 4.6896891835356316, 2.630798287952365, 3.3711303414401605]
add2queue(test_name, res, out)


test_name = "Exps::logan(Vector<double>, double)"
a = np.array([1, 2.71828, 15, 50, 666, 69, 420])
res = np.log(a) / np.log(4)
out = [0.0, 0.7213470352281115, 1.9534452978042594, 2.8219280948873622, 4.6896891835356316, 3.054262228389085, 4.3571227588330617]
add2queue(test_name, res, out)


test_name = "Exps::loga10(Vector<double>)"
a = np.array([1, 2.71828, 15, 50, 666, 69, 420])
res = np.log10(a)
out = [0.0, 0.43429418977388817, 1.1760912590556813, 1.6989700043360187, 2.823474229170301, 1.8388490907372552, 2.6232492903979003]
add2queue(test_name, res, out)


test_name = "Exps::expo(Vector<double>)"
a = np.array([1, 2.71828, 15, 50, 666, 69, 420])
res = np.exp(a)
out = [2.7182818284590451, 15.154234532556728, 3269017.3724721107, 5.184705528587072e+21, 1.7383008701505047e+289, 9.2537817255877872e+29, 2.5332753623607178e+182]
add2queue(test_name, res, out)


test_name = "Exps::expom1(Vector<double>)"
a = np.array([1, 2.71828, 15, 50, 666, 69, 420])
res = np.expm1(a)
out = [1.7182818284590451, 14.154234532556728, 3269016.3724721107, 5.184705528587072e+21, 1.7383008701505047e+289, 9.2537817255877872e+29, 2.5332753623607178e+182]
add2queue(test_name, res, out)


test_name = "Exps::expo2(Vector<double>)"
a = np.array([1, 2.71828, 15, 50, 666, 69, 420])
res = np.exp2(a)
out = [2.0, 6.5808776504660029, 32768.0, 1125899906842624, 3.061802069160839e+200, 5.9029581035870565e+20, 2.7076852481648583e+126]
add2queue(test_name, res, out)


test_name = "Exps::logaddexp(Vector<double>, Vector<double>)"
b = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
res = np.logaddexp(a, b)
out = [1.3132616875182228, 2.8832641340880647, 15.000002260326852, 50.0, 666.0, 69.0, 420.0]
add2queue(test_name, res, out)


test_name = "Exps::logaddexp2(Vector<double>, Vector<double>)"
b = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
res = np.logaddexp2(a, b)
out = [1.5849625007211561, 3.1011252138344139, 15.000176099486442, 50.000000000000007, 666.0, 69.0, 420.0]
add2queue(test_name, res, out)


test_name = "Linalg::dot(Matrix<int>, Vector<int>)"
a = np.array([
 [0, 1, 2],
 [3, 4, 5]
])
b = np.array([0, 1, 2])
res = np.dot(a, b)
out = [5, 14]
add2queue(test_name, res, out)


test_name = "Linalg::dot(Vector<int>, Matrix<int>)"
a = np.array([
 [0, 1],
 [2, 3],
 [4, 5]
])
b = np.array([0, 1, 2])
res = np.dot(b, a)
out = [10, 13]
add2queue(test_name, res, out)


test_name = "Linalg::dot(Vector<int>, Vector<int>)"
a = np.array([
 [0, 1],
 [2, 3],
 [4, 5]
])
b = np.array([0, 1, 2])
res = np.dot(b, a)
b = np.array([0, 1, 2])
res = np.dot(b, b)
out = 5
add2queue(test_name, res, out)


test_name = "Linalg::dot(Matrix<int>, Vector<int>)"
a = np.array([
 [0, 1, 2],
 [3, 4, 5]
])
b = np.array([
 [0, 1],
 [2, 3],
 [4, 5]
])
res = np.dot(b, a)
out = [[3, 4, 5], [9, 14, 19], [15, 24, 33]]
add2queue(test_name, res, out)


test_name = "Linalg::inner(Vector<int>, Vector<int>)"
a = np.array([0, 1, 2, 3, 4, 5])
b = np.array([0, 1, 2, 3, 4, 5])
res = np.inner(a, b)
out = 55
add2queue(test_name, res, out)


test_name = "Linalg::inner(Vector<int>, Matrix<int>)"
a = np.array([
 [0, 1, 2],
 [3, 4, 5]
])
b = np.array([0, 1, 2])
res = np.inner(a, b)
out = [5, 14]
add2queue(test_name, res, out)


test_name = "Linalg::inner(Matrix<int>, Vector<int>)"
a = np.array([
 [0, 1, 2],
 [3, 4, 5]
])
b = np.array([0, 1, 2])
res = np.inner(b, a)
out = [5, 14]
add2queue(test_name, res, out)


test_name = "Linalg::inner(Matrix<int>, Vector<int>)"
a = np.array([
 [0, 1, 2],
 [3, 4, 5]
])
b = np.array([
 [0, 1, 2],
 [3, 4, 5]
])
res = np.inner(a, b)
out = [[5, 14], [14, 50]]
add2queue(test_name, res, out)


test_name = "Linalg::outer(Vector<int>, Vector<int>)"
a = np.array([0, 1, 2, 3, 4, 5])
b = np.array([0, 1, 2, 3, 4, 5])
res = np.outer(a, b)
out = [[0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5], [0, 2, 4, 6, 8, 10], [0, 3, 6, 9, 12, 15], [0, 4, 8, 12, 16, 20], [0, 5, 10, 15, 20, 25]]
add2queue(test_name, res, out)


test_name = "Linalg::outer(Vector<int>, Matrix<int>)"
a = np.array([
 [0, 1, 2],
 [3, 4, 5]
])
b = np.array([0, 1, 2])
res = np.outer(a, b)
out = [[0, 0, 0], [0, 1, 2], [0, 2, 4], [0, 3, 6], [0, 4, 8], [0, 5, 10]]
add2queue(test_name, res, out)


test_name = "Linalg::outer(Matrix<int>, Vector<int>)"
a = np.array([
 [0, 1, 2],
 [3, 4, 5]
])
b = np.array([0, 1, 2])
res = np.outer(b, a)
out = [[0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5], [0, 2, 4, 6, 8, 10]]
add2queue(test_name, res, out)


test_name = "Linalg::outer(Matrix<int>, Vector<int>)"
a = np.array([
 [0, 1, 2],
 [3, 4, 5]
])
b = np.array([
 [0, 1, 2],
 [3, 4, 5]
])
res = np.outer(a, b)
out = [[0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5], [0, 2, 4, 6, 8, 10], [0, 3, 6, 9, 12, 15], [0, 4, 8, 12, 16, 20], [0, 5, 10, 15, 20, 25]]
add2queue(test_name, res, out)


test_name = "Linalg::det(Matrix<int>)"
mat = np.array([
 [0, 1, 2, 3],
 [4, 5, 6, 7],
 [8, 9, 10, 11],
 [12, 13, 14, 15]
])
res = np.linalg.det(mat)
out = 0
add2queue(test_name, res, out)


test_name = "Linalg::trace(Matrix<int>, int)"
mat = np.array([
 [0, 1, 2, 3],
 [4, 5, 6, 7],
 [8, 9, 10, 11],
 [12, 13, 14, 15]
])
res = np.trace(mat, 2)
out = 9
add2queue(test_name, res, out)


test_name = "Linalg::trace(Matrix<int>)"
mat = np.array([
 [0, 1, 2],
 [3, 4, 5],
 [6, 7, 8],
 [9, 10, 11],
 [12, 13, 14]
])
res = np.trace(mat, 0)
out = 12
add2queue(test_name, res, out)


test_name = "Linalg::solve(Matrix<int>, Vector<int>)"
mat = np.array([
 [1.0, 2.0],
 [3.0, 5.0]
])
vec = np.array([1.0, 2.0])
res = np.linalg.solve(mat, vec)
out = [-0.99999999999999944, 0.99999999999999967]
add2queue(test_name, res, out)


test_name = "Linalg.roots(Vector<double>)"
p = np.array([1, 1, -1])
res = np.roots(p)
out = [-1.6180339887498945, 0.6180339887498949]
add2queue(test_name, res, out)


test_name = "Linalg.inv(Matrix<double>)"
p = np.array([
 [1.0, 2.0],
 [3.0, 4.0]
])
res = np.linalg.inv(p)
out = [[-1.9999999999999996, 0.99999999999999978], [1.4999999999999998, -0.49999999999999994]]
add2queue(test_name, res, out)


test_name = "Linalg.convolve(Vector<int>, Vector<int>)"
a = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
b = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
res = np.convolve(a, b)
out = [0.0, 0.0, 1.0, 4.0, 10.0, 20.0, 35.0, 56.0, 84.0, 120.0, 156.0, 192.0, 228.0, 264.0, 300.0, 336.0, 372.0, 392.0, 395.0, 380.0, 346.0, 292.0, 217.0, 120.0]
add2queue(test_name, res, out)


test_name = "Linalg.clip(Vector<int>, int, int)"
vec = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])
res = np.clip(vec, 5, 40)
out = [5, 5, 5, 5, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40]
add2queue(test_name, res, out)


test_name = "Linalg.clip(Vector<int>, int)"
vec = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])
res = np.clip(vec, -50, 20)
out = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
add2queue(test_name, res, out)


test_name = "Statistics::ptp(Vector<double>)"
vec = np.array([4.0, 9.0, 2.0, 10.0, 6.0, 9.0, 7.0, 12.0])
res = np.ptp(vec)
out = 10
add2queue(test_name, res, out)


test_name = "Statistics::percentile(Vector<double>, double)"
vec = np.array([4.0, 9.0, 2.0, 10.0, 6.0, 9.0, 7.0, 12.0])
res = np.percentile(vec, 5)
out = 2.700000
add2queue(test_name, res, out)


test_name = "Statistics::percentile(Vector<double>, Vector<float>)"
vec = np.array([10.0, 7.0, 4.0, 3.0, 2.0, 1.0])
qs = np.array([5.0, 44.0, 99.0, 100.0, 0.0, 50.0, 69.419998168945312])
res = np.percentile(vec, qs)
out = [1.25, 3.2000000476837158, 9.8499994277954102, 10.0, 1.0, 3.5, 5.4129998683929443]
add2queue(test_name, res, out)


test_name = "Statistics::quantile(Vector<double>, double)"
vec = np.array([4.0, 9.0, 2.0, 10.0, 6.0, 9.0, 7.0, 12.0])
res = np.quantile(vec, 0.05)
out = 2.700000
add2queue(test_name, res, out)


test_name = "Statistics::quantile(Vector<double>, Vector<float>)"
vec = np.array([10.0, 7.0, 4.0, 3.0, 2.0, 1.0])
qs = np.array([0.05000000074505806, 0.43999999761581421, 0.99000000953674316, 0.10000000149011612, 0.0, 0.5, 0.69419997930526733])
res = np.quantile(vec, qs)
out = [1.25, 3.2000000476837158, 9.8499994277954102, 1.5, 1.0, 3.5, 5.4129998683929443]
add2queue(test_name, res, out)


test_name = "Statistics::median(Vector<double>)"
vec = np.array([4.0, 9.0, 2.0, 10.0, 6.0, 9.0, 7.0, 12.0])
res = np.median(vec)
out = 8.000000
add2queue(test_name, res, out)


test_name = "Statistics::average(Vector<int>, Vector<int>)"
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
b = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
res = np.average(a, weights=b)
out = 4
add2queue(test_name, res, out)


test_name = "Statistics::average(Vector<int>)"
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
res = np.average(a, weights=None)
out = 5.500000
add2queue(test_name, res, out)


test_name = "Statistics::mean(Vector<int>)"
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
res = np.mean(a)
out = 5.500000
add2queue(test_name, res, out)


test_name = "Statistics::std(Vector<int>)"
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
res = np.std(a)
out = 2.872281
add2queue(test_name, res, out)


test_name = "Statistics::var(Vector<int>)"
a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
res = np.var(a)
out = 8.250000
add2queue(test_name, res, out)


test_name = "Statistics::cov(Matrix<int>, true)"
a = np.array([
 [0.0, 1.0, 2.0],
 [3.0, 4.0, 5.0],
 [6.0, 7.0, 8.0]
])
res = np.cov(a, rowvar=True)
out = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
add2queue(test_name, res, out)


test_name = "Statistics::cov(Matrix<int>, false)"
a = np.array([
 [0.0, 1.0, 2.0],
 [3.0, 4.0, 5.0],
 [6.0, 7.0, 8.0]
])
res = np.cov(a, rowvar=False)
out = [[9.0, 9.0, 9.0], [9.0, 9.0, 9.0], [9.0, 9.0, 9.0]]
add2queue(test_name, res, out)


test_name = "Statistics::cov(Matrix<int>)"
a = np.array([
 [0.0, 1.0, 2.0],
 [3.0, 4.0, 5.0],
 [6.0, 7.0, 8.0]
])
res = np.cov(a, rowvar=True)
out = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
add2queue(test_name, res, out)


test_name = "Statistics::corrcoef(Matrix<int>, true)"
a = np.array([
 [0.0, 1.0, 2.0],
 [3.0, 4.0, 5.0],
 [6.0, 7.0, 8.0]
])
res = np.corrcoef(a, rowvar=True)
out = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
add2queue(test_name, res, out)


test_name = "Statistics::corrcoef(Matrix<int>, false)"
a = np.array([
 [0.0, 1.0, 2.0],
 [3.0, 4.0, 5.0],
 [6.0, 7.0, 8.0]
])
res = np.corrcoef(a, rowvar=False)
out = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
add2queue(test_name, res, out)


test_name = "Statistics::corrcoef(Matrix<int>)"
a = np.array([
 [0.0, 1.0, 2.0],
 [3.0, 4.0, 5.0],
 [6.0, 7.0, 8.0]
])
res = np.corrcoef(a, rowvar=True)
out = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
add2queue(test_name, res, out)


test_name = "Statistics::histogram(Vector<int>, Vector<int>)"
a = np.array([1.0, 2.0, 1.0])
b = np.array([0.0, 1.0, 2.0, 3.0])
res, _ = np.histogram(a, bins = b)
out = [0.0, 2.0, 1.0]
add2queue(test_name, res, out)


test_name = "Statistics::histogram(Vector<int>)"
a = np.array([1.0, 2.0, 1.0])
res, _ = np.histogram(a)
out = [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
add2queue(test_name, res, out)


test_name = "Statistics::bincount(Vector<int>)"
a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
res = np.bincount(a)
out = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
add2queue(test_name, res, out)


test_name = "Statistics::digitize(Vector<int>, Vector<int>, false)"
x = np.array([0.20000000000000001, 6.4000000000000004, 3.0, 1.6000000000000001])
bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
res = np.digitize(x, bins, false)
out = [1, 4, 3, 2]
add2queue(test_name, res, out)


test_name = "Statistics::digitize(Vector<int>, Vector<int>, true)"
x = np.array([0.20000000000000001, 6.4000000000000004, 3.0, 1.6000000000000001])
bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
res = np.digitize(x, bins, true)
out = [1, 4, 3, 2]
add2queue(test_name, res, out)


print_res()
