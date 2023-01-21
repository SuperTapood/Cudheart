from util import *

import warnings

warnings.filterwarnings('ignore', category = DeprecationWarning)

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
res = res.reshape((3, 2))

out = [[0, 1], [2, 3], [4, 5]]
add2queue(test_name, res, out)



test_name = "MatrixOps::arange<int>(int, int)"
res = np.arange(0, 5, 2, dtype=int)
res = res.reshape((1, 3))

out = [[0, 2, 4]]
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
mat = mat.reshape((3, 3))
res = np.diagflat(mat, 1)

out = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 3, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 4, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 5, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 6, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 7, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 8], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
add2queue(test_name, res, out)



test_name = "MatrixOps::diagflat(Vector<int>)"
mat = np.arange(0, 9, 1, dtype=int)
mat = mat.reshape((3, 3))
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

out = [1, 2, 3, 4, 5]
add2queue(test_name, res, out)



print_res()
