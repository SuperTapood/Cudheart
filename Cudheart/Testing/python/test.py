import numpy as np
from numpy import ndindex

from util import *

a = np.arange(25).reshape((5, 5)) + 1
b = np.arange(75).reshape((3, 5, 5))
# c = np.repeat(a, repeats=3, axis=0)
#
# # print(np.sum(b, axis=0))
# # print(b[0, :, :])
# # print(np.add.reduce(b, 0))
#
# print(b)
# print(np.rot90(b, k=1))
# print(np.rot90(b).shape)

# a = [4, 3, 5, 7, 6, 8]
# a = np.array(a)
# indices = [0, 1, 4]
# # indices = [[0, 1], [2, 2]]
# indices = np.array(indices)
# print(indices[1:])
# print(np.take(a, indices, axis=1))
# print([a for a in np.ndindex(1, 5, 5)])

# axis = 0
# out = np.empty_like(indices)
# Ni, Nk = a.shape[:axis], a.shape[axis+1:]
# Nj = indices.shape
# for ii in ndindex(Ni):
#     for jj in ndindex(Nj):
#         for kk in ndindex(Nk):
#             print("start:")
#             print(f"{ii} + {jj} + {kk} = {ii + jj + kk}")
#             print(f"{ii} + {(indices[jj],)} + {kk} = {ii + (indices[jj],) + kk}")
#             out[ii + jj + kk] = a[ii + (indices[jj],) + kk]
#
# print(out)

# print(np.take(b, indices=[0, 2, 2], axis=0))

# import numpy as np
#
#
# def my_concatenate(arrays, axis=0):
#     # Determine the shape of the resulting array
#     shape = list(arrays[0].shape)
#     shape[axis] = sum(arr.shape[axis] for arr in arrays)
#
#     # Create an empty array with the determined shape
#     result = np.empty(shape, dtype=arrays[0].dtype)
#
#     # Fill the result array with values from the input arrays
#     index = 0
#     for arr in arrays:
#         size = arr.shape[axis]
#         for i in range(size):
#             idx = [slice(None)] * result.ndim
#             idx[axis] = index + i
#             print(idx)
#             print(np.take(arr, i, axis=axis))
#             result[tuple(idx)] = np.take(arr, i, axis=axis)
#         index += size
#
#     return result


# print(np.concatenate((a, b)))
# print(np.concatenate((a, b)).shape)
# print(my_concatenate((a, b)))

# assert (my_concatenate((a, b)) == np.concatenate((a, b))).all()

# print(np.tile(a, 3))
# print(b[(0, 1, None)])
# np.broadcast(a, np.empty(shape=(3, 15, 15)))
print(a)
print(np.cumprod(a, axis=0))
