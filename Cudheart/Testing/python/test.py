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
# print(a)
# print(np.cumprod(a, axis=0))

def transpose(a, axes=None):
    if axes is None:
        axes = list(range(len(a.shape)))[::-1]
    else:
        axes = list(axes)
    result = []
    print(axes)
    print(a.shape)
    print([a.shape[axis] for axis in axes])
    for index in np.ndindex(*[a.shape[axis] for axis in axes]):
        v = [index[axis] for axis in axes]
        t = tuple(v)
        print(v, a[tuple(v)])
        result.append(a[t])
    return np.array(result).reshape([a.shape[axis] for axis in axes])


def dot(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    print(a)
    print(b)
    if a.ndim == 0 or b.ndim == 0:
        return a * b
    if a.ndim == 1 and b.ndim == 1:
        return sum([x * y for x, y in zip(a, b)])
    if a.ndim >= 2 and b.ndim >= 2:
        result = np.zeros([a.shape[0]] + list(b.shape[1:]))
        for index in np.ndindex(*result.shape):
            i = index[0]
            j = index[1:]
            for k in range(a.shape[1]):
                print(f"{index} => {[i, k]} ({a[i, k]}), {(k,) + j} ({b[(k,) + j]})")
            result[index] = sum([a[i, k] * b[(k,) + j] for k in range(a.shape[1])])
            # print([a[i, k] * b[(k,) + j] for k in range(a.shape[1])])
        return result
    raise ValueError("Invalid input shapes")


def tensordot(a, b, axes=2):
    a = np.asarray(a)
    b = np.asarray(b)
    if type(axes) == int:
        na = a.ndim
        nb = b.ndim
        a_axes = list(range(na - axes, na))
        b_axes = list(range(0, axes))
    else:
        a_axes, b_axes = axes
    a_axes = [a_axes]
    b_axes = [b_axes]
    try:
        iter(a_axes)
    except TypeError:
        a_axes = [a_axes]
    try:
        iter(b_axes)
    except TypeError:
        b_axes = [b_axes]
    a_shape = [a.shape[axis] for axis in a_axes]
    b_shape = [b.shape[axis] for axis in b_axes]
    if a_shape != b_shape:
        raise ValueError("shape-mismatch for sum")
    olda = [i for i in range(a.ndim) if i not in a_axes]
    oldb = [i for i in range(b.ndim) if i not in b_axes]
    newaxes_a = olda + a_axes
    newaxes_b = b_axes + oldb
    N2 = 1
    for dim in a_shape:
        N2 *= dim
    N1 = 1
    for dim in [a.shape[i] for i in olda]:
        N1 *= dim
    N3 = 1
    for dim in [b.shape[i] for i in oldb]:
        N3 *= dim
   #  print(olda)
    # print(oldb)
    # print(N1)
    # print(N2)
   #  print(N3)
    at = np.transpose(a, newaxes_a).reshape((N1, N2))
    bt = transpose(b, newaxes_b).reshape((N2, N3))
    print(at.shape)
    print(bt.shape)
    print(b)
    print(bt)
    res = dot(at, bt)
    # print(res)
    return res.reshape([a.shape[i] for i in olda] + [b.shape[i] for i in oldb])

a = np.arange(8).reshape(2, 2, 2)
b = np.arange(4).reshape(2, 2)

print(np.inner(a, b))

# print(a[0, 0])
