import numpy as np
from .tensor_data import (
    to_index,
    index_to_position,
    broadcast_index,
    shape_broadcast,
    MAX_DIMS,
)
from numba import njit, prange


# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


def tensor_map(fn):
    """
    NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

        * Main loop in parallel
        * All indices use numpy buffers
        * When `out` and `in` are stride-aligned, avoid indexing

    Args:
        fn: function mappings floats-to-floats to apply.
        out (array): storage for out tensor.
        out_shape (array): shape for out tensor.
        out_strides (array): strides for out tensor.
        in_storage (array): storage for in tensor.
        in_shape (array): shape for in tensor.
        in_strides (array): strides for in tensor.

    Returns:
        None : Fills in `out`
    """

    def _map(out, out_shape, out_strides, in_storage, in_shape, in_strides):
        # TODO: Implement for Task 3.1.
        # replace np.zeros -> np.empty, 减少赋值的循环；将out_idx和in_idx调整到外面（这里不会出错吗？）

        for i in prange(len(out)):
            out_idx = np.empty(MAX_DIMS, np.int16)
            in_idx = np.empty(MAX_DIMS, np.int16)
            # out_idx = [0] * len(out_shape)
            to_index(i, out_shape, out_idx)
          # in_idx = [0] * len(in_shape)
            broadcast_index(out_idx, out_shape, in_shape, in_idx)
            in_pos = index_to_position(in_idx, in_strides)
            out[i] = fn(in_storage[in_pos])

    return njit(parallel=True)(_map)


def map(fn):
    """
    Higher-order tensor map function ::

      fn_map = map(fn)
      fn_map(a, out)
      out

    Args:
        fn: function from float-to-float to apply.
        a (:class:`Tensor`): tensor to map over
        out (:class:`Tensor`): optional, tensor data to fill in,
               should broadcast with `a`

    Returns:
        :class:`Tensor` : new tensor
    """

    # This line JIT compiles your tensor_map
    f = tensor_map(njit()(fn))

    def ret(a, out=None):
        if out is None:
            out = a.zeros(a.shape)
        f(*out.tuple(), *a.tuple())
        return out

    return ret


def tensor_zip(fn):
    """
    NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.


    Optimizations:

        * Main loop in parallel
        * All indices use numpy buffers
        * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
        fn: function maps two floats to float to apply.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        b_storage (array): storage for `b` tensor.
        b_shape (array): shape for `b` tensor.
        b_strides (array): strides for `b` tensor.

    Returns:
        None : Fills in `out`
    """

    def _zip(
        out,
        out_shape,
        out_strides,
        a_storage,
        a_shape,
        a_strides,
        b_storage,
        b_shape,
        b_strides,
    ):

        # TODO: Implement for Task 3.1.


        for i in prange(len(out)):
            out_idx = np.empty(MAX_DIMS, np.int32)
            a_idx = np.empty(MAX_DIMS, np.int32)
            b_idx = np.empty(MAX_DIMS, np.int32)
            # out_idx = [0] * len(out_shape)
            # a_idx = [0] * len(a_shape)
            # b_idx = [0] * len(b_shape)
            to_index(i, out_shape, out_idx)

            broadcast_index(out_idx, out_shape, a_shape, a_idx)
            broadcast_index(out_idx, out_shape, b_shape, b_idx)
            a_pos = index_to_position(a_idx, a_strides)
            b_pos = index_to_position(b_idx, b_strides)

            out[i] = fn(a_storage[a_pos], b_storage[b_pos])

    return njit(parallel=True)(_zip)


def zip(fn):
    """
    Higher-order tensor zip function.

      fn_zip = zip(fn)
      c = fn_zip(a, b)

    Args:
        fn: function from two floats-to-float to apply
        a (:class:`Tensor`): tensor to zip over
        b (:class:`Tensor`): tensor to zip over

    Returns:
        :class:`Tensor` : new tensor data
    """
    f = tensor_zip(njit()(fn))

    def ret(a, b):
        c_shape = shape_broadcast(a.shape, b.shape)
        out = a.zeros(c_shape)
        f(*out.tuple(), *a.tuple(), *b.tuple())
        return out

    return ret


def tensor_reduce(fn):
    """
    NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

        * Main loop in parallel
        * All indices use numpy buffers
        * Inner-loop should not call any functions or write non-local variables

    Args:
        fn: reduction function mapping two floats to float.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        reduce_dim (int): dimension to reduce out

    Returns:
        None : Fills in `out`

    """

    def _reduce(out, out_shape, out_strides, a_storage, a_shape, a_strides, reduce_dim):
        # TODO: Implement for Task 3.1.
        for i in prange(len(out)):
            start = out[i]
            out_idx = np.empty(MAX_DIMS, np.int16)
            # out_idx = [0] * len(out_shape)
            to_index(i, out_shape, out_idx)

            # 在out中的index，就是in中被reduce的维度的第一个index；
            a_pos = index_to_position(out_idx, a_strides)

            # reduce_size = a_shape[reduce_dim] - int(start)
            # reduce_array = np.empty(reduce_size, np.int16)
            # for j in prange(int(start), a_shape[reduce_dim]):
            #     reduce_array[j] = a_storage[a_pos + j * a_strides[reduce_dim]]

            # ret_val = reduce_array[0]
            # for j in range(1, reduce_size):
            #     ret_val = fn(ret_val, reduce_array[j])
            ret_val = a_storage[a_pos + int(start) * a_strides[reduce_dim]]
            for j in range(int(start) + 1, a_shape[reduce_dim]):
                ret_val = fn(ret_val, a_storage[a_pos + j * a_strides[reduce_dim]])

            out[i] = ret_val

    return njit(parallel=True)(_reduce)


def reduce(fn, start=0.0):
    """
    Higher-order tensor reduce function. ::

      fn_reduce = reduce(fn)
      out = fn_reduce(a, dim)


    Args:
        fn: function from two floats-to-float to apply
        a (:class:`Tensor`): tensor to reduce over
        dim (int): int of dim to reduce

    Returns:
        :class:`Tensor` : new tensor
    """

    f = tensor_reduce(njit()(fn))

    def ret(a, dim):
        out_shape = list(a.shape)
        out_shape[dim] = 1

        # Other values when not sum.
        out = a.zeros(tuple(out_shape))
        out._tensor._storage[:] = start

        f(*out.tuple(), *a.tuple(), dim)
        return out

    return ret


@njit(parallel=True, fastmath=True)
def tensor_matrix_multiply(
    out,
    out_shape,
    out_strides,
    a_storage,
    a_shape,
    a_strides,
    b_storage,
    b_shape,
    b_strides,
):
    """
    NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as ::

        assert a_shape[-1] == b_shape[-2]

    Optimizations:

        * Outer loop in parallel
        * No index buffers or function calls
        * Inner loop should have no global writes, 1 multiply.


    Args:
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        b_storage (array): storage for `b` tensor
        b_shape (array): shape for `b` tensor
        b_strides (array): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """
    # 前行后列，即为第一个矩阵的在列维度的长度；
    n_iters = a_shape[-1]
    for i in prange(len(out)):
        out_idx = np.empty(MAX_DIMS, np.int32)
        to_index(i, out_shape, out_idx)
        a_idx = np.copy(out_idx)
        b_idx = np.copy(out_idx)
        # Batch Dims
        if a_shape[0] == 1:
            a_idx[0] = 0
        if b_shape[0] == 1:
            b_idx[0] = 0

        cur_res = 0
        for j in prange(n_iters):
            a_idx[len(out_shape) - 1] = j
            b_idx[len(out_shape) - 2] = j

            a_pos = index_to_position(a_idx, a_strides)
            b_pos = index_to_position(b_idx, b_strides)
            cur_res += a_storage[a_pos] * b_storage[b_pos]
        out[i] = cur_res


def matrix_multiply(a, b):
    """
    Batched tensor matrix multiply ::

        for n:
          for i:
            for j:
              for k:
                out[n, i, j] += a[n, i, k] * b[n, k, j]

    Where n indicates an optional broadcasted batched dimension.

    Should work for tensor shapes of 3 dims ::

        assert a.shape[-1] == b.shape[-2]

    Args:
        a (:class:`Tensor`): tensor data a
        b (:class:`Tensor`): tensor data b

    Returns:
        :class:`Tensor` : new tensor data
    """

    # Make these always be a 3 dimensional multiply
    both_2d = 0
    if len(a.shape) == 2:
        a = a.contiguous().view(1, a.shape[0], a.shape[1])
        both_2d += 1
    if len(b.shape) == 2:
        b = b.contiguous().view(1, b.shape[0], b.shape[1])
        both_2d += 1
    both_2d = both_2d == 2

    ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
    ls.append(a.shape[-2])
    ls.append(b.shape[-1])
    assert a.shape[-1] == b.shape[-2]
    out = a.zeros(tuple(ls))

    tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

    # Undo 3d if we added it.
    if both_2d:
        out = out.view(out.shape[1], out.shape[2])
    return out


class FastOps:
    map = map
    zip = zip
    reduce = reduce
    matrix_multiply = matrix_multiply
