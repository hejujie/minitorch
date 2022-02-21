from numba import cuda
import numba
from .tensor_data import (
    to_index,
    index_to_position,
    TensorData,
    broadcast_index,
    shape_broadcast,
    MAX_DIMS,
)

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

to_index = cuda.jit(device=True)(to_index)
index_to_position = cuda.jit(device=True)(index_to_position)
broadcast_index = cuda.jit(device=True)(broadcast_index)

THREADS_PER_BLOCK = 32


def tensor_map(fn):
    """
    CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
        fn: function mappings floats-to-floats to apply.
        out (array): storage for out tensor.
        out_shape (array): shape for out tensor.
        out_strides (array): strides for out tensor.
        out_size (array): size for out tensor.
        in_storage (array): storage for in tensor.
        in_shape (array): shape for in tensor.
        in_strides (array): strides for in tensor.

    Returns:
        None : Fills in `out`
    """

    def _map(out, out_shape, out_strides, out_size, in_storage, in_shape, in_strides):
        # TODO: Implement for Task 3.3.
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i < len(out):
            out_idx = cuda.local.array(MAX_DIMS, numba.uint32)
            in_idx = cuda.local.array(MAX_DIMS, numba.uint32)
            to_index(i, out_shape, out_idx)
            broadcast_index(out_idx, out_shape, in_shape, in_idx)
            in_pos = index_to_position(in_idx, in_strides)
            out[i] = fn(in_storage[in_pos])

    return cuda.jit()(_map)


def map(fn):
    # CUDA compile your kernel
    f = tensor_map(cuda.jit(device=True)(fn))

    def ret(a, out=None):
        if out is None:
            out = a.zeros(a.shape)

        # Instantiate and run the cuda kernel.
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())
        return out

    return ret


def tensor_zip(fn):
    """
    CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
        fn: function mappings two floats to float to apply.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        out_size (array): size for `out` tensor.
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
        out_size,
        a_storage,
        a_shape,
        a_strides,
        b_storage,
        b_shape,
        b_strides,
    ):
        # TODO: Implement for Task 3.3.
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i < len(out):
            out_idx = cuda.local.array(MAX_DIMS, numba.uint32)
            a_idx = cuda.local.array(MAX_DIMS, numba.uint32)
            b_idx = cuda.local.array(MAX_DIMS, numba.uint32)

            to_index(i, out_shape, out_idx)
            broadcast_index(out_idx, out_shape, a_shape, a_idx)
            broadcast_index(out_idx, out_shape, b_shape, b_idx)

            a_pos = index_to_position(a_idx, a_strides)
            b_pos = index_to_position(b_idx, b_strides)

            out[i] = fn(a_storage[a_pos], b_storage[b_pos])

    return cuda.jit()(_zip)


def zip(fn):
    f = tensor_zip(cuda.jit(device=True)(fn))

    def ret(a, b):
        c_shape = shape_broadcast(a.shape, b.shape)
        out = a.zeros(c_shape)
        threadsperblock = THREADS_PER_BLOCK
        blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
        f[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )
        return out

    return ret


def _sum_practice(out, a, size):
    """
    This is a practice sum kernel to prepare for reduce.

    Given an array of length :math:`n` and out of size :math:`n // blockDIM`
    it should sum up each blockDim values into an out cell.

    [a_1, a_2, ..., a_100]

    |

    [a_1 +...+ a_32, a_32 + ... + a_64, ... ,]

    Note: Each block must do the sum using shared memory!

    Args:
        out (array): storage for `out` tensor.
        a (array): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32
    # TODO: Implement for Task 3.3.

    if cuda.threadIdx.x == 0:
        cur_block_start = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        cur_res = 0
        for b in range(BLOCK_DIM):
            i = cur_block_start + b
            if i < len(a):
                cur_res += a[i]
        out[int(cuda.blockIdx.x)] = cur_res


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a):
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(fn):
    """
    CUDA higher-order tensor reduce function.

    Args:
        fn: reduction function maps two floats to float.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        out_size (array): size for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        reduce_dim (int): dimension to reduce out

    Returns:
        None : Fills in `out`
    """

    def _reduce(
        out,
        out_shape,
        out_strides,
        out_size,
        a_storage,
        a_shape,
        a_strides,
        reduce_dim,
        reduce_value,
    ):
        # 如果使用1024作为迭代的stop，那么如何判定何处使用reduce_value? 返回shape进行判断吗？
        BLOCK_DIM = 32
        # TODO: Implement for Task 3.3.
        # 只有第一个block会进行复杂的计算逻辑，其余都不会通过if的判断条件
        # 如何能利用shared memory做优化？
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i < out_size:
            out_idx = cuda.local.array(MAX_DIMS, numba.uint32)
            to_index(i, out_shape, out_idx)

            a_pos = index_to_position(out_idx, a_strides)
            ret_val = a_storage[a_pos + int(reduce_value) * a_strides[reduce_dim]]
            for j in range(int(reduce_value) + 1, a_shape[reduce_dim]):
                ret_val = fn(ret_val, a_storage[a_pos + j * a_strides[reduce_dim]])
            out[i] = ret_val

    return cuda.jit()(_reduce)


def reduce(fn, start=0.0):
    """
    Higher-order tensor reduce function. ::

      fn_reduce = reduce(fn)
      out = fn_reduce(a, dim)

    Simple version ::

        for j:
            out[1, j] = start
            for i:
                out[1, j] = fn(out[1, j], a[i, j])


    Args:
        fn: function from two floats-to-float to apply
        a (:class:`Tensor`): tensor to reduce over
        dim (int): int of dim to reduce

    Returns:
        :class:`Tensor` : new tensor
    """
    f = tensor_reduce(cuda.jit(device=True)(fn))

    #NOTE 如果a.shape[dim] > 1024怎么办？是否需要递归处理，直到该dim变成1？
    def ret(a, dim):
        out_shape = list(a.shape)
        out_shape[dim] = 1
        out_a = a.zeros(tuple(out_shape))

        threadsperblock = 32
        blockspergrid = out_a.size
        f[blockspergrid, threadsperblock](
            *out_a.tuple(), out_a.size, *a.tuple(), dim, start
        )

        return out_a

    return ret


def _mm_practice(out, a, b, size):
    """
    This is a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

      * All data must be first moved to shared memory.
      * Only read each cell in `a` and `b` once.
      * Only write to global memory once per kernel.

    Compute ::

    for i:
        for j:
             for k:
                 out[i, j] += a[i, k] * b[k, j]

    Args:
        out (array): storage for `out` tensor.
        a (array): storage for `a` tensor.
        b (array): storage for `a` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    # Basic Implementation
    
    out_idx = cuda.local.array(2, numba.uint32)
    out_idx[0], out_idx[1] = cuda.threadIdx.x, cuda.threadIdx.y
    out_pos = index_to_position(out_idx, (size, 1))
    if out_idx[0] > size or out_idx[1] > size:
        return
    
    a_idx = cuda.local.array(2, numba.uint32)
    b_idx = cuda.local.array(2, numba.uint32)
    a_idx[0] = out_idx[0]
    b_idx[1] = out_idx[1]

    cur_res = 0
    for i in range(size):
        a_idx[1] = i
        b_idx[0] = i

        a_pos = index_to_position(a_idx, (size, 1))
        b_pos = index_to_position(b_idx, (size, 1))
        cur_res += a[a_pos] * b[b_pos]
    out[out_pos] = cur_res

jit_mm_practice = cuda.jit()(_mm_practice)


def mm_practice(a, b):

    (size, _) = a.shape
    threadsperblock = (size, size)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


@cuda.jit()
def tensor_matrix_multiply(
    out,
    out_shape,
    out_strides,
    out_size,
    a_storage,
    a_shape,
    a_strides,
    b_storage,
    b_shape,
    b_strides,
):
    """
    CUDA tensor matrix multiply function.

    Requirements:

      * All data must be first moved to shared memory.
      * Only read each cell in `a` and `b` once.
      * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

        assert a_shape[-1] == b_shape[-2]

    Args:
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        out_size (array): size for `out` tensor.
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        b_storage (array): storage for `b` tensor
        b_shape (array): shape for `b` tensor
        b_strides (array): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """
    BLOCK_DIM = 32
    # TODO: Implement for Task 3.4.
    # TODO(@hejujie): All data must be first moved to shared memory.
    # TODO(@hejujie):  Only read each cell in `a` and `b` once.

    out_idx = cuda.local.array(MAX_DIMS, numba.uint32)
    out_idx[0] = cuda.blockIdx.z
    out_idx[1] = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    out_idx[2] = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    out_pos = index_to_position(out_idx, out_strides)

    if out_idx[0] >= out_shape[0] or out_idx[1] >= out_shape[1] or out_idx[2] >= out_shape[2]:
        return

    a_idx = cuda.local.array(MAX_DIMS, numba.uint32)
    b_idx = cuda.local.array(MAX_DIMS, numba.uint32)

    a_idx[len(out_shape) - 2] = out_idx[len(out_shape) - 2]
    b_idx[len(out_shape) - 1] = out_idx[len(out_shape) - 1]
    # Batch Dims
    a_idx[0] = 0 if a_shape[0] == 1 else out_idx[0]
    b_idx[0] = 0 if b_shape[0] == 1 else out_idx[0]

    n_iters = a_shape[-1]
    cur_res = 0
    for i in range(n_iters):
        a_idx[len(out_shape) - 1] = i
        b_idx[len(out_shape) - 2] = i

        a_pos = index_to_position(a_idx, a_strides)
        b_pos = index_to_position(b_idx, b_strides)
        cur_res += a_storage[a_pos] * b_storage[b_pos]
    
    out[out_pos] = cur_res


def matrix_multiply(a, b):
    """
    Tensor matrix multiply

    Should work for any tensor shapes that broadcast in the first n-2 dims and
    have ::

        assert a.shape[-1] == b.shape[-2]

    Args:
        a (:class:`Tensor`): tensor a
        b (:class:`Tensor`): tensor b

    Returns:
        :class:`Tensor` : new tensor
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

    # One block per batch, extra rows, extra col
    blockspergrid = (
        (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
        (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
        out.shape[0],
    )
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

    tensor_matrix_multiply[blockspergrid, threadsperblock](
        *out.tuple(), out.size, *a.tuple(), *b.tuple()
    )

    # Undo 3d if we added it.
    if both_2d:
        out = out.view(out.shape[1], out.shape[2])
    return out


class CudaOps:
    map = map
    zip = zip
    reduce = reduce
    matrix_multiply = matrix_multiply
