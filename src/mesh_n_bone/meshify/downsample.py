# copied from https://github.com/janelia-flyem/neuclease/blob/11c6aacb30d12c51e7f63c92ac6c9b58b9b38f76/neuclease/util/downsample_with_numba.py#L168
import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def downsample_box(box, block_shape):
    assert block_shape.shape[0] == box.shape[1]
    downsampled_box = np.zeros_like(box)
    downsampled_box[0] = box[0] // block_shape
    downsampled_box[1] = (box[1] + block_shape - 1) // block_shape
    return downsampled_box


def make_blockwise_reducer_3d(reducer_func, nopython=True):
    """Returns a function that reduces an array block-by-block."""

    @jit(nopython=nopython)
    def _reduce_blockwise_compiled(data, block_shape, data_box, reduced_box):
        _output_shape = reduced_box[1] - reduced_box[0]
        output_shape = (_output_shape[0], _output_shape[1], _output_shape[2])
        output = np.zeros(output_shape, data.dtype)

        for block_index in np.ndindex(*output_shape):
            block_bounds = np.zeros((2, 3), dtype=np.int32)
            block_bounds[0] = block_index
            block_bounds[1] = 1 + block_bounds[0]
            block_bounds[:] += reduced_box[0]
            block_bounds[:] *= block_shape

            block_bounds[0] = np.maximum(block_bounds[0], data_box[0])
            block_bounds[1] = np.minimum(block_bounds[1], data_box[1])

            z0, y0, x0 = block_bounds[0] - data_box[0]
            z1, y1, x1 = block_bounds[1] - data_box[0]

            block_data = data[z0:z1, y0:y1, x0:x1]

            bi_z, bi_y, bi_x = block_index
            output[bi_z, bi_y, bi_x] = reducer_func(block_data)
        return output

    def reduce_blockwise(data, block_shape, data_box=None):
        assert data.ndim == 3

        if data_box is None:
            data_box = np.array([(0, 0, 0), data.shape])
        else:
            data_box = np.asarray(data_box)

        assert data_box.shape == (2, 3)

        if np.issubdtype(type(block_shape), np.integer):
            block_shape = (block_shape, block_shape, block_shape)

        block_shape = np.array(block_shape)
        assert block_shape.shape == (3,)

        if (block_shape == 1).all():
            return data, data_box.copy()

        reduced_box = downsample_box(data_box, block_shape)
        reduced_output = _reduce_blockwise_compiled(
            data, block_shape, data_box, reduced_box
        )
        return reduced_output, reduced_box

    return reduce_blockwise


@jit(nopython=True, cache=True)
def flat_mode_except_zero(data):
    """Return mode of flattened data, excluding zeros if possible."""
    data = data.copy().reshape(-1)
    data = data[data != 0]
    if data.size == 0:
        return 0
    return _flat_mode(data)


@jit(nopython=True, cache=True)
def flat_mode(data):
    """Return mode of flattened array."""
    data = data.copy().reshape(-1)
    return _flat_mode(data)


@jit(nopython=True, cache=True)
def _flat_mode(data):
    data.sort()
    diff = np.diff(data)
    diff_bool = np.ones((len(diff) + 2,), dtype=np.uint8)
    diff_bool[1:-1] = diff != 0

    diff_nonzero = diff_bool.nonzero()[0]
    run_lengths = diff_nonzero[1:] - diff_nonzero[:-1]
    max_run = np.argmax(run_lengths)
    return data[diff_nonzero[max_run]]


@jit(nopython=True, cache=True)
def flat_binary_mode(data):
    nonzero = 0
    for index in np.ndindex(data.shape):
        z, y, x = index
        if data[z, y, x] != 0:
            nonzero += 1

    if nonzero > data.size // 2:
        return 1
    return 0


downsample_labels_3d = make_blockwise_reducer_3d(flat_mode)
downsample_binary_3d = make_blockwise_reducer_3d(flat_binary_mode)
downsample_labels_3d_suppress_zero = make_blockwise_reducer_3d(flat_mode_except_zero)
downsample_binary_3d_suppress_zero = make_blockwise_reducer_3d(np.any)
