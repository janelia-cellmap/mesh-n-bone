"""Volume reads through worldarray, in z, y, x voxel indices.

worldarray opens every format mesh-n-bone reads (zarr v2/v3, N5,
neuroglancer precomputed; local or remote), resolves where the array
sits, and reads index boxes with padding and retries. This module keeps
the two things mesh-n-bone needs on top of it.
"""

import worldarray as wa

# Meshify's block math is z, y, x by position, so every volume is opened
# in that order whatever order it is stored in (an x, y, z OME-Zarr, as
# TensorSwitch writes). worldarray moves the axes named z, y and x to the
# end in that order and keeps other axes (a channel) in front.
AXES_ORDER = "zyx"


def open_volume(path, context=None):
    """Open *path* as a :class:`worldarray.WorldArray`, z, y, x, in its
    stored units.

    A group opens at its first level (an OME-NGFF multiscales group's
    first dataset, a precomputed volume's first scale), as mesh-n-bone
    always has; an array, or a precomputed path that names a scale
    (``.../seg/8.0x8.0x8.0``), opens as itself. The result's
    ``level_path`` reopens exactly that level. Each open gets its own
    capped tensorstore context unless *context* is given (see
    :func:`worldarray.capped_context`), so dozens of worker processes on
    one host stay at a few threads each.
    """
    if context is None:
        context = wa.capped_context()
    return wa.open(
        path, level="first", units=None, context=context, axes_order=AXES_ORDER,
    )


def read_block(volume, begin, end, fill_value=0):
    """Read the voxels ``[begin, end)`` of *volume*, z, y, x indices.

    Of a multi-channel volume, channel 0 is read, as mesh-n-bone always
    has; the part of the box outside the volume is filled with
    *fill_value*.
    """
    lead = len(volume.shape) - 3
    begin = [0] * lead + [int(v) for v in begin]
    end = [1] * lead + [int(v) for v in end]
    block = volume.read_index(begin, end, padding=fill_value)
    return block.reshape(block.shape[lead:])
