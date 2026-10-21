"""Volume reads through worldarray, in Meshify's integer block frame.

worldarray opens every format mesh-n-bone reads (zarr v2/v3, N5,
neuroglancer precomputed; local or remote), resolves where the array
sits, and reads index boxes with padding and retries. This module keeps
the three things mesh-n-bone needs on top of it.
"""

import worldarray as wa

# worldarray's format names -> the tensorstore driver names mesh-n-bone keys on
_DRIVER = {
    "zarr2": "zarr",
    "zarr3": "zarr3",
    "n5": "n5",
    "precomputed": "neuroglancer_precomputed",
}


def detect_driver(path):
    """Return the tensorstore driver name for the volume at *path*."""
    return _DRIVER[wa.detect(path).format]


def open_volume(path, context=None):
    """Open *path* as a :class:`worldarray.WorldArray` in its stored units.

    A group opens at its first level (an OME-NGFF multiscales group's
    first dataset, a precomputed volume's first scale), as mesh-n-bone
    always has; the result's ``level_path`` reopens exactly that level.
    Each open gets its own capped tensorstore context unless *context*
    is given (see :func:`worldarray.capped_context`), so dozens of
    worker processes on one host stay at a few threads each.
    """
    if context is None:
        context = wa.capped_context()
    return wa.open(path, level="first", units=None, context=context)


def read_block(volume, roi, voxel_size, offset, fill_value=0):
    """Read funlib *roi* from *volume* in Meshify's block frame.

    Voxel ``i`` sits at ``offset + i * voxel_size``, where *offset* is
    the integer-rounded translation (voxel 0's centre), and *roi* is
    snapped outward to that grid. Reading by index rather than by world
    box keeps neighbouring blocks exact when Meshify works on rounded
    values of a non-integer translation or voxel size. Leading channel
    axes are read whole; the part of *roi* outside the volume is filled
    with *fill_value*.
    """
    roi = (roi - offset).snap_to_grid(voxel_size) / voxel_size
    lead = len(volume.shape) - roi.dims
    begin = [0] * lead + list(roi.begin)
    end = list(volume.shape[:lead]) + list(roi.end)
    return volume.read_index(begin, end, padding=fill_value)
