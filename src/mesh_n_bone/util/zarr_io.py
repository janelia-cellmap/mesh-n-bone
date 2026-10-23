"""Opening a segmentation volume for Meshify, through worldarray."""

from mesh_n_bone.util.cellmap_array import CellMapArray
from mesh_n_bone.util.image_data_interface import open_volume


def open_dataset(path):
    """Open the volume at *path* and return a :class:`CellMapArray`.

    *path* is an array, or a group (an OME-NGFF multiscales group, a
    precomputed volume), which opens at its first level. Any format and
    location worldarray reads is accepted: zarr v2/v3, N5, precomputed;
    local, ``s3://``, ``gs://``, ``http(s)://``.
    """
    volume = open_volume(path)
    return CellMapArray(volume, dataset_path=volume.level_path)
