"""Opening a segmentation volume for Meshify, through worldarray."""

from mesh_n_bone.util.image_data_interface import open_volume
from mesh_n_bone.util.volume_geometry import VolumeGeometry


def open_dataset(path):
    """Open the volume at *path* and return its :class:`VolumeGeometry`.

    *path* is an array, or a group (an OME-NGFF multiscales group, a
    precomputed volume), which opens at its first level. Any format and
    location worldarray reads is accepted: zarr v2/v3, N5, precomputed;
    local, ``s3://``, ``gs://``, ``http(s)://``. The geometry is z, y, x
    whatever order the volume is stored in.
    """
    return VolumeGeometry(open_volume(path))
