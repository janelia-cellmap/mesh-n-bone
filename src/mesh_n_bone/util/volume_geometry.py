"""Where a volume's voxels sit in the world, for Meshify's block math.

Meshify works in the voxel indices of the level it reads (z, y, x) and
maps them to world positions with the level's float voxel size and
translation, as worldarray resolves them. Nothing here holds a
tensorstore handle, so a :class:`VolumeGeometry` travels to workers and
survives pickling.
"""

from dataclasses import dataclass

import numpy as np
from worldarray import index_range

SPATIAL_AXES = ("z", "y", "x")


@dataclass(frozen=True)
class IndexBox:
    """The voxels ``[begin, end)`` of a volume, per axis in z, y, x.

    Either bound may lie outside the volume: a read of such a box is
    filled with background past the volume's edge.
    """

    begin: tuple[int, int, int]
    end: tuple[int, int, int]

    def __post_init__(self):
        begin = tuple(int(v) for v in self.begin)
        end = tuple(int(v) for v in self.end)
        if len(begin) != 3 or len(end) != 3:
            raise ValueError(f"an IndexBox has three axes, got {begin} / {end}")
        if any(e < b for b, e in zip(begin, end)):
            raise ValueError(f"end {end} is before begin {begin}")
        object.__setattr__(self, "begin", begin)
        object.__setattr__(self, "end", end)

    @property
    def shape(self):
        return tuple(e - b for b, e in zip(self.begin, self.end))


class VolumeGeometry:
    """A level's voxel grid in the world: shape, chunks, dtype and the
    float voxel size and translation (voxel 0's centre), all in z, y, x.

    Built from a :class:`worldarray.WorldArray` opened with
    ``axes_order="zyx"``, so a volume stored in another order (an
    x, y, z OME-Zarr) arrives here already in z, y, x. Of a volume with
    non-spatial axes (a channel), only the spatial axes are kept;
    :func:`mesh_n_bone.util.image_data_interface.read_block` reads
    channel 0.
    """

    def __init__(self, volume):
        """
        Parameters
        ----------
        volume : worldarray.WorldArray
            The opened volume, in its stored units, presented z, y, x.
        """
        meta = volume.meta
        spatial = meta.spatial
        if len(spatial) != 3:
            raise ValueError(
                f"{volume.path}: Meshify needs three spatial axes; this array "
                f"has axes {meta.axes} of which {len(spatial)} are spatial"
            )
        names = tuple(meta.axes[i] for i in spatial)
        if meta.axes_named and names != SPATIAL_AXES:
            raise ValueError(
                f"{volume.path}: spatial axes are presented as {names}; Meshify "
                f"needs them as {SPATIAL_AXES} (open the volume with "
                f"axes_order='zyx')"
            )
        self.meta = meta
        self.path = volume.level_path
        self.format = meta.format
        self.dtype = np.dtype(meta.dtype)
        self.ndim = len(meta.shape)
        self.shape = tuple(int(meta.shape[i]) for i in spatial)
        self.chunk_shape = tuple(int(meta.chunk_shape[i]) for i in spatial)
        self.voxel_size = np.array([meta.voxel_size[i] for i in spatial], dtype=float)
        self.translation = np.array([meta.translation[i] for i in spatial], dtype=float)
        units = [meta.units[i] for i in spatial]
        self.units = units[0] or None

    @property
    def corner(self):
        """Voxel 0's corner, ``translation - voxel_size / 2``."""
        return self.translation - self.voxel_size / 2

    @property
    def box(self):
        """Every voxel of the volume."""
        return IndexBox((0, 0, 0), self.shape)

    def index_box(self, begin, end):
        """The voxels whose centres lie in the world box ``[begin, end)``.

        A box aligned to voxel corners and one aligned to voxel centres
        both select whole voxels (worldarray's world-box rule), so a ROI
        written against either convention meshes the voxels it always
        did. The result is not clipped to the volume.
        """
        lo, hi = index_range(begin, end, self.voxel_size, self.translation)
        return IndexBox(lo, hi)

    def world(self, index):
        """The centre of voxel ``index`` (z, y, x), in the volume's units."""
        return self.translation + np.asarray(index, dtype=float) * self.voxel_size

    def __repr__(self):
        return (
            f"VolumeGeometry({self.path!r}, shape={self.shape}, "
            f"voxel_size={self.voxel_size.tolist()}, "
            f"translation={self.translation.tolist()})"
        )
