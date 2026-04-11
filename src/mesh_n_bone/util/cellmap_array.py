"""Lightweight wrapper around a zarr.Array with physical coordinate metadata.

Replaces funlib.persistence.Array for read-only use cases.
"""

import numpy as np
from funlib.geometry import Coordinate, Roi


class CellMapArray:
    """Wrapper around a zarr.Array providing ROI-based indexing.

    Converts between physical coordinates and voxel indices,
    replacing funlib.persistence.Array for read-only workflows.
    """

    def __init__(self, data, voxel_size, offset):
        """
        Parameters
        ----------
        data : zarr.Array
            Underlying zarr array.
        voxel_size : Coordinate or tuple
            Voxel dimensions in physical units.
        offset : Coordinate or tuple
            Array origin in physical units.
        """
        self.data = data
        self.voxel_size = (
            voxel_size if isinstance(voxel_size, Coordinate) else Coordinate(voxel_size)
        )
        n_spatial = len(self.voxel_size)
        shape_spatial = self.data.shape[-n_spatial:]
        offset = offset if isinstance(offset, Coordinate) else Coordinate(offset)
        self.roi = Roi(offset, Coordinate(shape_spatial) * self.voxel_size)

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def chunk_shape(self):
        return Coordinate(self.data.chunks)

    @property
    def shape(self):
        return self.data.shape

    def _to_slices(self, roi):
        """Convert a physical ROI to voxel slices."""
        voxel_roi = (roi - self.roi.offset) / self.voxel_size
        return voxel_roi.to_slices()

    def to_ndarray(self, roi, fill_value=0):
        """Read data for a physical ROI, padding with fill_value if needed."""
        shape = roi.shape / self.voxel_size
        data = np.zeros(shape, dtype=self.dtype)
        if fill_value != 0:
            data[:] = fill_value

        shared_roi = self.roi.intersect(roi)
        if not shared_roi.empty:
            target_slices = ((shared_roi - roi.offset) / self.voxel_size).to_slices()
            source_slices = self._to_slices(shared_roi)
            data[target_slices] = self.data[source_slices]

        return data
