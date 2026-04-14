"""Lightweight wrapper around array metadata with physical coordinate support.

Replaces funlib.persistence.Array for read-only use cases.
"""

import numpy as np
from funlib.geometry import Coordinate, Roi


class CellMapArray:
    """Wrapper providing ROI-based indexing over array metadata.

    Converts between physical coordinates and voxel indices,
    replacing funlib.persistence.Array for read-only workflows.
    """

    def __init__(self, data, voxel_size, offset, dataset_path=None):
        """
        Parameters
        ----------
        data : ArrayMetadata
            Metadata container with shape, dtype, chunks, and attrs.
        voxel_size : Coordinate or tuple
            Voxel dimensions in physical units.
        offset : Coordinate or tuple
            Array origin in physical units.
        dataset_path : str or None
            Filesystem path to the dataset (used for parent attr lookup).
        """
        self.data = data
        self._dataset_path = dataset_path
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
