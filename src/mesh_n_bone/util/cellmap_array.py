"""A volume's geometry in the integer funlib frame Meshify does block math in."""

import logging

import numpy as np
from funlib.geometry import Coordinate, Roi

logger = logging.getLogger(__name__)


class CellMapArray:
    """Integer ``voxel_size`` / ``roi`` for funlib block arithmetic, plus
    the volume's true float geometry.

    Built from a :class:`worldarray.WorldArray`. ``roi.offset`` is the
    rounded translation (voxel 0's centre), so voxel ``i`` sits at
    ``roi.offset + i * voxel_size`` in the integer frame; ``true_offset``
    and ``true_voxel_size`` keep the floats Meshify maps final mesh
    vertices back onto. Holds metadata only, no tensorstore handle.
    """

    def __init__(self, volume, dataset_path):
        """
        Parameters
        ----------
        volume : worldarray.WorldArray
            The opened volume, in its stored units.
        dataset_path : str
            Path of the array (the level, when a group was opened).
        """
        self.data = volume.meta
        self._dataset_path = dataset_path
        self.true_voxel_size = np.array(volume.voxel_size, dtype=float)
        self.true_offset = np.array(volume.translation, dtype=float)
        if np.any(self.true_voxel_size != np.round(self.true_voxel_size)):
            logger.warning(
                "Rounding non-integer voxel_size %s to integers for funlib "
                "block arithmetic; mesh vertices are mapped back to the true "
                "values.",
                self.true_voxel_size.tolist(),
            )
        self.voxel_size = Coordinate(int(round(v)) for v in self.true_voxel_size)
        offset = Coordinate(int(round(v)) for v in self.true_offset)
        shape_spatial = tuple(volume.shape[i] for i in volume.spatial)
        self.roi = Roi(offset, Coordinate(shape_spatial) * self.voxel_size)
        units = [self.data.units[i] for i in self.data.spatial]
        self.units = units[0] or None

    @property
    def format(self):
        """``"zarr2"``, ``"zarr3"``, ``"n5"`` or ``"precomputed"``."""
        return self.data.format

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def chunk_shape(self):
        return Coordinate(self.data.chunk_shape[i] for i in self.data.spatial)

    @property
    def shape(self):
        return self.data.shape
