"""TensorStore-based data reading utilities for segmentation volumes."""

import logging
import os
import time
import random

import numpy as np
import tensorstore as ts
from funlib.geometry import Coordinate, Roi

from mesh_n_bone.util.zarr_io import (
    _is_http_url,
    _path_join,
    _read_json_file,
)

logger = logging.getLogger(__name__)


def _detect_zarr_driver(dataset_path):
    """Detect whether a dataset is zarr v2, v3, or N5 format.

    Parameters
    ----------
    dataset_path : str
        Full filesystem path or HTTP(S) URL to the dataset.

    Returns
    -------
    str
        ``"zarr"`` for v2, ``"zarr3"`` for v3, ``"n5"`` for N5.
    """
    if dataset_path.rfind(".n5") > dataset_path.rfind(".zarr"):
        return "n5"
    if _read_json_file(_path_join(dataset_path, "zarr.json")) is not None:
        return "zarr3"
    if _read_json_file(_path_join(dataset_path, ".zarray")) is not None:
        return "zarr"
    if _read_json_file(_path_join(dataset_path, "attributes.json")) is not None:
        return "n5"
    return "zarr"


def open_ds_tensorstore(dataset_path, mode="r", filetype=None):
    """Open a zarr/n5 dataset with TensorStore.

    Parameters
    ----------
    dataset_path : str
        Full filesystem path to the dataset (container + internal path).
    mode : str
        ``"r"`` for read-only, ``"w"`` for write.

    Returns
    -------
    tensorstore.TensorStore
        Opened dataset handle.
    """
    filetype = filetype or _detect_zarr_driver(dataset_path)
    if _is_http_url(dataset_path):
        if mode != "r":
            raise ValueError("HTTP(S) TensorStore datasets are read-only")
        kvstore = {
            "driver": "http",
            "base_url": dataset_path,
        }
    else:
        kvstore = {
            "driver": "file",
            "path": os.path.abspath(dataset_path),
        }

    spec = {
        "driver": filetype,
        "kvstore": kvstore,
    }
    if mode == "r":
        dataset_future = ts.open(spec, read=True, write=False)
    else:
        dataset_future = ts.open(spec, read=False, write=True)
    return dataset_future.result()


def read_with_retries(dataset, valid_slices, max_retries=10, timeout=5):
    """Read from TensorStore with exponential backoff on timeout.

    Parameters
    ----------
    dataset : tensorstore.TensorStore
        Opened dataset handle.
    valid_slices : tuple of slice
        Slices to read.
    max_retries : int
        Maximum retry attempts.
    timeout : float
        Base timeout in seconds per attempt.

    Returns
    -------
    numpy.ndarray
        Data read from the dataset.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return dataset[valid_slices].read().result(timeout=timeout * attempt)
        except TimeoutError as e:
            logger.error(
                f"[Attempt {attempt}/{max_retries}] "
                f"Timeout reading slices={valid_slices!r}: {e}"
            )
            if attempt == max_retries:
                raise
            delay = 1.0 * (1.3 ** (attempt - 1))
            jitter = random.uniform(0, 1.0)
            time.sleep(delay + jitter)


def to_ndarray_tensorstore(dataset, roi, voxel_size, offset, swap_axes=False,
                           fill_value=0, max_retries=10, timeout=5):
    """Read a region of a TensorStore dataset as a numpy array.

    Handles padding when the ROI extends beyond dataset bounds.

    Parameters
    ----------
    dataset : tensorstore.TensorStore
        Opened dataset handle.
    roi : funlib.geometry.Roi or None
        Region of interest in physical coordinates.  ``None`` reads
        the entire dataset.
    voxel_size : Coordinate
        Native voxel size of the dataset.
    offset : Coordinate
        Spatial offset of the dataset origin.
    swap_axes : bool
        If ``True``, reverse axis order (for N5 format).
    fill_value : int or float
        Padding value for out-of-bounds regions.
    max_retries : int
        Maximum retry attempts for reading.
    timeout : float
        Base timeout in seconds per read attempt.

    Returns
    -------
    numpy.ndarray
        Data array for the requested region.
    """
    if swap_axes:
        if roi:
            roi = Roi(roi.begin[::-1], roi.shape[::-1])
        if offset:
            offset = Coordinate(offset[::-1])
        # Reverse voxel_size too — the division below uses it on the
        # already-reversed ROI, so they must be in the same axis order.
        voxel_size = Coordinate(reversed(tuple(voxel_size)))

    domain = dataset.domain
    if len(domain) > 3:
        channel_offset = 1
        domain = domain[1:]
    else:
        channel_offset = 0

    if roi is None:
        return dataset.read().result()

    if offset is None:
        offset = Coordinate(np.zeros(roi.dims, dtype=int))

    # Subtract offset first so snap_to_grid aligns to the dataset's
    # voxel grid (offset + k*voxel_size), not multiples of voxel_size.
    # Without this, datasets with non-zero offset (e.g., 60nm) get
    # misaligned reads where adjacent blocks read different physical
    # voxels for what should be overlap.
    roi -= offset
    roi = roi.snap_to_grid(voxel_size)
    roi /= voxel_size

    roi_slices = roi.to_slices()

    valid_slices = tuple(
        slice(max(s.start, inclusive_min), min(s.stop, exclusive_max))
        for s, inclusive_min, exclusive_max in zip(
            roi_slices, domain.inclusive_min, domain.exclusive_max
        )
    )

    no_overlap = any(vs.start >= vs.stop for vs in valid_slices)

    pad_width = [
        [valid_slice.start - s.start, s.stop - valid_slice.stop]
        for s, valid_slice in zip(roi_slices, valid_slices)
    ]

    if channel_offset > 0:
        pad_width = [[0, 0]] + pad_width
        channels = slice(dataset.domain[0].inclusive_min, dataset.domain[0].exclusive_max)
        valid_slices = (channels,) + valid_slices

    if no_overlap:
        output_shape = (
            ([dataset.shape[0]] if channel_offset > 0 else [])
            + [s.stop - s.start for s in roi_slices]
        )
        return np.full(output_shape, fill_value, dtype=dataset.dtype.numpy_dtype)

    data = read_with_retries(dataset, valid_slices, max_retries, timeout)

    if np.any(np.array(pad_width)):
        data = np.pad(
            data,
            pad_width=pad_width,
            mode="constant",
            constant_values=fill_value,
        )

    if swap_axes:
        data = np.swapaxes(data, 0 + channel_offset, 2 + channel_offset)

    return data
