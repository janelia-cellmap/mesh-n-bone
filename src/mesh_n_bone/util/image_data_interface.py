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
    _is_remote_url,
    _path_join,
    _read_json_file,
    _strip_precomputed_prefix,
    kvstore_for_path,
)
from mesh_n_bone.util.precomputed_io import (
    is_precomputed_path,
    open_precomputed_tensorstore,
)

logger = logging.getLogger(__name__)


def _capped_tensorstore_context_spec():
    """Tensorstore Context spec that limits internal thread pools.

    Tensorstore defaults every concurrency resource (HTTP fetch pools,
    data-copy / decompression pool, file I/O pool) to a high number,
    typically N_CPU. In a dask-worker process — where dask itself
    already runs us in a 1-thread-per-worker model — those default
    pools combine with the dozens of worker processes stacked on a
    single LSF host to produce thousands of runnable threads and
    nodes-over-100%-load complaints from cluster admins. Cap each
    pool to a small fixed limit so per-process thread count is bounded.
    """
    return {
        # In-memory copying + decompression (e.g. gzip/blosc for zarr/n5).
        "data_copy_concurrency": {"limit": 1},
        # Local file I/O (irrelevant for remote sources but caps anyway).
        "file_io_concurrency": {"limit": 1},
        # GCS / S3 / generic HTTP fetch pools — allow 2 so a fetch can
        # be in flight while the previous chunk is being decompressed.
        "gcs_request_concurrency": {"limit": 2},
        "s3_request_concurrency": {"limit": 2},
        "http_request_concurrency": {"limit": 2},
    }


def _detect_zarr_driver(dataset_path):
    """Detect the tensorstore driver for *dataset_path* by content probing.

    Tries marker files in order: precomputed ``info``, zarr v3
    ``zarr.json``, zarr v2 ``.zarray``, N5 ``attributes.json``. The
    optional ``precomputed://`` URL prefix is stripped before probing
    (it's accepted for backward compatibility but no longer required;
    the format is inferred from content, like neuroglancer).

    Falls back to a ``.n5`` / ``.zarr`` filename heuristic for the
    rare case where no marker file is reachable (e.g. when running
    against private buckets with no anonymous access).

    Parameters
    ----------
    dataset_path : str
        Local filesystem path or remote URL (``gs://``, ``s3://``,
        ``http(s)://``, optionally with a ``precomputed://`` prefix).

    Returns
    -------
    str
        ``"zarr"``, ``"zarr3"``, ``"n5"``, or ``"neuroglancer_precomputed"``.
    """
    explicit_precomputed = is_precomputed_path(dataset_path)
    if explicit_precomputed:
        return "neuroglancer_precomputed"

    canonical = _strip_precomputed_prefix(dataset_path)

    # Precomputed: ``info`` lives at the dataset root. Try the path
    # itself first, then its parent (covers paths like
    # ``.../segmentation/8.0x8.0x8.0`` where the trailing segment is a
    # scale key rather than a real subdirectory).
    if _read_json_file(_path_join(canonical, "info")) is not None:
        return "neuroglancer_precomputed"
    from mesh_n_bone.util.zarr_io import _path_dirname
    parent = _path_dirname(canonical)
    if parent and parent != canonical:
        if _read_json_file(_path_join(parent, "info")) is not None:
            return "neuroglancer_precomputed"

    if _read_json_file(_path_join(canonical, "zarr.json")) is not None:
        return "zarr3"
    if _read_json_file(_path_join(canonical, ".zarray")) is not None:
        return "zarr"
    if _read_json_file(_path_join(canonical, "attributes.json")) is not None:
        return "n5"

    # No marker reachable (likely private bucket without anonymous
    # access). Fall back to filename hints.
    if canonical.rfind(".n5") > canonical.rfind(".zarr"):
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
    if filetype == "neuroglancer_precomputed":
        if mode != "r":
            raise ValueError("neuroglancer precomputed datasets are read-only")
        return open_precomputed_tensorstore(dataset_path)

    canonical = _strip_precomputed_prefix(dataset_path)
    if _is_remote_url(canonical) and mode != "r":
        raise ValueError("Remote TensorStore datasets are read-only")

    kvstore, kv_path = kvstore_for_path(canonical)
    if kv_path:
        kvstore["path"] = kv_path.rstrip("/") + "/"

    spec = {
        "driver": filetype,
        "kvstore": kvstore,
        "context": _capped_tensorstore_context_spec(),
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
