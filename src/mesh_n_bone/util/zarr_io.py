"""Dataset opening utilities using direct JSON metadata reading and TensorStore."""

import json
import logging
import os

from funlib.geometry import Coordinate

from mesh_n_bone.util.cellmap_array import CellMapArray

logger = logging.getLogger(__name__)


class ArrayMetadata:
    """Metadata container replacing zarr.Array for CellMapArray.

    Provides the same interface as zarr.Array for the properties
    that CellMapArray accesses: ``shape``, ``dtype``, ``chunks``,
    and ``attrs``.
    """

    def __init__(self, shape, dtype, chunks, attrs):
        self.shape = shape
        self.dtype = dtype
        self.chunks = chunks
        self.attrs = attrs


def _read_json_file(path):
    """Read a JSON file from a local path.

    Parameters
    ----------
    path : str
        Path to the JSON file.

    Returns
    -------
    dict or None
        Parsed JSON, or ``None`` if the file does not exist or is
        not valid JSON.
    """
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def _is_n5(path):
    """Return True if *path* refers to an N5 container."""
    return path.rfind(".n5") > path.rfind(".zarr")


def _read_attrs(dataset_path):
    """Read custom attributes from the metadata file at *dataset_path*.

    Supports N5 (``attributes.json``), zarr v3 (``zarr.json``),
    and zarr v2 (``.zattrs``).

    Parameters
    ----------
    dataset_path : str
        Path to the dataset directory.

    Returns
    -------
    dict
        Parsed attributes (may be empty).
    """
    if _is_n5(dataset_path):
        return _read_json_file(os.path.join(dataset_path, "attributes.json")) or {}
    zarr_json = _read_json_file(os.path.join(dataset_path, "zarr.json"))
    if zarr_json is not None:
        return zarr_json.get("attributes", {})
    return _read_json_file(os.path.join(dataset_path, ".zattrs")) or {}


def split_dataset_path(dataset_path):
    """Split a dataset path into container path and internal dataset name.

    Parameters
    ----------
    dataset_path : str
        Full path like ``/data/seg.zarr/volumes/labels/s0``.

    Returns
    -------
    tuple[str, str]
        ``(container_path, dataset_name)`` e.g.
        ``("/data/seg.zarr", "volumes/labels/s0")``.
    """
    splitter = (
        ".zarr" if dataset_path.rfind(".zarr") > dataset_path.rfind(".n5") else ".n5"
    )
    parts = dataset_path.split(splitter)
    container = parts[0] + splitter
    dataset_name = parts[1].lstrip("/") if len(parts) > 1 else ""
    return container, dataset_name


def open_dataset(filename, ds_name, mode="r"):
    """Open a zarr/n5 dataset and return a CellMapArray.

    Uses direct JSON file reading for metadata and TensorStore
    for array shape, dtype, and chunk layout.

    Parameters
    ----------
    filename : str
        Path to the zarr/n5 container.
    ds_name : str
        Dataset name within the container.
    mode : str
        Open mode (default ``"r"``).

    Returns
    -------
    CellMapArray
        Array wrapper with physical coordinate metadata.
    """
    from mesh_n_bone.util.image_data_interface import open_ds_tensorstore

    logger.debug("opening dataset %s in %s", ds_name, filename)
    full_path = os.path.join(filename, ds_name) if ds_name else filename

    attrs = _read_attrs(full_path)

    try:
        ts_ds = open_ds_tensorstore(full_path, mode=mode)
    except Exception as e:
        logger.error("failed to open %s/%s: %s", filename, ds_name, e)
        raise

    shape = tuple(ts_ds.shape)
    dtype = ts_ds.dtype.numpy_dtype
    chunks = tuple(ts_ds.chunk_layout.read_chunk.shape)

    data = ArrayMetadata(shape, dtype, chunks, attrs)
    voxel_size, offset = _read_voxel_size_offset(data)
    return CellMapArray(data, voxel_size, offset, dataset_path=full_path)


def _read_voxel_size_offset(data):
    """Read voxel_size and offset from array attributes.

    Checks funlib-style, N5, and OME-Zarr metadata formats.

    Parameters
    ----------
    data : ArrayMetadata
        Metadata container with ``.attrs`` dict.

    Returns
    -------
    tuple[Coordinate, Coordinate]
        ``(voxel_size, offset)``
    """
    attrs = data.attrs

    if "resolution" in attrs:
        voxel_size = Coordinate(int(v) for v in attrs["resolution"])
    elif "voxel_size" in attrs:
        voxel_size = Coordinate(int(v) for v in attrs["voxel_size"])
    elif "pixelResolution" in attrs:
        voxel_size = Coordinate(
            int(v) for v in attrs["pixelResolution"]["dimensions"]
        )
    else:
        voxel_size = Coordinate(1 for _ in data.shape)

    if "offset" in attrs:
        offset = Coordinate(int(v) for v in attrs["offset"])
    else:
        offset = Coordinate(0 for _ in voxel_size)

    return voxel_size, offset


def read_raw_voxel_size(ds):
    """Read the original float voxel_size from dataset attributes.

    Unlike ``_read_voxel_size_offset``, preserves float precision
    (funlib.geometry.Coordinate truncates to int).

    Parameters
    ----------
    ds : CellMapArray
        Dataset wrapper.

    Returns
    -------
    tuple[float, ...]
        True voxel size as floats.
    """
    attrs = ds.data.attrs

    if "voxel_size" in attrs:
        return tuple(float(v) for v in attrs["voxel_size"])

    if "resolution" in attrs:
        return tuple(float(v) for v in attrs["resolution"])

    if "pixelResolution" in attrs:
        return tuple(float(v) for v in attrs["pixelResolution"]["dimensions"])

    # Check OME-Zarr multiscales on parent group
    parent_attrs = _read_parent_attrs(ds)
    if parent_attrs and "multiscales" in parent_attrs:
        return _extract_ome_scale(parent_attrs)

    return tuple(float(v) for v in ds.voxel_size)


def _extract_ome_scale(attrs):
    """Extract voxel size from OME-Zarr multiscales metadata."""
    transforms = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"]
    for t in transforms:
        if t["type"] == "scale":
            return tuple(float(v) for v in t["scale"])
    raise ValueError("No scale transform found in OME-Zarr multiscales metadata")


def _read_parent_attrs(ds):
    """Read attributes from the parent directory of a dataset.

    Parameters
    ----------
    ds : CellMapArray
        Dataset wrapper. Uses the stored ``_dataset_path`` if
        available, otherwise falls back to filesystem path
        navigation.

    Returns
    -------
    dict or None
        Parent directory attributes, or ``None`` if unavailable.
    """
    dataset_path = getattr(ds, "_dataset_path", None)
    if dataset_path is None:
        return None

    parent = os.path.dirname(dataset_path)
    if parent and parent != dataset_path:
        attrs = _read_attrs(parent)
        if attrs:
            return attrs
    return None
