"""Zarr dataset opening utilities, replacing funlib.persistence.open_ds."""

import logging
import os

import zarr
from funlib.geometry import Coordinate

from mesh_n_bone.util.cellmap_array import CellMapArray

logger = logging.getLogger(__name__)


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

    Supports zarr v2, v3, and N5 formats.

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
    logger.debug("opening dataset %s in %s", ds_name, filename)
    full_path = os.path.join(filename, ds_name) if ds_name else filename
    try:
        ds = zarr.open_array(full_path, mode=mode)
    except Exception as e:
        logger.error("failed to open %s/%s: %s", filename, ds_name, e)
        raise

    voxel_size, offset = _read_voxel_size_offset(ds)
    return CellMapArray(ds, voxel_size, offset)


def _read_voxel_size_offset(ds):
    """Read voxel_size and offset from a zarr array's attributes.

    Checks funlib-style, N5, and OME-Zarr metadata formats.

    Parameters
    ----------
    ds : zarr.Array
        Opened zarr array.

    Returns
    -------
    tuple[Coordinate, Coordinate]
        ``(voxel_size, offset)``
    """
    attrs = dict(ds.attrs)

    if "resolution" in attrs:
        voxel_size = Coordinate(int(v) for v in attrs["resolution"])
    elif "voxel_size" in attrs:
        voxel_size = Coordinate(int(v) for v in attrs["voxel_size"])
    elif "pixelResolution" in attrs:
        voxel_size = Coordinate(
            int(v) for v in attrs["pixelResolution"]["dimensions"]
        )
    else:
        voxel_size = Coordinate(1 for _ in ds.shape)

    if "offset" in attrs:
        offset = Coordinate(int(v) for v in attrs["offset"])
    else:
        offset = Coordinate(0 for _ in voxel_size)

    return voxel_size, offset


def read_raw_voxel_size(ds):
    """Read the original float voxel_size from zarr attributes.

    Unlike `_read_voxel_size_offset`, preserves float precision
    (funlib.geometry.Coordinate truncates to int).

    Parameters
    ----------
    ds : CellMapArray or object with ``.data.attrs``
        Dataset wrapper.

    Returns
    -------
    tuple[float, ...]
        True voxel size as floats.
    """
    attrs = dict(ds.data.attrs)

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
    """Try to read attributes from the parent zarr group."""
    try:
        store = ds.data.store
        store_root = getattr(store, "root", None)
        if store_root:
            store_root = str(store_root)
            if store_root.startswith("file://"):
                store_root = store_root[len("file://"):]

        array_name = getattr(ds.data, "name", None) or getattr(ds.data, "path", "")
        array_name = array_name.strip("/")

        if array_name and store_root:
            parent_path = "/".join(array_name.split("/")[:-1])
            if parent_path:
                parent = zarr.open_group(store, mode="r", path=parent_path)
                return dict(parent.attrs)
        elif store_root:
            parent_dir = os.path.dirname(store_root)
            if os.path.isdir(parent_dir):
                parent = zarr.open_group(parent_dir, mode="r")
                return dict(parent.attrs)
    except Exception:
        pass
    return None
