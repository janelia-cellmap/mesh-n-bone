"""Dataset opening utilities using direct JSON metadata reading and TensorStore."""

import json
import logging
import os
import posixpath
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

from funlib.geometry import Coordinate

from mesh_n_bone.util.cellmap_array import CellMapArray

logger = logging.getLogger(__name__)


def _is_http_url(path):
    """Return True if *path* is an HTTP(S) URL."""
    return urlparse(str(path)).scheme in {"http", "https"}


def _path_join(base, path):
    """Join local filesystem paths or append URL path components."""
    if not path:
        return base
    if _is_http_url(base):
        parsed = urlparse(base)
        joined_path = posixpath.join(parsed.path.rstrip("/"), str(path).lstrip("/"))
        return urlunparse(parsed._replace(path=joined_path))
    return os.path.join(base, path)


def _path_dirname(path):
    """Return the parent path for local filesystem paths or URLs."""
    if _is_http_url(path):
        parsed = urlparse(path)
        dirname = posixpath.dirname(parsed.path.rstrip("/"))
        return urlunparse(parsed._replace(path=dirname or "/"))
    return os.path.dirname(path)


def _path_basename(path):
    """Return the final path component for local filesystem paths or URLs."""
    if _is_http_url(path):
        return posixpath.basename(urlparse(path).path.rstrip("/"))
    return os.path.basename(path)


class ArrayMetadata:
    """Metadata container replacing zarr.Array for CellMapArray.

    Provides the same interface as zarr.Array for the properties
    that CellMapArray accesses: ``shape``, ``dtype``, ``chunks``,
    and ``attrs``.
    """

    def __init__(
        self, shape, dtype, chunks, attrs, parent_attrs=None, dataset_name=None,
    ):
        self.shape = shape
        self.dtype = dtype
        self.chunks = chunks
        self.attrs = attrs
        self.parent_attrs = parent_attrs
        self.dataset_name = dataset_name


def _read_json_file(path):
    """Read a JSON file from a local path or HTTP(S) URL.

    Parameters
    ----------
    path : str
        Path or URL to the JSON file.

    Returns
    -------
    dict or None
        Parsed JSON, or ``None`` if the file does not exist or is
        not valid JSON.
    """
    try:
        if _is_http_url(path):
            request = Request(path, headers={"Accept": "application/json"})
            with urlopen(request, timeout=10) as f:
                return json.load(f)
        else:
            with open(path) as f:
                return json.load(f)
    except (
        FileNotFoundError,
        OSError,
        HTTPError,
        URLError,
        TimeoutError,
        json.JSONDecodeError,
    ):
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
        return _read_json_file(_path_join(dataset_path, "attributes.json")) or {}
    zarr_json = _read_json_file(_path_join(dataset_path, "zarr.json"))
    if zarr_json is not None:
        return zarr_json.get("attributes", {})
    zattrs = _read_json_file(_path_join(dataset_path, ".zattrs"))
    if zattrs is not None:
        return zattrs
    return _read_json_file(_path_join(dataset_path, "attributes.json")) or {}


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
    zarr_pos = dataset_path.rfind(".zarr")
    n5_pos = dataset_path.rfind(".n5")
    if zarr_pos == -1 and n5_pos == -1:
        return dataset_path, ""

    splitter = ".zarr" if zarr_pos > n5_pos else ".n5"
    # Split on the LAST occurrence so nested containers like
    # ``outer.zarr/inner.zarr/s0`` resolve to ``inner.zarr`` + ``s0``.
    parts = dataset_path.rsplit(splitter, 1)
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
    from mesh_n_bone.util.image_data_interface import (
        _detect_zarr_driver,
        open_ds_tensorstore,
    )

    logger.debug("opening dataset %s in %s", ds_name, filename)
    full_path = _path_join(filename, ds_name) if ds_name else filename

    attrs = _read_attrs(full_path)
    parent_attrs = None
    metadata_dataset_name = None
    selected_dataset_path = None
    if not ds_name:
        selected_dataset_path = _first_multiscales_dataset_path(attrs)
        if selected_dataset_path:
            parent_attrs = attrs
            metadata_dataset_name = selected_dataset_path
            full_path = _path_join(full_path, selected_dataset_path)
            attrs = _read_attrs(full_path)

    try:
        filetype = _detect_zarr_driver(full_path)
        ts_ds = open_ds_tensorstore(full_path, mode=mode, filetype=filetype)
    except Exception as e:
        logger.error("failed to open %s/%s: %s", filename, ds_name, e)
        raise

    shape = tuple(ts_ds.shape)
    dtype = ts_ds.dtype.numpy_dtype
    chunks = tuple(ts_ds.chunk_layout.read_chunk.shape)

    # N5 stores shape in XYZ order, but funlib expects ZYX.  Reverse so
    # ROI computation (shape * voxel_size) and chunk-aligned operations
    # use consistent axis order.
    if filetype == "n5":
        shape = shape[::-1]
        chunks = chunks[::-1]

    if parent_attrs is None:
        parent_dir = _path_dirname(full_path)
        if parent_dir and parent_dir != full_path:
            parent_attrs = _read_attrs(parent_dir)
    if metadata_dataset_name is None:
        metadata_dataset_name = _path_basename(full_path) or None

    data = ArrayMetadata(
        shape, dtype, chunks, attrs,
        parent_attrs=parent_attrs,
        dataset_name=metadata_dataset_name,
    )
    voxel_size, offset = _read_voxel_size_offset(
        data, parent_attrs=parent_attrs, dataset_name=metadata_dataset_name,
    )
    return CellMapArray(data, voxel_size, offset, dataset_path=full_path)


def _read_funlib_voxel_offset(attrs):
    """Pull funlib/N5-style ``voxel_size`` and ``offset`` from a single attrs dict.

    Returns ``(voxel_size, offset)`` as float lists, with each component
    ``None`` when not declared. Combined into one helper so both the
    array attrs and the parent-group attrs can use the same lookup.
    """
    if not attrs:
        return None, None
    voxel_size = None
    if "resolution" in attrs:
        voxel_size = [float(v) for v in attrs["resolution"]]
    elif "voxel_size" in attrs:
        voxel_size = [float(v) for v in attrs["voxel_size"]]
    elif "pixelResolution" in attrs:
        # N5 pixelResolution.dimensions is XYZ; funlib expects ZYX.
        # Prefer transform.scale (already ZYX via transform.axes).
        transform = attrs.get("transform", {})
        if "scale" in transform:
            voxel_size = [float(v) for v in transform["scale"]]
        else:
            dims = list(attrs["pixelResolution"]["dimensions"])
            voxel_size = [float(v) for v in reversed(dims)]

    offset = [float(v) for v in attrs["offset"]] if "offset" in attrs else None
    return voxel_size, offset


def _resolve_voxel_size_offset(attrs, parent_attrs, dataset_name):
    """Resolve voxel_size and offset as float lists from any supported source.

    Checks, in order: (1) the array's own attrs (funlib/N5 keys),
    (2) the parent group's attrs (same funlib/N5 keys — used by some
    N5 multiscales setups that put resolution at the group level),
    (3) OME-Zarr multiscales on the parent group (v0.4 + v0.5,
    arbitrary axis order, root-level coordinateTransformations,
    arbitrary dataset path).

    Either component may come back ``None`` if no source declared it.
    """
    voxel_size, offset = _read_funlib_voxel_offset(attrs)

    if (voxel_size is None or offset is None) and parent_attrs:
        p_voxel, p_offset = _read_funlib_voxel_offset(parent_attrs)
        if voxel_size is None:
            voxel_size = p_voxel
        if offset is None:
            offset = p_offset

    if (voxel_size is None or offset is None) and parent_attrs:
        ome_scale, ome_translation = _extract_ome_scale_translation(
            parent_attrs, dataset_name=dataset_name,
        )
        if voxel_size is None and ome_scale is not None:
            voxel_size = list(ome_scale)
        if offset is None and ome_translation is not None:
            offset = list(ome_translation)

    return voxel_size, offset


def _read_voxel_size_offset(data, parent_attrs=None, dataset_name=None):
    """Read voxel_size and offset and return them as funlib ``Coordinate``s.

    Composes ``_resolve_voxel_size_offset`` and casts to int; logs a
    warning if non-integer voxel sizes are silently rounded so callers
    needing float precision use :func:`read_raw_voxel_size` instead.
    """
    voxel_size, offset = _resolve_voxel_size_offset(
        data.attrs, parent_attrs, dataset_name,
    )

    if voxel_size is not None and any(v != int(v) for v in voxel_size):
        logger.warning(
            "Rounding non-integer voxel_size %s to integers for "
            "Coordinate; use read_raw_voxel_size for float-precision access.",
            voxel_size,
        )

    if voxel_size is None:
        voxel_size = [1] * len(data.shape)
    if offset is None:
        offset = [0] * len(voxel_size)

    return (
        Coordinate(int(round(v)) for v in voxel_size),
        Coordinate(int(round(v)) for v in offset),
    )


def read_raw_voxel_size(ds):
    """Return the float voxel_size, preserving non-integer precision.

    Same source-resolution as :func:`_read_voxel_size_offset` but does
    not round. Falls back to ``ds.voxel_size`` (already a Coordinate)
    when no metadata source declares a value.
    """
    parent_attrs = getattr(ds.data, "parent_attrs", None) or _read_parent_attrs(ds)
    dataset_name = (
        getattr(ds.data, "dataset_name", None)
        or _path_basename(getattr(ds, "_dataset_path", ""))
        or None
    )
    voxel_size, _ = _resolve_voxel_size_offset(
        ds.data.attrs, parent_attrs, dataset_name,
    )
    if voxel_size is not None:
        return tuple(float(v) for v in voxel_size)
    return tuple(float(v) for v in ds.voxel_size)


def _get_multiscales(attrs):
    """Return the OME-Zarr ``multiscales`` list, or ``None``.

    Handles OME-Zarr v0.4 (top-level ``multiscales``) and v0.5
    (``ome.multiscales``). Logs a warning and returns ``None`` if an
    ``ome`` block is present but does not contain ``multiscales``,
    which usually means the spec has moved on to a layout this
    helper does not recognise yet.
    """
    if not attrs:
        return None
    if "multiscales" in attrs:
        return attrs["multiscales"]
    ome = attrs.get("ome")
    if isinstance(ome, dict):
        if "multiscales" in ome:
            return ome["multiscales"]
        logger.warning(
            "OME-Zarr 'ome' attribute block present but no 'multiscales' "
            "found inside it (keys=%s). Falling back to default voxel size; "
            "this likely means a newer OME-Zarr spec layout the helper "
            "needs to be updated for.",
            sorted(ome.keys()),
        )
    return None


def _first_multiscales_dataset_path(attrs):
    """Return the first dataset path from OME-Zarr multiscales metadata."""
    multiscales = _get_multiscales(attrs)
    if not multiscales:
        return None
    ms = multiscales[0] if isinstance(multiscales, list) and multiscales else None
    if not isinstance(ms, dict):
        return None
    datasets = ms.get("datasets") or []
    if not datasets or not isinstance(datasets[0], dict):
        return None
    return datasets[0].get("path")


def _read_transforms(transforms):
    """Compose a coordinateTransformations list in document order.

    OME-Zarr applies transformations left-to-right, so ``[scale,
    translation]`` differs from ``[translation, scale]``. This helper
    accumulates the equivalent affine ``(scale, translation)`` such
    that ``physical = scale * voxel + translation``, regardless of
    order. Unknown types (e.g. ``identity``) are ignored.

    Returns ``(scale, translation)`` as ``list[float] | None``.
    """
    s_acc = None
    t_acc = None
    for entry in transforms or []:
        if not isinstance(entry, dict):
            continue
        ttype = entry.get("type")
        if ttype == "scale" and "scale" in entry:
            s = [float(v) for v in entry["scale"]]
            if s_acc is None:
                s_acc = list(s)
            else:
                s_acc = [s_acc[i] * s[i] for i in range(len(s))]
            if t_acc is not None:
                t_acc = [t_acc[i] * s[i] for i in range(len(s))]
        elif ttype == "translation" and "translation" in entry:
            t = [float(v) for v in entry["translation"]]
            if t_acc is None:
                t_acc = list(t)
            else:
                t_acc = [t_acc[i] + t[i] for i in range(len(t))]
    return s_acc, t_acc


def _compose_transforms(scale_d, trans_d, scale_r, trans_r):
    """Compose dataset-level then root-level transforms.

    OME-Zarr applies coordinateTransformations in document order; the
    multiscales root-level list applies on top of the per-dataset list.

    Going from voxel coords ``v`` to physical coords::

        physical = trans_r + scale_r * (trans_d + scale_d * v)
                 = (trans_r + scale_r * trans_d) + (scale_r * scale_d) * v

    Missing components default to scale=1, translation=0. Returns the
    composed ``(scale, translation)`` as ``list[float] | None``.
    """
    if scale_r is None and trans_r is None:
        return scale_d, trans_d

    n = None
    for vec in (scale_d, trans_d, scale_r, trans_r):
        if vec is not None:
            n = len(vec)
            break
    if n is None:
        return None, None

    s_d = [float(v) for v in scale_d] if scale_d else [1.0] * n
    t_d = [float(v) for v in trans_d] if trans_d else [0.0] * n
    s_r = [float(v) for v in scale_r] if scale_r else [1.0] * n
    t_r = [float(v) for v in trans_r] if trans_r else [0.0] * n

    composed_scale = [s_r[i] * s_d[i] for i in range(n)]
    composed_trans = [t_r[i] + s_r[i] * t_d[i] for i in range(n)]
    return composed_scale, composed_trans


def _spatial_permutation(axes):
    """Return indices that select spatial axes in ZYX order.

    Falls back to ``None`` (caller should treat the array as already
    ZYX) when the axes list is missing, has no per-axis ``type``
    metadata, or doesn't use the canonical x/y/z names.

    This lets OME datasets with ``axes=[t, c, z, y, x]`` (5D) or
    ``axes=[x, y, z]`` (XYZ) be read correctly from a 3D mesh-n-bone
    pipeline.
    """
    if not axes:
        return None

    space_indices = []
    space_names = []
    for i, ax in enumerate(axes):
        if not isinstance(ax, dict):
            continue
        ax_type = ax.get("type", "space")
        if ax_type == "space":
            space_indices.append(i)
            space_names.append(str(ax.get("name", "")).lower())

    if not space_indices:
        return None

    if set(space_names) >= {"x", "y", "z"} and len(space_names) >= 3:
        ordered = []
        for target in ("z", "y", "x"):
            ordered.append(space_indices[space_names.index(target)])
        return ordered

    return space_indices


def _apply_permutation(values, permutation):
    """Reorder *values* by *permutation*; return a tuple of floats."""
    if values is None:
        return None
    if permutation is None:
        return tuple(float(v) for v in values)
    return tuple(float(values[i]) for i in permutation)


def _select_dataset(datasets, dataset_name):
    """Pick the multiscales dataset entry matching *dataset_name*.

    Falls back to the first entry when *dataset_name* is None or when
    no entry has a matching ``path``. Returns ``None`` if there are no
    datasets.
    """
    if not datasets:
        return None
    if dataset_name is not None:
        for d in datasets:
            if isinstance(d, dict) and d.get("path") == dataset_name:
                return d
    return datasets[0]


def _extract_ome_scale_translation(attrs, dataset_name=None):
    """Read scale and translation from OME-Zarr multiscales metadata.

    Robust to:
      * v0.4 top-level ``multiscales`` and v0.5 ``ome.multiscales``,
      * ``s0`` not being the first dataset (matches by ``path``),
      * root-level ``coordinateTransformations`` (composed with the
        per-dataset transforms),
      * non-ZYX axis ordering and 4D/5D axes lists with non-spatial
        dimensions (axes are filtered to ``type=="space"`` and reordered
        to ZYX when named ``x``/``y``/``z``).

    Parameters
    ----------
    attrs : dict
        Group attributes containing multiscales metadata.
    dataset_name : str, optional
        ``path`` of the dataset to match (e.g. ``"s0"``). When omitted,
        the first dataset entry is used.

    Returns
    -------
    tuple
        ``(scale, translation)`` as float tuples in ZYX order, or
        ``(None, None)`` if no multiscales metadata is found. Either
        component may be ``None`` if the corresponding transform was
        not declared.
    """
    multiscales = _get_multiscales(attrs)
    if not multiscales:
        return None, None

    ms = multiscales[0] if isinstance(multiscales, list) and multiscales else None
    if not isinstance(ms, dict):
        return None, None

    chosen = _select_dataset(ms.get("datasets", []), dataset_name)
    if chosen is None:
        return None, None

    scale_d, trans_d = _read_transforms(chosen.get("coordinateTransformations", []))
    scale_r, trans_r = _read_transforms(ms.get("coordinateTransformations", []))
    scale, translation = _compose_transforms(scale_d, trans_d, scale_r, trans_r)

    perm = _spatial_permutation(ms.get("axes", []))
    return _apply_permutation(scale, perm), _apply_permutation(translation, perm)


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

    parent_attrs = getattr(ds.data, "parent_attrs", None)
    if parent_attrs:
        return parent_attrs

    parent = _path_dirname(dataset_path)
    if parent and parent != dataset_path:
        attrs = _read_attrs(parent)
        if attrs:
            return attrs
    return None
