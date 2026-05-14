"""Neuroglancer precomputed volume readers.

Supports input paths of the form ``precomputed://<URL>`` where ``<URL>``
may be ``gs://bucket/path``, ``s3://bucket/path``, ``http(s)://...``, or
a local filesystem path. A trailing path segment may optionally name a
specific scale by its ``key`` in the ``info`` file (e.g. ``/s0`` or
``/8.0x8.0x8.0``); otherwise scale 0 (highest resolution) is used.
"""

import json
import logging
import posixpath
from urllib.parse import urlparse
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


PRECOMPUTED_PREFIX = "precomputed://"


def is_precomputed_path(path):
    """Return True if *path* explicitly carries the ``precomputed://`` prefix.

    The prefix is no longer required — format auto-detection probes
    for an ``info`` marker file — but it's still accepted for
    backward compatibility and to allow copy-paste from neuroglancer.
    Use ``_detect_zarr_driver`` to determine the actual format of
    a path that may or may not carry the prefix.
    """
    return isinstance(path, str) and path.startswith(PRECOMPUTED_PREFIX)


def _strip_precomputed_prefix(path):
    return path[len(PRECOMPUTED_PREFIX):] if is_precomputed_path(path) else path


def _kvstore_for_url(url):
    """Build a TensorStore kvstore spec for *url* (excluding any path).

    Thin wrapper around :func:`mesh_n_bone.util.zarr_io.kvstore_for_path`
    kept for backward compatibility with existing test imports.
    """
    from mesh_n_bone.util.zarr_io import kvstore_for_path
    return kvstore_for_path(url)


def _fetch_info(kvstore_spec, base_path):
    """Read the precomputed ``info`` JSON at ``<base_path>/info``."""
    driver = kvstore_spec["driver"]
    info_path = posixpath.join(base_path.rstrip("/"), "info")
    if driver == "http":
        url = f"{kvstore_spec['base_url']}/{info_path}"
        request = Request(url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=15) as f:
            return json.load(f)
    if driver == "gcs":
        url = (
            f"https://storage.googleapis.com/"
            f"{kvstore_spec['bucket']}/{info_path}"
        )
        with urlopen(url, timeout=15) as f:
            return json.load(f)
    if driver == "s3":
        url = (
            f"https://{kvstore_spec['bucket']}.s3.amazonaws.com/{info_path}"
        )
        with urlopen(url, timeout=15) as f:
            return json.load(f)
    if driver == "file":
        with open(posixpath.join(base_path, "info")) as f:
            return json.load(f)
    raise ValueError(f"Unsupported kvstore driver: {driver!r}")


def parse_precomputed_path(path):
    """Resolve a precomputed path into (kvstore, base_path, info, scale_index).

    Accepts paths with or without the ``precomputed://`` prefix. The
    base_path points at the directory containing the ``info`` file.
    If the input path's final segment names a scale ``key`` in the info,
    that scale is selected; otherwise scale 0 is selected.

    Returns
    -------
    tuple[dict, str, dict, int]
        ``(kvstore_spec, base_path, info_dict, scale_index)``.
    """
    inner = _strip_precomputed_prefix(path).rstrip("/")
    kvstore, full_path = _kvstore_for_url(inner)

    # First try reading info at the full path.  If that fails, peel off
    # the last segment and treat it as a scale key.
    info = None
    base_path = full_path
    scale_key = None
    try:
        info = _fetch_info(kvstore, full_path)
    except Exception as e:
        logger.debug(f"info not found at {full_path!r}: {e}; trying parent")

    if info is None:
        parent = posixpath.dirname(full_path.rstrip("/"))
        candidate_key = posixpath.basename(full_path.rstrip("/"))
        info = _fetch_info(kvstore, parent)
        base_path = parent
        scale_key = candidate_key

    scales = info.get("scales", [])
    if not scales:
        raise ValueError(f"precomputed info at {base_path!r} has no scales")

    scale_index = 0
    if scale_key is not None:
        for i, s in enumerate(scales):
            if s.get("key") == scale_key:
                scale_index = i
                break
        else:
            raise ValueError(
                f"scale key {scale_key!r} not found in info "
                f"(available keys: {[s.get('key') for s in scales]})"
            )

    return kvstore, base_path, info, scale_index


def open_precomputed_tensorstore(path, scale_index=None):
    """Open a precomputed volume with TensorStore and drop the channel axis.

    Returns a 3D TensorStore handle in (x, y, z) order, matching the
    layout of an N5 dataset so the existing swap_axes=True codepath
    converts it to ZYX.
    """
    import tensorstore as ts

    kvstore, base_path, info, default_scale = parse_precomputed_path(path)
    if scale_index is None:
        scale_index = default_scale

    kvstore_with_path = dict(kvstore)
    kvstore_with_path["path"] = base_path.rstrip("/") + "/"

    spec = {
        "driver": "neuroglancer_precomputed",
        "kvstore": kvstore_with_path,
        "scale_index": int(scale_index),
    }
    ds = ts.open(spec, read=True, write=False).result()

    # Drop the channel dim so the dataset looks 3D in (x, y, z) order.
    if "channel" in ds.domain.labels:
        ds = ds[ts.d["channel"][0]]
    elif len(ds.domain) == 4:
        ds = ds[..., 0]
    return ds


def precomputed_array_metadata(path, scale_index=None):
    """Read shape/dtype/chunks/voxel_size/offset for a precomputed scale.

    All spatial values are returned in ZYX order to match the rest of
    the codebase.
    """
    _, _, info, default_scale = parse_precomputed_path(path)
    if scale_index is None:
        scale_index = default_scale
    scale = info["scales"][scale_index]

    size_xyz = list(scale["size"])
    resolution_xyz = list(scale["resolution"])
    voxel_offset_xyz = list(scale.get("voxel_offset") or [0, 0, 0])
    chunk_sizes = scale.get("chunk_sizes") or [[64, 64, 64]]
    chunk_xyz = list(chunk_sizes[0])
    dtype = info.get("data_type", "uint64")

    shape_zyx = tuple(size_xyz[::-1])
    chunk_zyx = tuple(chunk_xyz[::-1])
    voxel_size_zyx = [float(v) for v in resolution_xyz[::-1]]
    # voxel_offset in info is in voxels; convert to physical units (ZYX).
    offset_zyx = [
        int(o * r)
        for o, r in zip(voxel_offset_xyz[::-1], resolution_xyz[::-1])
    ]
    return {
        "shape": shape_zyx,
        "chunks": chunk_zyx,
        "dtype": dtype,
        "voxel_size": voxel_size_zyx,
        "offset": offset_zyx,
        "scale_index": scale_index,
        "info": info,
    }
