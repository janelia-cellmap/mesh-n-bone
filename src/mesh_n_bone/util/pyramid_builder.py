"""Auto-build missing OME-NGFF multiscale levels for the input zarr.

When ``Meshify(multires_strategy="downsample")`` runs on an input that
only exposes ``s0`` (no pre-computed coarser scales), every LOD beyond 0
has to downsample s0 in-worker, which serializes I/O and re-reads the
full volume per LOD. This module pre-computes the missing ``s_k``
arrays once, in parallel chunks, so each LOD reads pre-built
properly-chunked data.

Output layout: ``{output_directory}/_intermediate_scales.zarr`` is an
OME-NGFF v0.4 multiscales group containing ``s0`` (symlinked to the
source on local filesystems; copy or skip for remote) plus ``s1..s_n``.

Per-axis factors handle anisotropic voxels. Standard practice:
downsample by 2x along axes whose voxel size is within ~50% of the
finest, until the volume is approximately isotropic, then by 2x
uniformly. ``per_lod_factors_for_anisotropy`` implements this.

Fence-post correctness: each ``s_k`` voxel is the mode of a per-axis
``factor_k_axis`` cube of s0 voxels. If the cube straddles the ROI
edge, two modes are available:

- ``snap`` (default): snap the ROI origin/extent to multiples of the
  max per-axis factor times s0 voxel size, dropping a small fringe of
  data (up to ``max_factor - 1`` s0 voxels per edge).
- ``halo``: keep the ROI exact, read beyond the ROI when a cube needs
  it (clipped to dataset bounds). No data loss.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-LOD factor calculation for anisotropic voxels
# ---------------------------------------------------------------------------


def per_lod_factors_for_anisotropy(
    voxel_size_zyx: np.ndarray, num_lods: int,
    anisotropy_tolerance: float = 1.5,
) -> list[tuple[int, int, int]]:
    """Compute per-axis downsample factors for each LOD.

    At each step, an axis is downsampled by 2 if its current voxel size
    is at or below ``anisotropy_tolerance * min(voxel_size)``. If no
    axis qualifies (already isotropic), all axes downsample by 2.

    Returns a list of cumulative (Fz, Fy, Fx) factors per LOD, including
    LOD 0 which is always (1, 1, 1).

    Example: voxel_size=[8, 8, 20], num_lods=4
        LOD 0: (1, 1, 1)  voxel=[8, 8, 20]
        LOD 1: (2, 2, 1)  voxel=[16, 16, 20]  → near-isotropic
        LOD 2: (4, 4, 2)  voxel=[32, 32, 40]
        LOD 3: (8, 8, 4)  voxel=[64, 64, 80]
    """
    cur_vs = np.asarray(voxel_size_zyx, dtype=float).copy()
    cur_factor = np.array([1, 1, 1], dtype=int)
    out = [tuple(cur_factor.tolist())]
    for _ in range(1, num_lods):
        min_vs = cur_vs.min()
        step = np.ones(3, dtype=int)
        for ax in range(3):
            if cur_vs[ax] <= anisotropy_tolerance * min_vs:
                step[ax] = 2
        if np.all(step == 1):
            step[:] = 2
        cur_factor = cur_factor * step
        cur_vs = cur_vs * step
        out.append(tuple(cur_factor.tolist()))
    return out


# ---------------------------------------------------------------------------
# ROI alignment
# ---------------------------------------------------------------------------


def align_roi_voxels(
    roi_origin_voxels: np.ndarray, roi_shape_voxels: np.ndarray,
    max_per_axis_factor: np.ndarray, mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align an ROI to ``max_per_axis_factor`` voxel boundaries.

    Returns ``(aligned_origin, aligned_shape, read_origin, read_shape)``
    where ``aligned_*`` is the OUTPUT region (always 2^k-aligned to the
    s0 grid) and ``read_*`` is the s0 region the pyramid worker reads:

    - ``mode="snap"``: read_* = aligned_*. Voxels in the original ROI
      but outside the snapped region are not processed.
    - ``mode="halo"``: aligned_* = original ROI snapped INWARD to factor
      multiples; read_* extends beyond aligned_* OUTWARD by up to
      ``max_per_axis_factor - 1`` voxels so every output cube reads its
      full s0 input. Caller is responsible for clipping ``read_*`` to
      the dataset bounds.
    """
    roi_origin = np.asarray(roi_origin_voxels, dtype=np.int64)
    roi_shape = np.asarray(roi_shape_voxels, dtype=np.int64)
    factor = np.asarray(max_per_axis_factor, dtype=np.int64)

    if mode == "snap":
        new_origin = ((roi_origin + factor - 1) // factor) * factor
        end = roi_origin + roi_shape
        new_end = (end // factor) * factor
        new_shape = np.maximum(new_end - new_origin, 0)
        return new_origin, new_shape, new_origin.copy(), new_shape.copy()

    if mode == "halo":
        # Output region: aligned-inward (so output cubes fit inside the ROI's
        # aligned envelope), but we extend READS outward to complete the
        # boundary cubes. Final output covers the ROI rounded to factor
        # boundaries OUTWARD (so the ROI is fully covered).
        out_origin = (roi_origin // factor) * factor
        out_end = ((roi_origin + roi_shape + factor - 1) // factor) * factor
        out_shape = out_end - out_origin
        return out_origin, out_shape, out_origin.copy(), out_shape.copy()

    raise ValueError(f"unknown alignment mode: {mode!r}")


# ---------------------------------------------------------------------------
# OME-NGFF multiscales metadata
# ---------------------------------------------------------------------------


def build_multiscales_metadata(
    s0_voxel_size_zyx: list[float],
    s0_translation_zyx: list[float],
    per_lod_factors: list[tuple[int, int, int]],
    axes: list[str] = ("z", "y", "x"),
    unit: str = "nanometer",
    version: str = "0.4",
) -> dict[str, Any]:
    """Build an OME-NGFF multiscales metadata dict for a pyramid.

    Translation per level uses the OME convention from cellmap-analyze
    (voxel centers): ``tr_k_axis = tr_0_axis + 0.5 * vs0_axis * (F_k_axis - 1)``
    where ``F_k_axis`` is the cumulative downsample factor at LOD k.
    """
    s0_vs = np.asarray(s0_voxel_size_zyx, dtype=float)
    s0_tr = np.asarray(s0_translation_zyx, dtype=float)
    datasets = []
    for k, factor in enumerate(per_lod_factors):
        f = np.asarray(factor, dtype=float)
        vs_k = (s0_vs * f).tolist()
        tr_k = (s0_tr + 0.5 * s0_vs * (f - 1.0)).tolist()
        datasets.append({
            "path": f"s{k}",
            "coordinateTransformations": [
                {"type": "scale", "scale": vs_k},
                {"type": "translation", "translation": tr_k},
            ],
        })
    return {
        "multiscales": [{
            "version": version,
            "name": "",
            "axes": [{"name": a, "type": "space", "unit": unit} for a in axes],
            "datasets": datasets,
        }],
    }


def write_multiscales_metadata(
    group_path: str, metadata: dict[str, Any], zarr_format: int = 2,
) -> None:
    """Persist multiscales attrs at the zarr group root.

    Writes ``.zattrs`` for zarr v2 layout or ``zarr.json`` for v3.
    """
    os.makedirs(group_path, exist_ok=True)
    if zarr_format == 2:
        with open(os.path.join(group_path, ".zattrs"), "w") as f:
            json.dump(metadata, f, indent=2)
        # Also need a minimal .zgroup
        zg_path = os.path.join(group_path, ".zgroup")
        if not os.path.exists(zg_path):
            with open(zg_path, "w") as f:
                json.dump({"zarr_format": 2}, f)
    elif zarr_format == 3:
        with open(os.path.join(group_path, "zarr.json"), "w") as f:
            json.dump({
                "zarr_format": 3,
                "node_type": "group",
                "attributes": metadata,
            }, f, indent=2)
    else:
        raise ValueError(f"unsupported zarr_format: {zarr_format}")


# ---------------------------------------------------------------------------
# Super-chunk worker
# ---------------------------------------------------------------------------


def downsample_super_chunk(
    s0_block: np.ndarray,
    super_chunk_origin_voxels: np.ndarray,
    per_lod_factors: list[tuple[int, int, int]],
    downsample_func,
    out_chunk_shape: np.ndarray,
    write_chunk=None,
) -> dict[int, tuple[np.ndarray, np.ndarray]] | None:
    """Direct-downsample a single super-chunk of s0 into all LODs.

    Each output LOD voxel = ``downsample_func(s0 cube)`` where the cube
    extent is the LOD's per-axis cumulative factor. We downsample s0
    directly at each LOD (not s_{k-1} → s_k cascade).

    Memory: with the ``write_chunk`` callback, each LOD's output is
    written immediately and dropped — peak in-memory at any moment is
    ``s0_block + one_lod_output``. Without the callback, all LOD blocks
    are collected and returned (legacy dict API kept for tests).

    Parameters
    ----------
    write_chunk : callable, optional
        ``write_chunk(lod_k, ds_block, out_origin)``. Called per LOD as
        soon as the block is computed. Function returns ``None``.
    """
    collected = None if write_chunk is not None else {}
    for k, factor in enumerate(per_lod_factors):
        if k == 0:
            continue
        f = np.asarray(factor, dtype=int)
        # Trim s0_block to the largest extent that's a multiple of f
        trim = (np.array(s0_block.shape) // f) * f
        block = s0_block[: trim[0], : trim[1], : trim[2]]
        if block.size == 0:
            continue
        ds_block, _ = downsample_func(block, tuple(f.tolist()))
        out_origin = super_chunk_origin_voxels // f
        if write_chunk is not None:
            # Drop the reference inside this iteration so the buffer
            # can be GC'd before the next downsample allocates
            write_chunk(k, ds_block, out_origin)
            del ds_block
        else:
            collected[k] = (ds_block, out_origin)
    return collected


# ---------------------------------------------------------------------------
# Dask-cluster worker: runs ONE super-chunk task in a dask worker process.
# Module-level so the function reference is picklable for dask.
# ---------------------------------------------------------------------------


# Per-worker-process cache: tensorstore handles, keyed by path.
# Open is non-trivial (info-file fetch) — caching saves time across the
# many tasks a single dask worker processes.
_PYRAMID_WORKER_TS_CACHE: dict[str, object] = {}


def _ts_handle_for_input(dataset_path):
    if dataset_path in _PYRAMID_WORKER_TS_CACHE:
        return _PYRAMID_WORKER_TS_CACHE[dataset_path]
    from mesh_n_bone.util.image_data_interface import open_ds_tensorstore
    handle = open_ds_tensorstore(dataset_path)
    _PYRAMID_WORKER_TS_CACHE[dataset_path] = handle
    return handle


def _ts_handle_for_output(path, zarr_format=2):
    """Open a pyramid-owned zarr array (one we created, or the immediately
    preceding cascade level, per this pyramid's own ``zarr_format``) for
    read/write. Not for arbitrary external arrays — see
    ``_ts_handle_for_input`` for those (driver auto-detected)."""
    cache_key = (path, zarr_format)
    if cache_key in _PYRAMID_WORKER_TS_CACHE:
        return _PYRAMID_WORKER_TS_CACHE[cache_key]
    import tensorstore as ts
    from mesh_n_bone.util.image_data_interface import (
        _capped_tensorstore_context_spec,
    )
    handle = ts.open({
        "driver": "zarr3" if zarr_format == 3 else "zarr",
        "kvstore": {"driver": "file", "path": path},
        "open": True,
        # Cap thread pools + disable decoded-chunk cache. Without this
        # the output handle's cache_pool accumulates encoded chunks for
        # the worker's lifetime, OOMing on large super-chunk runs.
        "context": _capped_tensorstore_context_spec(),
    }).result()
    _PYRAMID_WORKER_TS_CACHE[cache_key] = handle
    return handle


def process_super_chunk_for_dask(sc_origin_tuple, worker_config):
    """Process ONE pyramid super-chunk inside a dask worker process.

    Module-level + serialization-safe so dask can ship this to LSF workers.
    ``worker_config`` is a plain dict of strings/tuples/numbers (no numpy
    arrays, no closures, no opened handles).

    Reads its s0 super-chunk, downsamples to each missing LOD via the
    configured ``downsample_method``, writes each LOD's chunk to the
    pre-created output zarr arrays under ``worker_config["pyramid_path"]``.

    Returns ``None`` (dask bag collects None values; we don't aggregate).
    """
    from funlib.geometry import Coordinate, Roi
    from mesh_n_bone.util.image_data_interface import to_ndarray_tensorstore
    from mesh_n_bone.meshify.downsample import (
        downsample_labels_3d,
        downsample_binary_3d,
        downsample_labels_3d_suppress_zero,
    )

    dispatch = {
        "mode": downsample_labels_3d,
        "mode_suppress_zero": downsample_labels_3d_suppress_zero,
        "binary": downsample_binary_3d,
    }
    downsample_func = dispatch[worker_config["downsample_method"]]

    sc_origin = np.array(sc_origin_tuple, dtype=np.int64)
    out_origin = np.array(worker_config["out_origin"], dtype=np.int64)
    super_chunk_shape = np.array(worker_config["super_chunk_shape"], dtype=np.int64)
    dataset_shape = np.array(worker_config["dataset_shape"], dtype=np.int64)
    factors_per_lod = [tuple(f) for f in worker_config["factors_per_lod"]]
    missing_lods = list(worker_config["missing_lods"])

    # Read s0 super-chunk in world coords (same pattern as the meshify worker)
    ds_offset = Coordinate(worker_config["dataset_offset"])
    voxel_size = Coordinate(worker_config["voxel_size"])
    read_end_voxels = np.minimum(sc_origin + super_chunk_shape, dataset_shape)
    read_size_voxels = read_end_voxels - sc_origin
    if np.any(read_size_voxels <= 0):
        return None
    world_origin = ds_offset + Coordinate(sc_origin.tolist()) * voxel_size
    world_shape = Coordinate(read_size_voxels.tolist()) * voxel_size
    roi = Roi(world_origin, world_shape)

    ts_s0 = _ts_handle_for_input(worker_config["s0_dataset_path"])
    s0_block = to_ndarray_tensorstore(
        ts_s0, roi, voxel_size, ds_offset,
        swap_axes=worker_config["swap_axes"], fill_value=0,
        source_path=worker_config["s0_dataset_path"],
    )

    # Downsample + write each LOD's chunk (streaming: drop the buffer after write)
    for k in missing_lods:
        f = np.asarray(factors_per_lod[k], dtype=int)
        trim = (np.array(s0_block.shape) // f) * f
        block = s0_block[: trim[0], : trim[1], : trim[2]]
        if block.size == 0:
            continue
        ds_block, _ = downsample_func(block, tuple(f.tolist()))
        ds_origin = sc_origin // f
        local_origin = ds_origin - (out_origin // f)
        out_path = os.path.join(worker_config["pyramid_path"], f"s{k}")
        out_arr = _ts_handle_for_output(out_path, worker_config.get("zarr_format", 2))
        z, y, x = local_origin.tolist()
        arr_shape = out_arr.shape
        zE = min(z + ds_block.shape[0], arr_shape[0])
        yE = min(y + ds_block.shape[1], arr_shape[1])
        xE = min(x + ds_block.shape[2], arr_shape[2])
        out_arr[z:zE, y:yE, x:xE].write(
            ds_block[: zE - z, : yE - y, : xE - x]
        ).result()
        del ds_block

    # Release the big s0 read buffer + force allocator release. Without this
    # the worker accumulates s0 read residue across the 8+ super-chunks it
    # processes before exit (cached tensorstore handles in
    # _PYRAMID_WORKER_TS_CACHE don't release on their own, and glibc malloc
    # doesn't munmap freed pages without an explicit trim).
    del s0_block
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass

    return None


def process_cascade_chunk_for_dask(sc_origin_tuple, worker_config):
    """Process ONE cascade super-chunk in a dask worker process.

    Unlike ``process_super_chunk_for_dask`` (which always reads a
    world-coordinate region of the *original* source array, with its own
    offset/voxel-size/``swap_axes`` bookkeeping), this downsamples a single
    per-axis ``step_factor`` from the immediately preceding pyramid level —
    a plain, already ZYX-native array starting at voxel (0, 0, 0), in
    whichever zarr format (``worker_config["zarr_format"]``) this whole
    pyramid was written in, since it was written by this same builder.

    Module-level so the function reference is picklable for dask.
    """
    from mesh_n_bone.meshify.downsample import (
        downsample_labels_3d,
        downsample_binary_3d,
        downsample_labels_3d_suppress_zero,
    )

    dispatch = {
        "mode": downsample_labels_3d,
        "mode_suppress_zero": downsample_labels_3d_suppress_zero,
        "binary": downsample_binary_3d,
    }
    downsample_func = dispatch[worker_config["downsample_method"]]

    sc_origin = np.array(sc_origin_tuple, dtype=np.int64)
    step = np.asarray(worker_config["step_factor"], dtype=np.int64)
    chunk_step = np.asarray(worker_config["super_chunk_shape"], dtype=np.int64)
    zarr_format = worker_config.get("zarr_format", 2)

    read_arr = _ts_handle_for_output(worker_config["read_path"], zarr_format)
    read_shape = np.array(read_arr.shape, dtype=np.int64)
    read_end = np.minimum(sc_origin + chunk_step, read_shape)
    read_size = read_end - sc_origin
    if np.any(read_size <= 0):
        return None
    z, y, x = sc_origin.tolist()
    zE, yE, xE = read_end.tolist()
    block = read_arr[z:zE, y:yE, x:xE].read().result()

    trim = (np.array(block.shape) // step) * step
    block = block[: trim[0], : trim[1], : trim[2]]
    if block.size == 0:
        return None
    ds_block, _ = downsample_func(block, tuple(step.tolist()))

    write_origin = sc_origin // step
    out_arr = _ts_handle_for_output(worker_config["write_path"], zarr_format)
    oz, oy, ox = write_origin.tolist()
    arr_shape = out_arr.shape
    ozE = min(oz + ds_block.shape[0], arr_shape[0])
    oyE = min(oy + ds_block.shape[1], arr_shape[1])
    oxE = min(ox + ds_block.shape[2], arr_shape[2])
    out_arr[oz:ozE, oy:oyE, ox:oxE].write(
        ds_block[: ozE - oz, : oyE - oy, : oxE - ox]
    ).result()
    del block, ds_block

    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass

    return None


def prepare_pyramid_metadata_and_arrays(
    *,
    output_zarr_path,
    factors_per_lod,
    missing_lods,
    out_shape,
    out_chunk_shape_voxels,
    dtype,
    s0_voxel_size_zyx,
    s0_translation_zyx,
    zarr_format=2,
    s0_source_path=None,
):
    """Driver-side setup before dispatching pyramid super-chunks to dask.

    Writes the OME-NGFF multiscales metadata, creates empty output zarr
    arrays for each missing LOD, and optionally symlinks s0. Returns
    ``(super_chunk_shape, max_factor)``.
    """
    os.makedirs(output_zarr_path, exist_ok=True)
    metadata = build_multiscales_metadata(
        s0_voxel_size_zyx=list(s0_voxel_size_zyx),
        s0_translation_zyx=list(s0_translation_zyx),
        per_lod_factors=factors_per_lod,
        version="0.4",
    )
    write_multiscales_metadata(output_zarr_path, metadata, zarr_format=zarr_format)

    out_chunk = np.asarray(out_chunk_shape_voxels, dtype=np.int64)
    out_shape = np.asarray(out_shape, dtype=np.int64)
    max_factor = np.max(np.array(factors_per_lod), axis=0)
    for k in missing_lods:
        f = np.asarray(factors_per_lod[k], dtype=np.int64)
        lod_shape = ((out_shape + f - 1) // f).tolist()
        ds_path = os.path.join(output_zarr_path, f"s{k}")
        if os.path.exists(ds_path):
            shutil.rmtree(ds_path)
        _create_zarr_array(
            ds_path, shape=lod_shape, chunks=out_chunk.tolist(), dtype=dtype,
            zarr_format=zarr_format,
        )

    if s0_source_path is not None:
        _try_symlink_s0(output_zarr_path, s0_source_path, zarr_format=zarr_format)

    super_chunk_shape = out_chunk * max_factor
    return super_chunk_shape, max_factor


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_missing_pyramid_levels(
    *,
    s0_reader,  # callable(origin_voxels, shape_voxels) -> np.ndarray
    s0_dataset_shape_voxels: np.ndarray,
    s0_voxel_size_zyx: list[float],
    s0_translation_zyx: list[float],
    dtype: np.dtype,
    num_lods: int,
    existing_factors: set[tuple[int, int, int]],
    output_zarr_path: str,
    downsample_func,
    roi_origin_voxels: np.ndarray | None = None,
    roi_shape_voxels: np.ndarray | None = None,
    out_chunk_shape_voxels: tuple[int, int, int] = (64, 64, 64),
    alignment_mode: str = "snap",
    anisotropy_tolerance: float = 1.5,
    zarr_format: int = 2,
    s0_source_path: str | None = None,
    dispatch=None,
    num_workers: int = 1,
    cascade: bool = False,
) -> str:
    """Build missing OME-NGFF pyramid levels and return the group path.

    Reads s0 via ``s0_reader`` in chunk-aligned super-chunks (each
    containing one output chunk worth of voxels at every LOD), computes
    the per-LOD downsamples in memory, and writes each LOD's voxels to
    ``output_zarr_path/s_k``.

    On local filesystems, ``s0_source_path`` (if provided) is symlinked
    in as ``output_zarr_path/s0``. For remote sources or symlink
    failures, ``s0`` is left absent — discovery is responsible for
    merging the source's own s0 metadata with this pyramid's s_k>0.

    ``dispatch`` is an optional ``callable(func, args_iter) -> results``
    for parallelizing the super-chunk pass. If ``None``, the pass is
    sequential.

    ``cascade``: when ``False`` (default), every missing level is
    downsampled directly from s0 by its cumulative factor (as described
    above). When ``True``, each missing level is instead downsampled from
    the immediately preceding level ``s_{k-1}`` — but only when that
    predecessor was ALSO just built by this call (i.e. ``k-1`` is missing
    and directly precedes ``k``); if there's a gap (e.g. ``s_{k-1}`` was
    already present in ``existing_factors``, so this driver has no reader
    for its contents), the chain resets and that level is built directly
    from s0 instead. This keeps peak per-task memory bounded by the
    largest *step* between consecutive missing levels rather than the
    largest *cumulative* factor — the win grows with ``num_lods``. Cascade
    composition is exact for associative reducers (e.g. ``np.any``-based
    binary downsampling) and an approximation for majority-vote reducers
    (``mode``, ``mode_suppress_zero``, ``binary``): mode-of-modes can
    differ from the true global mode at label-boundary voxels.

    Returns the path to the pyramid group.
    """
    if num_lods <= 1:
        return output_zarr_path  # nothing to build

    voxel_size = np.asarray(s0_voxel_size_zyx, dtype=float)
    translation = np.asarray(s0_translation_zyx, dtype=float)
    dataset_shape = np.asarray(s0_dataset_shape_voxels, dtype=np.int64)

    # Per-LOD cumulative factors (per-axis)
    factors_per_lod = per_lod_factors_for_anisotropy(
        voxel_size, num_lods, anisotropy_tolerance=anisotropy_tolerance,
    )
    max_factor = np.max(np.array(factors_per_lod), axis=0)  # per-axis

    # ROI handling
    if roi_origin_voxels is None:
        roi_origin_voxels = np.zeros(3, dtype=np.int64)
    if roi_shape_voxels is None:
        roi_shape_voxels = dataset_shape - roi_origin_voxels

    out_origin, out_shape, read_origin, read_shape = align_roi_voxels(
        roi_origin_voxels, roi_shape_voxels, max_factor, alignment_mode,
    )

    if not np.array_equal(out_origin, roi_origin_voxels) or not np.array_equal(
        out_shape, roi_shape_voxels
    ):
        logger.info(
            "pyramid_builder: %s aligned ROI from origin=%s shape=%s to "
            "origin=%s shape=%s (max per-axis factor=%s)",
            alignment_mode, roi_origin_voxels.tolist(), roi_shape_voxels.tolist(),
            out_origin.tolist(), out_shape.tolist(), max_factor.tolist(),
        )

    # Decide which levels need building (factors not in existing_factors)
    missing = [
        (k, factor) for k, factor in enumerate(factors_per_lod)
        if k > 0 and factor not in existing_factors
    ]
    if not missing:
        logger.info("pyramid_builder: all required scales already present")
        return output_zarr_path

    # Emit OME-NGFF metadata (covering ALL levels including s0 for completeness)
    metadata = build_multiscales_metadata(
        s0_voxel_size_zyx=voxel_size.tolist(),
        s0_translation_zyx=translation.tolist(),
        per_lod_factors=factors_per_lod,
        version="0.4",
    )
    write_multiscales_metadata(output_zarr_path, metadata, zarr_format=zarr_format)

    # Create empty zarr arrays for each missing level
    out_chunk = np.asarray(out_chunk_shape_voxels, dtype=np.int64)
    out_arrays = {}
    for k, factor in missing:
        f = np.asarray(factor, dtype=np.int64)
        lod_shape = ((out_shape + f - 1) // f).tolist()
        ds_path = os.path.join(output_zarr_path, f"s{k}")
        if os.path.exists(ds_path):
            shutil.rmtree(ds_path)
        out_arrays[k] = _create_zarr_array(
            ds_path, shape=lod_shape, chunks=out_chunk.tolist(), dtype=dtype,
            zarr_format=zarr_format,
        )

    # Optionally symlink s0
    if s0_source_path is not None:
        _try_symlink_s0(output_zarr_path, s0_source_path, zarr_format=zarr_format)

    def _run_chunk_grid(sc_grid, process_one, label):
        n_chunks = len(sc_grid)
        if n_chunks == 0:
            return
        if dispatch is not None:
            dispatch(process_one, sc_grid)
            return
        if num_workers > 1 and n_chunks > 1:
            from concurrent.futures import ThreadPoolExecutor
            import threading
            workers = min(num_workers, n_chunks)
            logger.info(
                "pyramid_builder: dispatching %d super-chunks across %d "
                "threads (%s)", n_chunks, workers, label,
            )
            done = [0]
            report_every = max(1, n_chunks // 20)
            lock = threading.Lock()

            def _worker(sc):
                process_one(sc)
                with lock:
                    done[0] += 1
                    if done[0] % report_every == 0 or done[0] == n_chunks:
                        logger.info(
                            "pyramid_builder: %d/%d super-chunks done", done[0], n_chunks,
                        )

            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(_worker, sc_grid))
        else:
            logger.info(
                "pyramid_builder: processing %d super-chunks sequentially "
                "(%s)", n_chunks, label,
            )
            report_every = max(1, n_chunks // 20)
            for i, sc in enumerate(sc_grid):
                process_one(sc)
                done = i + 1
                if done % report_every == 0 or done == n_chunks:
                    logger.info(
                        "pyramid_builder: %d/%d super-chunks done", done, n_chunks,
                    )

    if not cascade:
        # Direct: one super-chunk pass computes EVERY missing LOD from a
        # single s0 read.
        super_chunk_shape = out_chunk * max_factor
        sc_grid = []
        for z0 in range(int(out_origin[0]), int(out_origin[0] + out_shape[0]), int(super_chunk_shape[0])):
            for y0 in range(int(out_origin[1]), int(out_origin[1] + out_shape[1]), int(super_chunk_shape[1])):
                for x0 in range(int(out_origin[2]), int(out_origin[2] + out_shape[2]), int(super_chunk_shape[2])):
                    sc_grid.append(np.array([z0, y0, x0], dtype=np.int64))

        def _process_super_chunk(sc_origin):
            read_end = np.minimum(sc_origin + super_chunk_shape, dataset_shape)
            read_size = read_end - sc_origin
            s0_block = s0_reader(sc_origin, read_size)

            def _write_one(k, ds_block, ds_origin):
                if k not in out_arrays:
                    return
                arr = out_arrays[k]
                f = np.asarray(factors_per_lod[k], dtype=np.int64)
                local_origin = ds_origin - (out_origin // f)
                _write_zarr_v2_region(arr, local_origin, ds_block)

            # Stream each LOD's output: compute → write → drop the buffer.
            # Peak memory per task ≈ s0_block + one_lod_output.
            downsample_super_chunk(
                s0_block, sc_origin, factors_per_lod,
                downsample_func, out_chunk,
                write_chunk=_write_one,
            )

        _run_chunk_grid(
            sc_grid, _process_super_chunk,
            f"super_chunk_shape={super_chunk_shape.tolist()} s0 voxels",
        )
    else:
        # Cascade: build missing levels sequentially, each from the
        # nearest predecessor this call ALSO just built. A gap (the
        # predecessor was already present in existing_factors, so we have
        # no reader for it here) resets the chain to read s0 directly.
        import zarr as _zarr

        prev_k, prev_factor = None, (1, 1, 1)
        for k, factor in missing:
            read_from_s0 = not (prev_k is not None and k == prev_k + 1)
            step = factor if read_from_s0 else tuple(
                int(a // b) for a, b in zip(factor, prev_factor)
            )
            step_arr = np.asarray(step, dtype=np.int64)
            chunk_step = out_chunk * step_arr
            out_arr = out_arrays[k]

            if read_from_s0:
                grid_origin, grid_shape, bound_shape = out_origin, out_shape, dataset_shape
                prev_arr = None
            else:
                grid_origin = np.zeros(3, dtype=np.int64)
                prev_arr = _zarr.open_array(
                    os.path.join(output_zarr_path, f"s{prev_k}"), mode="r",
                )
                grid_shape = np.asarray(prev_arr.shape, dtype=np.int64)
                bound_shape = grid_shape

            sc_grid = []
            for z0 in range(int(grid_origin[0]), int(grid_origin[0] + grid_shape[0]), int(chunk_step[0])):
                for y0 in range(int(grid_origin[1]), int(grid_origin[1] + grid_shape[1]), int(chunk_step[1])):
                    for x0 in range(int(grid_origin[2]), int(grid_origin[2] + grid_shape[2]), int(chunk_step[2])):
                        sc_grid.append(np.array([z0, y0, x0], dtype=np.int64))

            def _process_cascade_chunk(sc_origin, chunk_step=chunk_step, bound_shape=bound_shape,
                                        read_from_s0=read_from_s0, prev_arr=prev_arr,
                                        step_arr=step_arr, out_origin_for_read=out_origin,
                                        out_arr=out_arr):
                read_end = np.minimum(sc_origin + chunk_step, bound_shape)
                read_size = read_end - sc_origin
                if np.any(read_size <= 0):
                    return
                if read_from_s0:
                    block = s0_reader(sc_origin, read_size)
                    write_origin = (sc_origin // step_arr) - (out_origin_for_read // step_arr)
                else:
                    z, y, x = sc_origin.tolist()
                    zE, yE, xE = read_end.tolist()
                    block = np.asarray(prev_arr[z:zE, y:yE, x:xE])
                    write_origin = sc_origin // step_arr

                trim = (np.array(block.shape) // step_arr) * step_arr
                block = block[: trim[0], : trim[1], : trim[2]]
                if block.size == 0:
                    return
                ds_block, _ = downsample_func(block, tuple(step_arr.tolist()))
                _write_zarr_v2_region(out_arr, write_origin, ds_block)

            _run_chunk_grid(
                sc_grid, _process_cascade_chunk,
                f"s{k} from {'s0' if read_from_s0 else f's{prev_k}'}, step={step}",
            )
            prev_k, prev_factor = k, factor

    logger.info(
        "pyramid_builder: built %d new scales at %s (factors=%s)",
        len(missing), output_zarr_path,
        [factor for _, factor in missing],
    )
    return output_zarr_path


# ---------------------------------------------------------------------------
# Minimal zarr v2/v3 array helpers (no extra deps)
# ---------------------------------------------------------------------------


_TS_DTYPE_MAP = {
    np.dtype("uint8"): "|u1",
    np.dtype("uint16"): "<u2",
    np.dtype("uint32"): "<u4",
    np.dtype("uint64"): "<u8",
    np.dtype("int8"): "|i1",
    np.dtype("int16"): "<i2",
    np.dtype("int32"): "<i4",
    np.dtype("int64"): "<i8",
    np.dtype("float32"): "<f4",
    np.dtype("float64"): "<f8",
}

# zarr v3's core spec names data types after their plain numpy names
# (no byte-order/size prefix like v2's "<u2").
_TS_V3_DTYPE_MAP = {
    np.dtype("uint8"): "uint8",
    np.dtype("uint16"): "uint16",
    np.dtype("uint32"): "uint32",
    np.dtype("uint64"): "uint64",
    np.dtype("int8"): "int8",
    np.dtype("int16"): "int16",
    np.dtype("int32"): "int32",
    np.dtype("int64"): "int64",
    np.dtype("float32"): "float32",
    np.dtype("float64"): "float64",
}


def _create_zarr_array(path, shape, chunks, dtype, zarr_format=2):
    """Create a zarr array on disk via tensorstore and return a handle.

    Uses tensorstore (already a core dep) instead of the optional ``zarr``
    package so the pyramid builder works in any pixi env that has the
    base mesh-n-bone install. ``zarr_format`` (2 or 3) should match
    whatever format this whole pyramid is being written in — see
    ``_try_symlink_s0`` for why that matters.
    """
    import tensorstore as ts
    if os.path.exists(path):
        shutil.rmtree(path)
    dt = np.dtype(dtype)
    if zarr_format == 3:
        ts_dtype = _TS_V3_DTYPE_MAP.get(dt)
        if ts_dtype is None:
            raise ValueError(
                f"Unsupported dtype for pyramid build: {dt}. "
                f"Add it to _TS_V3_DTYPE_MAP."
            )
        spec = {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": path},
            "metadata": {
                "shape": list(shape),
                "data_type": ts_dtype,
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": list(chunks)},
                },
                # Uncompressed, matching the v2 branch's "compressor": None
                # — mode/label downsamples are CPU- not I/O-bound here.
                "codecs": [{"name": "bytes"}],
                "fill_value": 0,
            },
            "create": True,
            "delete_existing": True,
        }
    else:
        ts_dtype = _TS_DTYPE_MAP.get(dt)
        if ts_dtype is None:
            raise ValueError(
                f"Unsupported dtype for pyramid build: {dt}. "
                f"Add it to _TS_DTYPE_MAP."
            )
        spec = {
            "driver": "zarr",  # zarr v2
            "kvstore": {"driver": "file", "path": path},
            "metadata": {
                "shape": list(shape),
                "chunks": list(chunks),
                "dtype": ts_dtype,
                "compressor": None,
                "fill_value": 0,
                "order": "C",
            },
            "create": True,
            "delete_existing": True,
        }
    return ts.open(spec).result()


def _write_zarr_v2_region(arr, origin_voxels, data):
    """Write ``data`` into ``arr`` at the given origin (zyx voxel coords).

    ``arr`` is a tensorstore handle from ``_create_zarr_array`` — despite
    the name (kept for backwards compatibility), this write path is
    format-agnostic; it works the same for v2 or v3 handles.
    """
    z, y, x = origin_voxels.tolist()
    shape = arr.shape
    zE = min(z + data.shape[0], shape[0])
    yE = min(y + data.shape[1], shape[1])
    xE = min(x + data.shape[2], shape[2])
    # Clip the data block to the array bounds (boundary chunks at the edge)
    clipped = data[: zE - z, : yE - y, : xE - x]
    arr[z:zE, y:yE, x:xE].write(clipped).result()


def _try_symlink_s0(pyramid_path, s0_source_path, zarr_format=2):
    """Symlink the pyramid's s0 to the source s0 array.

    Returns True on success. Logs a warning and returns False otherwise
    (e.g. cross-filesystem, remote source, permission denied, or a zarr
    format mismatch — see below).
    """
    target = os.path.join(pyramid_path, "s0")
    if os.path.exists(target):
        return True

    # This pyramid's own group metadata (.zgroup/.zattrs or zarr.json) and
    # its s1+ arrays are always written in ``zarr_format`` (see
    # _create_zarr_array). If the real source array is in a DIFFERENT
    # format, symlinking it in as s0 produces a group that LOOKS complete
    # but is format-inconsistent: generic OME-zarr multiscale readers
    # (e.g. neuroglancer) resolve the group's declared format and then
    # look for THAT format's array metadata inside s0 — which has the
    # other format's metadata file instead — and fail with something like
    # "zarr v{version} array metadata not found", even though s0 is
    # perfectly readable on its own (e.g. with an explicit driver
    # override). s1+ are unaffected since those really do match.
    # Leave s0 absent rather than link in something unreadable through
    # the group's declared format — mesh-n-bone's own LOD-reading logic
    # never depends on this symlink anyway (LOD 0 always reads the real
    # source path directly); it's a convenience for external tools only.
    # (Callers should normally have already matched ``zarr_format`` to
    # the source's own format — see Meshify._build_missing_pyramid_scales
    # — so this is mostly a safety net for n5/precomputed sources, which
    # can't be represented as a zarr array at all.)
    expected_driver = "zarr3" if zarr_format == 3 else "zarr"
    try:
        from mesh_n_bone.util.image_data_interface import _detect_zarr_driver
        actual_driver = _detect_zarr_driver(s0_source_path)
        if actual_driver != expected_driver:
            logger.warning(
                "pyramid_builder: not symlinking s0 from %s into %s — "
                "source format (%s) doesn't match this pyramid's "
                "zarr_format=%d (%s). s0 stays absent from the pyramid; "
                "it's still directly browsable at its own real path/URL.",
                s0_source_path, target, actual_driver, zarr_format, expected_driver,
            )
            return False
    except Exception as e:
        logger.warning(
            "pyramid_builder: could not detect zarr format of %s (%s); "
            "proceeding with symlink attempt.", s0_source_path, e,
        )

    try:
        os.symlink(s0_source_path, target)
        return True
    except OSError as e:
        logger.warning(
            "pyramid_builder: could not symlink s0 from %s into %s: %s. "
            "Discovery will fall back to merging the source group's own s0.",
            s0_source_path, target, e,
        )
        return False
