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
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Compute all-LOD downsamples for a single super-chunk of s0.

    ``s0_block`` is the s0 voxel data covering the super-chunk extent
    (caller is responsible for reading + clipping/zero-padding).
    Returns ``{lod: (output_block, output_origin_voxels)}`` for every
    LOD ≥ 1. LOD 0 isn't emitted — the caller writes/links s0 separately.
    """
    out = {}
    for k, factor in enumerate(per_lod_factors):
        if k == 0:
            continue
        f = np.asarray(factor, dtype=int)
        # Trim s0_block to the largest extent that's a multiple of f.
        trim = (np.array(s0_block.shape) // f) * f
        block = s0_block[: trim[0], : trim[1], : trim[2]]
        if block.size == 0:
            continue
        ds_block, _ = downsample_func(block, tuple(f.tolist()))
        out_origin = super_chunk_origin_voxels // f
        out[k] = (ds_block, out_origin)
    return out


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
        out_arrays[k] = _create_zarr_v2_array(
            ds_path, shape=lod_shape, chunks=out_chunk.tolist(), dtype=dtype,
        )

    # Optionally symlink s0
    if s0_source_path is not None:
        _try_symlink_s0(output_zarr_path, s0_source_path)

    # Super-chunk grid in s0 voxels
    super_chunk_shape = out_chunk * max_factor
    sc_grid = []
    for z0 in range(int(out_origin[0]), int(out_origin[0] + out_shape[0]), int(super_chunk_shape[0])):
        for y0 in range(int(out_origin[1]), int(out_origin[1] + out_shape[1]), int(super_chunk_shape[1])):
            for x0 in range(int(out_origin[2]), int(out_origin[2] + out_shape[2]), int(super_chunk_shape[2])):
                sc_grid.append(np.array([z0, y0, x0], dtype=np.int64))

    def _process_super_chunk(sc_origin):
        # Read range in s0 voxels — clip to dataset bounds for halo / dataset edge
        read_end = np.minimum(sc_origin + super_chunk_shape, dataset_shape)
        read_size = read_end - sc_origin
        s0_block = s0_reader(sc_origin, read_size)
        # Downsample to all missing LODs
        downsamples = downsample_super_chunk(
            s0_block, sc_origin, factors_per_lod,
            downsample_func, out_chunk,
        )
        # Write each LOD's chunk
        for k, (ds_block, ds_origin) in downsamples.items():
            if k not in out_arrays:
                continue
            arr = out_arrays[k]
            # Output position relative to out_origin/factor
            f = np.asarray(factors_per_lod[k], dtype=np.int64)
            local_origin = ds_origin - (out_origin // f)
            _write_zarr_v2_region(arr, local_origin, ds_block)

    if dispatch is None:
        for sc in sc_grid:
            _process_super_chunk(sc)
    else:
        dispatch(_process_super_chunk, sc_grid)

    logger.info(
        "pyramid_builder: built %d new scales at %s (factors=%s)",
        len(missing), output_zarr_path,
        [factor for _, factor in missing],
    )
    return output_zarr_path


# ---------------------------------------------------------------------------
# Minimal zarr v2 array helpers (no extra deps)
# ---------------------------------------------------------------------------


def _create_zarr_v2_array(path, shape, chunks, dtype):
    """Create a zarr v2 array on disk and return a handle."""
    import zarr
    arr = zarr.open_array(
        store=path, mode="w", shape=shape, chunks=chunks, dtype=dtype,
        fill_value=0,
    )
    return arr


def _write_zarr_v2_region(arr, origin_voxels, data):
    """Write ``data`` into ``arr`` at the given origin (zyx voxel coords)."""
    z, y, x = origin_voxels.tolist()
    zE, yE, xE = z + data.shape[0], y + data.shape[1], x + data.shape[2]
    # Clip in case the block extends past array bounds (boundary chunks)
    zE = min(zE, arr.shape[0]); yE = min(yE, arr.shape[1]); xE = min(xE, arr.shape[2])
    arr[z:zE, y:yE, x:xE] = data[: zE - z, : yE - y, : xE - x]


def _try_symlink_s0(pyramid_path, s0_source_path):
    """Symlink the pyramid's s0 to the source s0 array.

    Returns True on success. Logs a warning and returns False otherwise
    (e.g. cross-filesystem, remote source, permission denied).
    """
    target = os.path.join(pyramid_path, "s0")
    if os.path.exists(target):
        return True
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
