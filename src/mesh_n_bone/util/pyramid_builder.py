"""Auto-build missing OME-NGFF multiscale levels for the input zarr.

When ``Meshify(multires_strategy="downsample")`` runs on an input that
only exposes ``s0`` (no pre-computed coarser scales), every LOD beyond 0
has to downsample s0 in-worker, which serializes I/O and re-reads the
full volume per LOD. This module pre-computes the missing ``s_k``
arrays once, in parallel chunks, so each LOD reads pre-built
properly-chunked data.

Output layout: ``{output_directory}/_intermediate_scales.zarr`` is an
OME-NGFF multiscales group (0.4 in zarr v2, 0.5 in zarr v3) containing
``s0`` (symlinked to the source on local filesystems; copy or skip for
remote) plus ``s1..s_n``. Levels are created, placed and written through
worldarray: level ``k`` covers s0 voxels from the aligned ROI origin on,
and its OME-NGFF translation says so.

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
  it. Where that reaches past the dataset, the cube is padded with 0.
  No data loss.

Every block is read whole (padded past the dataset) so that each level
write covers whole chunks: worldarray only writes whole chunks without
coordination, which is what lets many dask workers fill one level.
"""

from __future__ import annotations

import logging
import os
import shutil

import numpy as np
import worldarray as wa

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
    is at or below ``anisotropy_tolerance * min(voxel_size)`` (the
    schedule of :func:`worldarray.level_factors`, which gives the
    per-step factors this compounds).

    Returns a list of cumulative (Fz, Fy, Fx) factors per LOD, including
    LOD 0 which is always (1, 1, 1).

    Example: voxel_size=[8, 8, 20], num_lods=4
        LOD 0: (1, 1, 1)  voxel=[8, 8, 20]
        LOD 1: (2, 2, 1)  voxel=[16, 16, 20]  → near-isotropic
        LOD 2: (4, 4, 2)  voxel=[32, 32, 40]
        LOD 3: (8, 8, 4)  voxel=[64, 64, 80]
    """
    out = [(1, 1, 1)]
    steps = wa.level_factors(
        [float(v) for v in voxel_size_zyx], anisotropy_tolerance,
        max_levels=max(0, num_lods - 1),
    )
    for step in steps:
        out.append(tuple(int(f * s) for f, s in zip(out[-1], step)))
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
# Level placement
# ---------------------------------------------------------------------------


def level_placements(
    s0_voxel_size_zyx, s0_translation_zyx, per_lod_factors, origin_voxels=0,
) -> list[tuple[tuple[float, ...], tuple[float, ...]]]:
    """``(voxel_size, translation)`` of each level, translation being
    voxel 0's centre as in OME-NGFF.

    Level ``k``'s voxel ``j`` covers s0 voxels ``origin + j * F_k`` to
    ``origin + (j + 1) * F_k - 1``, where ``origin`` is the aligned ROI
    origin in s0 voxels, so each level is placed from that origin rather
    than from s0's voxel 0 (:func:`worldarray.level_placement`).
    """
    origin = tuple(int(o) for o in np.broadcast_to(origin_voxels, (3,)))
    return [
        wa.level_placement(s0_voxel_size_zyx, s0_translation_zyx, factor, origin)
        for factor in per_lod_factors
    ]


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
# Dask-cluster worker: runs ONE block task in a dask worker process.
# Module-level so the function reference is picklable for dask.
# ---------------------------------------------------------------------------


# Per-worker-process cache: worldarray handles, keyed by path.
# Open is non-trivial (info-file fetch) — caching saves time across the
# many tasks a single dask worker processes.
_PYRAMID_WORKER_TS_CACHE: dict[str, object] = {}


def _downsample_func(method):
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
    return dispatch[method]


def _ts_handle_for_input(dataset_path):
    if dataset_path in _PYRAMID_WORKER_TS_CACHE:
        return _PYRAMID_WORKER_TS_CACHE[dataset_path]
    from mesh_n_bone.util.image_data_interface import open_volume
    handle = open_volume(dataset_path)
    _PYRAMID_WORKER_TS_CACHE[dataset_path] = handle
    return handle


def _open_level(path):
    """Open a level this pyramid created, for reading and writing. Not
    for arbitrary external arrays — see ``_ts_handle_for_input`` for
    those."""
    cache_key = ("level", path)
    if cache_key in _PYRAMID_WORKER_TS_CACHE:
        return _PYRAMID_WORKER_TS_CACHE[cache_key]
    # Cap thread pools + disable decoded-chunk cache. Without this
    # the output handle's cache_pool accumulates encoded chunks for
    # the worker's lifetime, OOMing on large super-chunk runs.
    handle = wa.open(path, mode="r+", units=None, context=wa.capped_context())
    _PYRAMID_WORKER_TS_CACHE[cache_key] = handle
    return handle


def _read_s0_box(worker_config, begin, end):
    """Read s0 voxels ``[begin, end)`` of the source dataset in the
    meshify worker's frame; the part past the dataset is 0."""
    from funlib.geometry import Coordinate, Roi
    from mesh_n_bone.util.image_data_interface import read_block

    ds_offset = Coordinate(worker_config["dataset_offset"])
    voxel_size = Coordinate(worker_config["voxel_size"])
    begin, end = np.asarray(begin, dtype=np.int64), np.asarray(end, dtype=np.int64)
    roi = Roi(
        ds_offset + Coordinate(begin.tolist()) * voxel_size,
        Coordinate((end - begin).tolist()) * voxel_size,
    )
    s0 = _ts_handle_for_input(worker_config["s0_dataset_path"])
    return read_block(s0, roi, voxel_size, ds_offset, fill_value=0)


def _release_memory():
    """Force the allocator to hand freed pages back. Without this the
    worker accumulates read residue across the 8+ blocks it processes
    before exit (cached handles in _PYRAMID_WORKER_TS_CACHE don't release
    on their own, and glibc malloc doesn't munmap freed pages without an
    explicit trim)."""
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def process_super_chunk_for_dask(block, worker_config):
    """Process ONE pyramid super-chunk inside a dask worker process.

    Module-level + serialization-safe so dask can ship this to LSF workers.
    ``block`` is a :class:`worldarray.Block` of the s0 grid (its write box
    is the super-chunk, in s0 voxels); ``worker_config`` is a plain dict
    of strings/tuples/numbers (no numpy arrays, no closures, no opened
    handles).

    Reads its s0 super-chunk, downsamples to each missing LOD via the
    configured ``downsample_method``, writes each LOD's chunks to the
    pre-created output arrays under ``worker_config["pyramid_path"]``.

    Returns ``None`` (dask bag collects None values; we don't aggregate).
    """
    downsample_func = _downsample_func(worker_config["downsample_method"])
    factors_per_lod = [tuple(f) for f in worker_config["factors_per_lod"]]
    out_origin = np.array(worker_config["out_origin"], dtype=np.int64)

    s0_block = _read_s0_box(worker_config, block.begin, block.end)

    # Downsample + write each LOD's chunks (streaming: drop the buffer after write)
    for k in worker_config["missing_lods"]:
        f = np.asarray(factors_per_lod[k], dtype=np.int64)
        ds_block, _ = downsample_func(s0_block, tuple(f.tolist()))
        begin = (np.asarray(block.begin) - out_origin) // f
        out_arr = _open_level(os.path.join(worker_config["pyramid_path"], f"s{k}"))
        out_arr.write_index(begin, begin + np.asarray(ds_block.shape), ds_block)
        del ds_block

    del s0_block
    _release_memory()
    return None


def process_cascade_chunk_for_dask(block, worker_config):
    """Process ONE cascade block in a dask worker process.

    ``block`` is a :class:`worldarray.Block` of the level being written,
    one output chunk. It is downsampled by a single per-axis
    ``step_factor`` from the immediately preceding pyramid level
    (``worker_config["read_path"]``), or, for the first level of a chain,
    from the original source s0 starting at the aligned ROI origin
    (``worker_config["s0_dataset_path"]`` and ``out_origin``). Either read
    is padded with 0 past its array, so the write is always the whole
    chunk.

    Module-level so the function reference is picklable for dask.
    """
    downsample_func = _downsample_func(worker_config["downsample_method"])
    step = np.asarray(worker_config["step_factor"], dtype=np.int64)
    begin = np.asarray(block.begin, dtype=np.int64) * step
    end = np.asarray(block.end, dtype=np.int64) * step

    if "read_path" in worker_config:
        data = _open_level(worker_config["read_path"]).read_index(begin, end, padding=0)
    else:
        out_origin = np.asarray(worker_config["out_origin"], dtype=np.int64)
        data = _read_s0_box(worker_config, out_origin + begin, out_origin + end)

    ds_block, _ = downsample_func(data, tuple(step.tolist()))
    _open_level(worker_config["write_path"]).write_index(block.begin, block.end, ds_block)
    del data, ds_block

    _release_memory()
    return None


# zstd level 3 for pyramid levels. Label volumes are mostly background,
# so even a low level shrinks 128^3 uint16 chunks ~100x (4 MiB -> tens
# of KiB). That matters both for the NRS writes/reads while building and
# for browsing the kept pyramid over HTTP in neuroglancer. zarr v3 uses
# the zstd codec; zarr v2 uses blosc with cname="zstd", the v2
# compressor every zarr reader (zarr-python, tensorstore, neuroglancer)
# supports.
_PYRAMID_CODECS = {
    "zarr3": "zstd",
    "zarr2": {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 1},
}


def prepare_pyramid_metadata_and_arrays(
    *,
    output_zarr_path,
    factors_per_lod,
    missing_lods,
    out_origin,
    out_shape,
    out_chunk_shape_voxels,
    dtype,
    s0_voxel_size_zyx,
    s0_translation_zyx,
    zarr_format=2,
    units="nanometer",
    s0_source_path=None,
):
    """Driver-side setup before dispatching pyramid blocks to dask.

    Creates an empty output array for each missing LOD, covering the
    aligned ROI ``[out_origin, out_origin + out_shape)`` of s0 and
    placed there, with its OME-NGFF multiscales entry; optionally
    symlinks s0 and lists it too. Returns ``{lod: WorldArray}``, each
    opened for writing.
    """
    fmt = "zarr3" if zarr_format == 3 else "zarr2"
    out_shape = np.asarray(out_shape, dtype=np.int64)
    placements = level_placements(
        s0_voxel_size_zyx, s0_translation_zyx, factors_per_lod, out_origin,
    )
    levels = {}
    for k in missing_lods:
        f = np.asarray(factors_per_lod[k], dtype=np.int64)
        ds_path = os.path.join(output_zarr_path, f"s{k}")
        # A level left by an earlier run (never s0, which may be a
        # symlink to the user's input).
        if os.path.isdir(ds_path) and not os.path.islink(ds_path):
            shutil.rmtree(ds_path)
        voxel_size, translation = placements[k]
        levels[k] = wa.create(
            ds_path, tuple(((out_shape + f - 1) // f).tolist()), np.dtype(dtype),
            voxel_size, translation, units=units,
            chunks=tuple(int(c) for c in out_chunk_shape_voxels),
            codecs=_PYRAMID_CODECS[fmt], format=fmt,
        )

    if s0_source_path is not None and _try_symlink_s0(
        output_zarr_path, s0_source_path, zarr_format=zarr_format,
    ):
        # s0 is the whole source array, placed as the source is.
        wa.ensure_group_entry(
            output_zarr_path, "s0", list(s0_voxel_size_zyx),
            list(s0_translation_zyx), axes=("z", "y", "x"), units=units,
        )
    return levels


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _padded_reader(s0_reader, dataset_shape, dtype):
    """``read(begin, end)`` over ``s0_reader``: s0 voxels ``[begin,
    end)``, with the part past the dataset filled with 0."""
    def read(begin, end):
        begin = np.asarray(begin, dtype=np.int64)
        end = np.asarray(end, dtype=np.int64)
        lo = np.clip(begin, 0, dataset_shape)
        hi = np.clip(end, 0, dataset_shape)
        out = np.zeros(tuple((end - begin).tolist()), dtype=dtype)
        if np.all(hi > lo):
            out[tuple(slice(a, b) for a, b in zip(lo - begin, hi - begin))] = (
                s0_reader(lo, hi - lo)
            )
        return out
    return read


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
    units: str = "nanometer",
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

    # Create the placed, empty level arrays (and link + list s0)
    out_chunk = np.asarray(out_chunk_shape_voxels, dtype=np.int64)
    out_arrays = prepare_pyramid_metadata_and_arrays(
        output_zarr_path=output_zarr_path,
        factors_per_lod=factors_per_lod,
        missing_lods=[k for k, _ in missing],
        out_origin=out_origin,
        out_shape=out_shape,
        out_chunk_shape_voxels=out_chunk.tolist(),
        dtype=dtype,
        s0_voxel_size_zyx=voxel_size.tolist(),
        s0_translation_zyx=translation.tolist(),
        zarr_format=zarr_format,
        units=units,
        s0_source_path=s0_source_path,
    )
    read_s0 = _padded_reader(s0_reader, dataset_shape, np.dtype(dtype))

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
        # single s0 read. The grid may reach past the dataset (halo).
        super_chunk_shape = out_chunk * max_factor
        sc_grid = wa.blocks(
            tuple(dataset_shape.tolist()), tuple(super_chunk_shape.tolist()),
            box=(tuple(out_origin.tolist()), tuple((out_origin + out_shape).tolist())),
        )

        def _process_super_chunk(block):
            s0_block = read_s0(block.begin, block.end)

            def _write_one(k, ds_block, _ds_origin):
                if k not in out_arrays:
                    return
                f = np.asarray(factors_per_lod[k], dtype=np.int64)
                begin = (np.asarray(block.begin) - out_origin) // f
                out_arrays[k].write_index(begin, begin + np.asarray(ds_block.shape), ds_block)

            # Stream each LOD's output: compute → write → drop the buffer.
            # Peak memory per task ≈ s0_block + one_lod_output.
            downsample_super_chunk(
                s0_block, np.asarray(block.begin), factors_per_lod,
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
        # Each pass tiles the level being written by its chunks.
        prev_k, prev_factor = None, (1, 1, 1)
        for k, factor in missing:
            read_from_s0 = not (prev_k is not None and k == prev_k + 1)
            step = factor if read_from_s0 else tuple(
                int(a // b) for a, b in zip(factor, prev_factor)
            )
            step_arr = np.asarray(step, dtype=np.int64)
            out_arr = out_arrays[k]

            if read_from_s0:
                def read(begin, end):
                    return read_s0(out_origin + begin, out_origin + end)
            else:
                # The level this call wrote one step ago.
                def read(begin, end, prev_arr=out_arrays[prev_k]):
                    return prev_arr.read_index(begin, end, padding=0)

            def _process_cascade_chunk(block, read=read, step_arr=step_arr,
                                       out_arr=out_arr):
                begin = np.asarray(block.begin, dtype=np.int64) * step_arr
                end = np.asarray(block.end, dtype=np.int64) * step_arr
                ds_block, _ = downsample_func(read(begin, end), tuple(step_arr.tolist()))
                out_arr.write_index(block.begin, block.end, ds_block)

            _run_chunk_grid(
                out_arr.blocks(), _process_cascade_chunk,
                f"s{k} from {'s0' if read_from_s0 else f's{prev_k}'}, step={step}",
            )
            prev_k, prev_factor = k, factor

    logger.info(
        "pyramid_builder: built %d new scales at %s (factors=%s)",
        len(missing), output_zarr_path,
        [factor for _, factor in missing],
    )
    return output_zarr_path


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
    # prepare_pyramid_metadata_and_arrays). If the real source array is in
    # a DIFFERENT format, symlinking it in as s0 produces a group that
    # LOOKS complete but is format-inconsistent: generic OME-zarr
    # multiscale readers (e.g. neuroglancer) resolve the group's declared
    # format and then look for THAT format's array metadata inside s0 —
    # which has the other format's metadata file instead — and fail with
    # something like "zarr v{version} array metadata not found", even
    # though s0 is perfectly readable on its own (e.g. with an explicit
    # driver override). s1+ are unaffected since those really do match.
    # Leave s0 absent rather than link in something unreadable through
    # the group's declared format — mesh-n-bone's own LOD-reading logic
    # never depends on this symlink anyway (LOD 0 always reads the real
    # source path directly); it's a convenience for external tools only.
    # (Callers should normally have already matched ``zarr_format`` to
    # the source's own format — see Meshify._build_missing_pyramid_scales
    # — so this is mostly a safety net for n5/precomputed sources, which
    # can't be represented as a zarr array at all.)
    expected_format = "zarr3" if zarr_format == 3 else "zarr2"
    try:
        actual_format = wa.detect(s0_source_path).format
        if actual_format != expected_format:
            logger.warning(
                "pyramid_builder: not symlinking s0 from %s into %s — "
                "source format (%s) doesn't match this pyramid's "
                "zarr_format=%d (%s). s0 stays absent from the pyramid; "
                "it's still directly browsable at its own real path/URL.",
                s0_source_path, target, actual_format, zarr_format, expected_format,
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
