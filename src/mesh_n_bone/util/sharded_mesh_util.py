"""Pack unsharded Neuroglancer multi-resolution Draco meshes into the
sharded precomputed format.

The unsharded writer produces two files per segment in `multires/`:
  - `<id>`       : concatenated Draco fragments (fragment data)
  - `<id>.index` : binary manifest

The sharded format stores all segments in a small number of `<shard>.shard`
files. For each chunk the layout is `<fragment_data> || <manifest>`, and the
minishard index points at the manifest only (size = len(manifest)). The
reader uses the manifest to locate the fragment block, which lives in the
shard file immediately before the manifest.

We use cloud-volume's `ShardingSpecification` for the hash / index math and
`synthesize_shard_file` to assemble each shard. Parallelism is driven by
dask (one task per shard).
"""

import json
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import dask.bag as db
import numpy as np
from cloudvolume.datasource.precomputed import mmh3 as _cv_mmh3
from cloudvolume.datasource.precomputed.sharding import (
    ShardingSpecification,
    synthesize_shard_file,
)
from dask.distributed import wait


def _murmur3_x86_128_unsigned(value: int) -> np.uint64:
    """Compute murmurhash3_x86_128 low 64 bits as an unsigned numpy uint64.

    Cloud-volume's stock hashfn does `uint64(hash64(...)[0])`, but its bundled
    pure-Python hash64 returns a signed int (negative for high-bit-set values),
    and `np.uint64(negative_int)` raises OverflowError under newer numpy. Mask
    to 64 bits before the conversion.
    """

    signed = _cv_mmh3.hash64(np.uint64(int(value)).tobytes(), x64arch=False)[0]
    return np.uint64(signed & 0xFFFFFFFFFFFFFFFF)


def choose_shard_params(num_segments: int) -> Dict[str, int]:
    """Pick reasonable preshift/minishard/shard bits for `num_segments`.

    Aims for roughly ~32 segments per minishard and a handful of shards.
    Returns a dict with preshift_bits, minishard_bits, shard_bits.
    """

    n = max(1, int(num_segments))
    # Total bins ~= n / 32, split between shards and minishards.
    total_bits = max(0, int(math.ceil(math.log2(max(1, n / 32.0)))))
    # Cap each below 16 so a single shard does not balloon.
    shard_bits = min(total_bits // 2, 6)
    minishard_bits = max(0, min(total_bits - shard_bits, 10))
    return {
        "preshift_bits": 0,
        "minishard_bits": int(minishard_bits),
        "shard_bits": int(shard_bits),
    }


def make_sharding_spec(
    preshift_bits: int = 0,
    minishard_bits: int = 6,
    shard_bits: int = 2,
    hash_fn: str = "murmurhash3_x86_128",
) -> ShardingSpecification:
    """Build a ShardingSpecification with raw encodings (draco is pre-compressed)."""

    spec = ShardingSpecification(
        type="neuroglancer_uint64_sharded_v1",
        preshift_bits=int(preshift_bits),
        minishard_bits=int(minishard_bits),
        shard_bits=int(shard_bits),
        hash=hash_fn,
        minishard_index_encoding="raw",
        data_encoding="raw",
    )
    if hash_fn == "murmurhash3_x86_128":
        spec.hashfn = _murmur3_x86_128_unsigned
    return spec


def group_segment_ids_by_shard(
    spec: ShardingSpecification, segment_ids: Iterable[int]
) -> Dict[str, List[int]]:
    """Return {shard_filename_base: [segment_id, ...]} for the given spec."""

    grouping: Dict[str, List[int]] = defaultdict(list)
    for seg_id in segment_ids:
        loc = spec.compute_shard_location(int(seg_id))
        grouping[str(loc.shard_number)].append(int(seg_id))
    return grouping


def _read_segment_chunk(
    multires_dir: str, seg_id: int
) -> Tuple[bytes, int] | None:
    """Read `<id>` + `<id>.index`, return (chunk_binary, manifest_size).

    Returns None if the segment has no fragments (no `<id>.index` written).
    """

    index_path = os.path.join(multires_dir, f"{seg_id}.index")
    frag_path = os.path.join(multires_dir, str(seg_id))
    if not os.path.exists(index_path):
        return None
    with open(index_path, "rb") as f:
        manifest_bytes = f.read()
    if os.path.exists(frag_path):
        with open(frag_path, "rb") as f:
            fragment_bytes = f.read()
    else:
        fragment_bytes = b""
    return fragment_bytes + manifest_bytes, len(manifest_bytes)


def pack_one_shard(
    shard_name: str,
    segment_ids: List[int],
    multires_dir: str,
    output_dir: str,
    spec_dict: Dict,
) -> str:
    """Assemble a single `<shard_name>.shard` file from its segments.

    Designed to run on a dask worker. Re-creates the ShardingSpecification
    from a JSON-able dict so it ships cleanly across processes.
    """

    spec = ShardingSpecification.from_dict(spec_dict)
    if spec.hash == "murmurhash3_x86_128":
        spec.hashfn = _murmur3_x86_128_unsigned

    data: Dict[int, bytes] = {}
    data_offset: Dict[int, int] = {}
    for seg_id in segment_ids:
        chunk = _read_segment_chunk(multires_dir, seg_id)
        if chunk is None:
            continue
        binary, manifest_size = chunk
        data[seg_id] = binary
        data_offset[seg_id] = manifest_size

    if not data:
        # Empty shard: still write a header-only file so reads don't 404.
        # synthesize_shard_file handles the empty case via its fixed index.
        pass

    shard_bytes = synthesize_shard_file(
        spec, data, data_offset=data_offset, progress=False, presorted=False
    )
    out_path = os.path.join(output_dir, f"{shard_name}.shard")
    with open(out_path, "wb") as f:
        f.write(shard_bytes)
    return out_path


def write_sharded_info_file(
    path: str, spec: ShardingSpecification, vertex_quantization_bits: int = 16
) -> None:
    """Write the top-level `info` for sharded multi-resolution Draco meshes.

    Default of 16 matches the encoder default in
    `mesh_n_bone.multires.multires.generate_neuroglancer_multires_mesh` and
    the unsharded `write_info_file`. The value MUST equal whatever the
    encoder used, or Neuroglancer will decode vertices at the wrong scale
    and the mesh geometry will appear stretched or shrunk by a power of 2.
    """

    # spec.to_dict() may contain numpy uint64 — coerce to plain ints for JSON.
    sharding = {
        k: (int(v) if isinstance(v, (np.integer,)) else v)
        for k, v in spec.to_dict().items()
    }
    info = {
        "@type": "neuroglancer_multilod_draco",
        "vertex_quantization_bits": int(vertex_quantization_bits),
        "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        "lod_scale_multiplier": 1,
        "segment_properties": "segment_properties",
        "sharding": sharding,
    }
    with open(os.path.join(path, "info"), "w") as f:
        json.dump(info, f)


def pack_meshes_to_shards(
    multires_dir: str,
    segment_ids: Iterable[int],
    spec: ShardingSpecification,
    num_workers: int,
    output_dir: str | None = None,
) -> List[str]:
    """Group segments by shard, dispatch one dask task per shard, return written paths."""

    output_dir = output_dir or multires_dir
    os.makedirs(output_dir, exist_ok=True)

    grouping = group_segment_ids_by_shard(spec, segment_ids)
    spec_dict = spec.to_dict()

    shard_items = list(grouping.items())  # [(shard_name, [ids]), ...]
    if not shard_items:
        return []

    bag = db.from_sequence(shard_items, npartitions=min(len(shard_items), max(1, num_workers)))
    bag = bag.map(
        lambda item: pack_one_shard(item[0], item[1], multires_dir, output_dir, spec_dict)
    )

    if num_workers == 1:
        # Synchronous scheduler — no distributed Client exists (see
        # `dask_util.start_dask`), so use `.compute()` rather than the
        # `persist()`+`wait()` Client-based path.
        return list(bag.compute())

    futures = bag.persist()
    [completed, _] = wait(futures)
    failed = [f for f in completed if f.exception() is not None]
    paths = [f.result() for f in completed if f.exception() is None]
    for c in completed:
        c.cancel()
    if failed:
        raise RuntimeError(f"Failed to pack {len(failed)} shards: {failed}")
    return paths


def delete_unsharded_segment_files(multires_dir: str, segment_ids: Iterable[int]) -> None:
    """Remove `<id>` and `<id>.index` after they've been packed into shards."""

    for seg_id in segment_ids:
        for suffix in ("", ".index"):
            p = os.path.join(multires_dir, f"{seg_id}{suffix}")
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
