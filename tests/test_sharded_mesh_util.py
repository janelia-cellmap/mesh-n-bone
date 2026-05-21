"""Unit tests for `mesh_n_bone.util.sharded_mesh_util`.

These exercise the sharded multi-resolution Draco mesh writer without
running the full Meshify pipeline: we synthesize fake manifest + fragment
files, pack them, then read them back per the Neuroglancer spec and
verify every byte lands where it should.
"""

import json
import os
import struct

import numpy as np
import pytest

from mesh_n_bone.util import sharded_mesh_util


# ---------- helpers ----------

def _fake_manifest_bytes(num_lods: int = 1) -> bytes:
    """Minimal valid multi-resolution mesh manifest."""

    chunk_shape = np.array([16.0, 16.0, 16.0], dtype="<f4").tobytes()
    grid_origin = np.zeros(3, dtype="<f4").tobytes()
    header = struct.pack("<I", num_lods)
    lod_scales = np.array([2 ** i for i in range(num_lods)], dtype="<f4").tobytes()
    vertex_offsets = np.zeros((num_lods, 3), dtype="<f4").tobytes(order="C")
    num_frags = np.array([1] * num_lods, dtype="<u4").tobytes()
    positions = np.zeros((3, 1), dtype="<u4").tobytes(order="C") * num_lods
    sizes = (np.array([8], dtype="<u4").tobytes()) * num_lods
    return (
        chunk_shape + grid_origin + header + lod_scales + vertex_offsets
        + num_frags + positions + sizes
    )


def _fake_fragment_bytes(seg_id: int, num_lods: int = 1) -> bytes:
    """`num_lods` × 8 bytes of "fragment data" tagged with the seg id."""

    return (seg_id.to_bytes(8, "little")) * num_lods


def _parse_manifest(buf: bytes):
    g = buf
    chunk_shape = np.frombuffer(g[:12], dtype="<f4"); g = g[12:]
    grid_origin = np.frombuffer(g[:12], dtype="<f4"); g = g[12:]
    (num_lods,) = struct.unpack("<I", g[:4]); g = g[4:]
    g = g[4 * num_lods :]                       # lod_scales
    g = g[12 * num_lods :]                      # vertex_offsets
    num_frags = np.frombuffer(g[: 4 * num_lods], dtype="<u4"); g = g[4 * num_lods:]
    sizes_per_lod = []
    for n in num_frags:
        n = int(n)
        g = g[4 * 3 * n :]                      # positions
        sizes_per_lod.append(np.frombuffer(g[: 4 * n], dtype="<u4"))
        g = g[4 * n :]
    return chunk_shape, grid_origin, sizes_per_lod


def _decode_minishard_index(msi_bytes: bytes, fixed_index_len: int):
    msi = (
        np.frombuffer(msi_bytes, dtype=np.uint64)
        .reshape(3, -1).T.copy()
    )
    for i in range(1, msi.shape[0]):
        msi[i, 0] += msi[i - 1, 0]
        msi[i, 1] += msi[i - 1, 1] + msi[i - 1, 2]
    msi[:, 1] += fixed_index_len
    return msi


def _stage_fake_meshes(multires_dir: str, seg_ids, num_lods: int = 1):
    os.makedirs(multires_dir, exist_ok=True)
    expected = {}
    for sid in seg_ids:
        manifest = _fake_manifest_bytes(num_lods)
        fragment = _fake_fragment_bytes(sid, num_lods)
        with open(os.path.join(multires_dir, f"{sid}.index"), "wb") as f:
            f.write(manifest)
        with open(os.path.join(multires_dir, str(sid)), "wb") as f:
            f.write(fragment)
        expected[sid] = (fragment, manifest)
    return expected


# ---------- tests ----------

class TestChooseShardParams:
    @pytest.mark.parametrize(
        "n",
        [1, 10, 100, 1_000, 9_000, 100_000, 1_000_000, 10_000_000],
    )
    def test_params_are_sane(self, n):
        p = sharded_mesh_util.choose_shard_params(n)
        assert p["preshift_bits"] == 0
        assert 0 <= p["shard_bits"] <= 6
        assert 0 <= p["minishard_bits"] <= 10
        assert p["shard_bits"] + p["minishard_bits"] <= 16  # total cap


class TestShardingSpec:
    def test_make_spec_round_trips_through_dict(self):
        spec = sharded_mesh_util.make_sharding_spec(0, 6, 2)
        d = spec.to_dict()
        assert d["@type"] == "neuroglancer_uint64_sharded_v1"
        assert int(d["preshift_bits"]) == 0
        assert int(d["minishard_bits"]) == 6
        assert int(d["shard_bits"]) == 2
        assert d["minishard_index_encoding"] == "raw"
        assert d["data_encoding"] == "raw"

    def test_hashfn_is_unsigned(self):
        """Cloud-volume's stock hashfn raises OverflowError on negative
        signed values under newer numpy — make sure ours doesn't."""
        spec = sharded_mesh_util.make_sharding_spec(0, 4, 2)
        # Compute hash for a sample of IDs; any one of these tripped the bug.
        for sid in (1, 12345, 999_999, 2**62):
            loc = spec.compute_shard_location(sid)
            assert 0 <= int(loc.shard_number, 16) < 2**2
            assert 0 <= int(loc.minishard_number) < 2**4

    def test_distribution_roughly_uniform(self):
        spec = sharded_mesh_util.make_sharding_spec(0, 6, 2)
        counts = {}
        for sid in range(1, 10_001):
            loc = spec.compute_shard_location(sid)
            counts[loc.shard_number] = counts.get(loc.shard_number, 0) + 1
        # 4 shards expected; each should get ~2500 ± reasonable slack.
        assert set(counts) == {format(i, "x") for i in range(4)}
        assert all(2000 < c < 3000 for c in counts.values())


class TestGroupByShard:
    def test_grouping_partitions_all_ids(self):
        spec = sharded_mesh_util.make_sharding_spec(0, 2, 1)
        ids = [1, 2, 3, 100, 999, 12345]
        grouping = sharded_mesh_util.group_segment_ids_by_shard(spec, ids)
        assert sorted(sum(grouping.values(), [])) == sorted(ids)
        # Two shard files possible (shard_bits=1).
        assert set(grouping) <= {"0", "1"}


class TestPackOneShard:
    def test_writes_shard_with_correct_layout(self, tmp_output_dir):
        """Pack two segments, then read back and verify:
          - shard file exists
          - minishard index points at the manifest bytes we wrote
          - the fragment data sits immediately before the manifest
        """
        multires = os.path.join(tmp_output_dir, "multires")
        expected = _stage_fake_meshes(multires, [1, 2])
        spec = sharded_mesh_util.make_sharding_spec(0, 2, 0)
        grouping = sharded_mesh_util.group_segment_ids_by_shard(spec, [1, 2])
        shard_name, shard_ids = next(iter(grouping.items()))

        out_path = sharded_mesh_util.pack_one_shard(
            shard_name, shard_ids, multires, multires, spec.to_dict(),
        )
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0

        with open(out_path, "rb") as f:
            shard = f.read()

        fixed_index_len = int(spec.index_length())
        fixed = np.frombuffer(
            shard[:fixed_index_len], dtype=np.uint64
        ).reshape(-1, 2)
        for sid in shard_ids:
            loc = spec.compute_shard_location(sid)
            msi_s = int(fixed[int(loc.minishard_number), 0])
            msi_e = int(fixed[int(loc.minishard_number), 1])
            msi_bytes = shard[
                fixed_index_len + msi_s : fixed_index_len + msi_e
            ]
            msi = _decode_minishard_index(msi_bytes, fixed_index_len)
            row = next(r for r in msi if int(r[0]) == sid)
            offset, size = int(row[1]), int(row[2])
            fragment, manifest = expected[sid]
            # Minishard index points at the manifest.
            assert shard[offset : offset + size] == manifest
            # Fragment data sits immediately before.
            assert shard[offset - len(fragment) : offset] == fragment

    def test_handles_missing_index_file_gracefully(self, tmp_output_dir):
        """If a segment has no .index file, it should be silently skipped."""
        multires = os.path.join(tmp_output_dir, "multires")
        os.makedirs(multires)
        # Stage only seg 1; reference both in the pack call.
        _stage_fake_meshes(multires, [1])
        spec = sharded_mesh_util.make_sharding_spec(0, 2, 0)
        path = sharded_mesh_util.pack_one_shard(
            "0", [1, 2], multires, multires, spec.to_dict(),
        )
        assert os.path.exists(path)


class TestInfoFile:
    def test_info_serializes_cleanly(self, tmp_output_dir):
        spec = sharded_mesh_util.make_sharding_spec(0, 6, 2)
        sharded_mesh_util.write_sharded_info_file(tmp_output_dir, spec)
        with open(os.path.join(tmp_output_dir, "info")) as f:
            info = json.load(f)
        assert info["@type"] == "neuroglancer_multilod_draco"
        assert info["sharding"]["@type"] == "neuroglancer_uint64_sharded_v1"
        # Plain ints (not numpy types) so the file is portable.
        assert isinstance(info["sharding"]["minishard_bits"], int)
        assert isinstance(info["sharding"]["shard_bits"], int)

    def test_default_quantization_matches_encoder(self, tmp_output_dir):
        """Default `vertex_quantization_bits` must match the encoder default
        (16) and the unsharded `write_info_file` default (16). A mismatch
        decodes vertex coordinates at the wrong scale and meshes render
        stretched or shrunk by a power of 2.
        """
        from mesh_n_bone.util import neuroglancer as ng_util
        spec = sharded_mesh_util.make_sharding_spec(0, 6, 2)
        sharded_mesh_util.write_sharded_info_file(tmp_output_dir, spec)
        with open(os.path.join(tmp_output_dir, "info")) as f:
            sharded_info = json.load(f)

        unsharded_dir = os.path.join(tmp_output_dir, "unsharded")
        os.makedirs(unsharded_dir)
        ng_util.write_info_file(unsharded_dir)
        with open(os.path.join(unsharded_dir, "info")) as f:
            unsharded_info = json.load(f)

        assert (
            sharded_info["vertex_quantization_bits"]
            == unsharded_info["vertex_quantization_bits"]
        ), "sharded and unsharded info files must declare the same quantization"


class TestPackMeshesToShards:
    def test_dask_path_matches_direct(self, tmp_output_dir):
        """The dask orchestrator should write byte-identical shards to the
        direct per-shard call."""
        from dask.distributed import Client, LocalCluster

        seg_ids = [1, 2, 3, 4, 5, 100, 999, 12345]
        a = os.path.join(tmp_output_dir, "direct")
        b = os.path.join(tmp_output_dir, "dask")
        _stage_fake_meshes(a, seg_ids)
        _stage_fake_meshes(b, seg_ids)

        spec = sharded_mesh_util.make_sharding_spec(0, 2, 1)

        # Direct
        for shard_name, ids in sharded_mesh_util.group_segment_ids_by_shard(
            spec, seg_ids
        ).items():
            sharded_mesh_util.pack_one_shard(
                shard_name, ids, a, a, spec.to_dict(),
            )

        # Dask
        cluster = LocalCluster(
            n_workers=2, processes=True, threads_per_worker=1,
            dashboard_address=None,
        )
        client = Client(cluster)
        try:
            sharded_mesh_util.pack_meshes_to_shards(b, seg_ids, spec, num_workers=2)
        finally:
            client.close()
            cluster.close()

        a_shards = sorted(n for n in os.listdir(a) if n.endswith(".shard"))
        b_shards = sorted(n for n in os.listdir(b) if n.endswith(".shard"))
        assert a_shards == b_shards
        for name in a_shards:
            with open(os.path.join(a, name), "rb") as fa, open(
                os.path.join(b, name), "rb"
            ) as fb:
                assert fa.read() == fb.read(), f"shard {name} differs"


class TestDeleteUnsharded:
    def test_removes_per_segment_files(self, tmp_output_dir):
        multires = os.path.join(tmp_output_dir, "multires")
        _stage_fake_meshes(multires, [1, 2, 3])
        # Sanity: files exist before.
        for sid in (1, 2, 3):
            assert os.path.exists(os.path.join(multires, str(sid)))
            assert os.path.exists(os.path.join(multires, f"{sid}.index"))
        sharded_mesh_util.delete_unsharded_segment_files(multires, [1, 2, 3])
        for sid in (1, 2, 3):
            assert not os.path.exists(os.path.join(multires, str(sid)))
            assert not os.path.exists(os.path.join(multires, f"{sid}.index"))

    def test_missing_files_do_not_raise(self, tmp_output_dir):
        # Should be a no-op when nothing is present.
        sharded_mesh_util.delete_unsharded_segment_files(tmp_output_dir, [42])
