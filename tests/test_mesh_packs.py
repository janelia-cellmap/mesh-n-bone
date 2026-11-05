"""Multires sharded output built from the meshing passes' pack files is
the same as the output built from one PLY per segment per level."""

import logging
import os

import numpy as np
import pytest
import zarr

from mesh_n_bone.meshify import meshify as meshify_module
from mesh_n_bone.meshify.meshify import Meshify


def _blob_volume(tmp_path):
    """48^3 voxels in 27 blocks of 16^3: small blobs (gone at coarse
    levels), blobs crossing block faces, one segment in two parts in
    different blocks, and a sphere that lasts through every level."""
    rng = np.random.default_rng(1)
    vol = np.zeros((48, 48, 48), dtype=np.uint32)
    label = 1
    for _ in range(120):
        size = rng.integers(2, 7, size=3)
        corner = rng.integers(0, 48 - size, size=3)
        region = tuple(slice(c, c + s) for c, s in zip(corner, size))
        if not vol[region].any():
            vol[region] = label
            label += 1
    vol[4:9, 4:9, 4:9] = 500
    vol[36:41, 36:41, 36:41] = 500
    zz, yy, xx = np.mgrid[:48, :48, :48]
    vol[((zz - 24) ** 2 + (yy - 24) ** 2 + (xx - 24) ** 2) < 9 ** 2] = 600
    root = zarr.open_group(str(tmp_path / "blobs.zarr"), mode="w")
    arr = root.create_array("labels/s0", data=vol, chunks=(16, 16, 16))
    arr.attrs["voxel_size"] = [8, 8, 8]
    arr.attrs["offset"] = [0, 0, 0]
    return str(tmp_path / "blobs.zarr" / "labels" / "s0")


def _multires(path, out):
    Meshify(
        input_path=path, output_directory=out, num_workers=1,
        do_multires=True, multires_strategy="downsample", num_lods=3,
        sharded=True, use_existing_scales=False,
        do_analysis=False, check_mesh_validity=False,
        read_write_block_shape_pixels=[16, 16, 16],
        target_reduction=0.5, n_smoothing_iter=2,
    ).get_meshes()
    contents = {}
    for root, _, files in os.walk(os.path.join(out, "multires")):
        for name in files:
            full = os.path.join(root, name)
            with open(full, "rb") as f:
                contents[os.path.relpath(full, out)] = f.read()
    return contents


@pytest.mark.parametrize("small_max_bytes, expect_large", [
    (1 << 20, False),  # every segment is small
    (3000, True),      # the sphere and the bigger blobs take the per-segment route
])
def test_shards_from_packs_match_shards_from_files(
    tmp_path, monkeypatch, caplog, small_max_bytes, expect_large,
):
    path = _blob_volume(tmp_path)
    monkeypatch.setattr(meshify_module, "SMALL_SEGMENT_MAX_BYTES", small_max_bytes)
    monkeypatch.setattr(meshify_module, "SMALL_MULTIRES_BATCH", 7)  # several batches per pack
    with caplog.at_level(logging.INFO):
        from_packs = _multires(path, str(tmp_path / "packs"))
    routes = [r.message for r in caplog.records if "small ones in batches from packs" in r.message]
    assert routes, "the pack route was not used"
    large = int(routes[0].split(", ")[1].split()[0])
    assert (large > 0) == expect_large, routes[0]

    monkeypatch.setattr(meshify_module, "USE_MESH_PACKS", False)
    from_files = _multires(path, str(tmp_path / "files"))

    assert sorted(from_packs) == sorted(from_files)
    assert any(name.endswith(".shard") for name in from_packs)
    differing = [name for name in from_files if from_packs[name] != from_files[name]]
    assert not differing, f"differ: {differing}"
    # Nothing per segment is left behind, and the level meshes are gone.
    assert all(
        name.endswith(".shard") or name in ("multires/info", "multires/segment_properties/info")
        for name in from_packs
    ), sorted(from_packs)
    assert not os.path.exists(tmp_path / "packs" / "mesh_lods")


class TestWhenPacksAreUsed:
    @staticmethod
    def _meshify(**overrides):
        m = object.__new__(Meshify)
        settings = dict(
            do_multires=True, sharded=True, multires_strategy="downsample",
            delete_decimated_meshes=True, delete_unsharded_files=True, do_analysis=False,
        )
        settings.update(overrides)
        for key, value in settings.items():
            setattr(m, key, value)
        return m

    def test_the_default_multires_run_uses_packs(self):
        assert self._meshify()._uses_mesh_packs()

    @pytest.mark.parametrize("override", [
        {"sharded": False},
        {"multires_strategy": "decimate"},
        {"delete_decimated_meshes": False},
        {"delete_unsharded_files": False},
        {"do_analysis": True},
        {"do_multires": False},
    ])
    def test_anything_that_reads_the_per_segment_files_keeps_them(self, override):
        assert not self._meshify(**override)._uses_mesh_packs()


def _index(rows):
    index = np.array(rows, dtype=meshify_module._FINISHED_INDEX_DTYPE)
    return index[np.argsort(index["id"], kind="stable")]


def test_small_segments_are_batched_by_their_lod0_pack():
    # LOD 0: ids 1-5 in pack a0, 6-7 in pack b0. LOD 1: some of them, in c1.
    s0 = _index([(i, 0, 10 * i, 5) for i in (1, 2, 3, 4, 5)] + [(i, 1, 10 * i, 5) for i in (6, 7)])
    s1 = _index([(2, 0, 0, 3), (6, 0, 3, 3)])
    indexes = {0: (["a0.pack", "b0.pack"], s0), 1: (["c1.pack"], s1)}
    batches = meshify_module._small_segment_batches(s0, indexes, [0, 1], batch_size=2)

    ids = [sorted(set(int(i) for i in rows["id"])) for _, rows, _ in batches]
    assert ids == [[1, 2], [3, 4], [5], [6, 7]]
    name, rows, paths = batches[0]
    assert name == "batch_0"
    assert paths == ["a0.pack", "c1.pack"]  # only the packs this batch reads
    assert [(int(r["id"]), int(r["lod"]), paths[int(r["path"])], int(r["offset"])) for r in rows] == [
        (1, 0, "a0.pack", 10), (2, 0, "a0.pack", 20), (2, 1, "c1.pack", 0),
    ]
    assert batches[3][2] == ["b0.pack", "c1.pack"]


def test_an_earlier_runs_packs_are_not_read(tmp_path, monkeypatch):
    """Finished-mesh packs and small-segment chunk batches an earlier,
    killed run left in the output directory are not taken as this run's."""
    from mesh_n_bone.multires.multires import SMALL_CHUNKS_SUBDIR

    path = _blob_volume(tmp_path)
    clean = _multires(path, str(tmp_path / "clean"))

    out = tmp_path / "reused"
    meshify_module._write_finished_pack(str(out / "mesh_lods" / "s0"), "block_0", [(424242, b"ply")])
    chunks = out / "multires" / SMALL_CHUNKS_SUBDIR / "_manifests"
    chunks.mkdir(parents=True)
    np.save(chunks / "batch_0.npy", np.zeros(0, dtype=[
        ("id", "u8"), ("offset", "u8"), ("bytes", "u8"), ("manifest_size", "u8"),
    ]))
    assert _multires(path, str(out)) == clean
