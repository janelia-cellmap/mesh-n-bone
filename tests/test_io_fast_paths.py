"""The per-segment file operations on shared filesystems: parallel deletes
with progress, the directory-rename promotion of assembled meshes, and the
block manifests that spare the assembly planner a stat per file."""

import logging
import os

import numpy as np
import pytest

from mesh_n_bone.meshify import meshify as mf
from mesh_n_bone.util import dask_util, sharded_mesh_util


class TestThreadPoolSizing:
    def test_grows_with_items_and_caps_at_a_multiple_of_workers(self):
        assert dask_util.local_io_workers(1, 48) == 1
        assert dask_util.local_io_workers(100, 48) == 5          # one thread per 20 items
        assert dask_util.local_io_workers(1_000_000, 48) == 192  # 48 * 4 cap
        assert dask_util.local_io_workers(1_000_000, 1) == 4


class TestParallelIo:
    def test_results_keep_input_order_and_progress_is_logged(self, caplog):
        items = list(range(100))
        with caplog.at_level(logging.INFO, logger="test.io"):
            out = dask_util.parallel_io(
                items, lambda x: x * 2, num_workers=8, logger=logging.getLogger("test.io"),
                what="things", report_fraction=0.25,
            )
        assert out == [x * 2 for x in items]
        progress = [r.message for r in caplog.records if r.message.startswith("things:")]
        assert progress == ["things: 25/100", "things: 50/100", "things: 75/100", "things: 100/100"]

    def test_small_inputs_run_inline(self):
        assert dask_util.parallel_io([1, 2, 3], lambda x: -x, num_workers=48) == [-1, -2, -3]
        assert dask_util.parallel_io([], lambda x: x, num_workers=48) == []

    def test_errors_propagate(self):
        def boom(x):
            raise RuntimeError("no")

        with pytest.raises(RuntimeError, match="no"):
            dask_util.parallel_io(list(range(200)), boom, num_workers=8)


class TestParallelDeleteTree:
    def test_removes_files_and_directories(self, tmp_path):
        root = tmp_path / "mesh_lods"
        for lod in range(3):
            d = root / f"s{lod}"
            d.mkdir(parents=True)
            for i in range(50):
                (d / f"{i}.ply").write_bytes(b"x")
        dask_util.parallel_delete_tree(str(root), num_workers=4)
        assert not root.exists()

    def test_symlinks_are_unlinked_not_followed(self, tmp_path):
        outside = tmp_path / "source"
        outside.mkdir()
        (outside / "keep.ply").write_bytes(b"keep")
        root = tmp_path / "tree"
        root.mkdir()
        (root / "s0").symlink_to(outside, target_is_directory=True)
        (root / "f.ply").symlink_to(outside / "keep.ply")
        (root / "s1").mkdir()
        (root / "s1" / "a.ply").write_bytes(b"a")
        dask_util.parallel_delete_tree(str(root), num_workers=4)
        assert not root.exists()
        assert (outside / "keep.ply").read_bytes() == b"keep"

    def test_a_symlinked_root_is_left_alone(self, tmp_path):
        outside = tmp_path / "source"
        outside.mkdir()
        (outside / "a.ply").write_bytes(b"a")
        link = tmp_path / "link"
        link.symlink_to(outside, target_is_directory=True)
        dask_util.parallel_delete_tree(str(link), num_workers=4)
        assert link.is_symlink() and (outside / "a.ply").exists()

    def test_missing_path_is_a_noop(self, tmp_path):
        dask_util.parallel_delete_tree(str(tmp_path / "nope"), num_workers=4)


class TestUnshardedDelete:
    def test_pairs_removed_in_parallel_and_missing_ones_tolerated(self, tmp_path, caplog):
        ids = list(range(1, 301))
        for i in ids[:250]:
            (tmp_path / str(i)).write_bytes(b"m")
            (tmp_path / f"{i}.index").write_bytes(b"i")
        (tmp_path / "0.shard").write_bytes(b"s")
        with caplog.at_level(logging.INFO, logger="test.shard"):
            sharded_mesh_util.delete_unsharded_segment_files(
                str(tmp_path), ids, num_workers=8, logger=logging.getLogger("test.shard"),
            )
        assert sorted(os.listdir(tmp_path)) == ["0.shard"]
        assert any("unsharded segment files deleted: 300/300" in r.message for r in caplog.records)


class TestPromoteMeshesSubdir:
    def test_meshes_move_up_with_two_renames_and_siblings_are_kept(self, tmp_path):
        s0 = tmp_path / "s0"
        (s0 / "meshes").mkdir(parents=True)
        for i in range(20):
            (s0 / "meshes" / f"{i}.ply").write_bytes(b"p")
        (s0 / "assembly_metadata").mkdir()
        (s0 / "assembly_metadata" / "plan.json").write_text("{}")
        (s0 / "tmp_chunked").mkdir()
        mf._promote_meshes_subdir(str(s0))
        names = sorted(os.listdir(s0))
        assert "meshes" not in names
        assert {f"{i}.ply" for i in range(20)} <= set(names)
        assert (s0 / "assembly_metadata" / "plan.json").read_text() == "{}"
        assert (s0 / "tmp_chunked").is_dir()
        assert not (tmp_path / "s0.promoting").exists()

    def test_noop_without_a_meshes_subdir(self, tmp_path):
        s1 = tmp_path / "s1"
        s1.mkdir()
        (s1 / "1.ply").write_bytes(b"p")
        mf._promote_meshes_subdir(str(s1))
        assert sorted(os.listdir(s1)) == ["1.ply"]

    def test_a_name_clash_is_refused(self, tmp_path):
        s0 = tmp_path / "s0"
        (s0 / "meshes").mkdir(parents=True)
        (s0 / "meshes" / "1.ply").write_bytes(b"new")
        (s0 / "1.ply").write_bytes(b"old")
        with pytest.raises(FileExistsError):
            mf._promote_meshes_subdir(str(s0))


def _ply(n_vertices, n_faces):
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n_vertices}\nproperty float x\nproperty float y\nproperty float z\n"
        f"element face {n_faces}\nproperty list uchar int vertex_indices\nend_header\n"
    ).encode()
    return header + b"\0" * (12 * n_vertices + 13 * n_faces)


class TestBlockManifests:
    def test_round_trip_and_totals_across_blocks(self, tmp_path):
        mf._write_block_manifest(str(tmp_path), 0, [(7, 100, 10, 20), (8, 50, 5, 6)])
        mf._write_block_manifest(str(tmp_path), 1, [(7, 300, 30, 60)])
        mf._write_block_manifest(str(tmp_path), 2, [])  # a block that wrote nothing
        assert sorted(os.listdir(tmp_path / mf._MANIFEST_SUBDIR)) == ["block_0.npy", "block_1.npy"]
        totals = mf._manifest_totals(str(tmp_path))
        assert totals == {7: [2, 400, 40, 80], 8: [1, 50, 5, 6]}

    def test_scan_uses_manifests_and_falls_back_for_uncovered_segments(self, tmp_path, caplog):
        amplification = 32
        # Segment 7: two blocks, both recorded in manifests.
        (tmp_path / "7").mkdir()
        (tmp_path / "7" / "block_0.ply").write_bytes(_ply(10, 20))
        (tmp_path / "7" / "block_1.ply").write_bytes(_ply(30, 60))
        mf._write_block_manifest(str(tmp_path), 0, [(7, len(_ply(10, 20)), 10, 20)])
        mf._write_block_manifest(str(tmp_path), 1, [(7, len(_ply(30, 60)), 30, 60)])
        # Segment 9: its block died before writing a manifest; files only.
        (tmp_path / "9").mkdir()
        (tmp_path / "9" / "block_2.ply").write_bytes(_ply(4, 2))
        with caplog.at_level(logging.INFO, logger="mesh_n_bone.meshify.meshify"):
            estimates = mf._scan_assembly_mesh_estimates(str(tmp_path), amplification, num_workers=1)
        by_id = {e.mesh_id: e for e in estimates}
        assert set(by_id) == {"7", "9"}
        assert (by_id["7"].num_files, by_id["7"].vertex_count, by_id["7"].face_count) == (2, 40, 80)
        assert (by_id["9"].num_files, by_id["9"].vertex_count, by_id["9"].face_count) == (1, 4, 2)
        # Both routes produce the same estimate for the same counts.
        slow = mf._scan_one_assembly_mesh_dir(str(tmp_path / "7"), "7", amplification)
        assert slow.estimated_peak_bytes == by_id["7"].estimated_peak_bytes
        assert any("1 of 2 chunked meshes totalled from block manifests" in r.message for r in caplog.records)

    def test_scan_without_manifests_is_unchanged(self, tmp_path):
        (tmp_path / "3").mkdir()
        (tmp_path / "3" / "block_0.ply").write_bytes(_ply(10, 20))
        estimates = mf._scan_assembly_mesh_estimates(str(tmp_path), 24, num_workers=1)
        assert [(e.mesh_id, e.vertex_count, e.face_count) for e in estimates] == [("3", 10, 20)]
