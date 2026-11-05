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


class TestBlockPacks:
    def test_round_trip_through_pack_and_manifest(self, tmp_path):
        pieces = [(7, b"seven-from-block-0", 10, 20, 0), (8, b"eight", 5, 6, 1)]
        mf._write_block_pack(str(tmp_path), 0, pieces)
        mf._write_block_pack(str(tmp_path), 1, [(7, b"seven-from-block-1", 30, 60, 0)])
        mf._write_block_pack(str(tmp_path), 2, [])  # a block that wrote nothing
        assert sorted(os.listdir(tmp_path / mf._MANIFEST_SUBDIR)) == ["block_0.npy", "block_1.npy"]
        assert sorted(os.listdir(tmp_path / mf._PACK_SUBDIR)) == ["block_0.pack", "block_1.pack"]
        rows = mf._read_block_manifests(str(tmp_path))
        assert len(rows) == 3
        for row in rows:
            data = mf._read_pack_piece(str(tmp_path), row["block"], row["offset"], row["bytes"])
            expected = {(0, 7): b"seven-from-block-0", (0, 8): b"eight", (1, 7): b"seven-from-block-1"}
            assert data == expected[(int(row["block"]), int(row["id"]))]

    def test_a_short_pack_is_an_error(self, tmp_path):
        mf._write_block_pack(str(tmp_path), 0, [(7, b"abc", 1, 1, 0)])
        with pytest.raises(IOError, match="shorter than its manifest"):
            mf._read_pack_piece(str(tmp_path), 0, 0, 10)

    def test_plan_separates_finished_segments_from_the_rest(self, tmp_path):
        # 1: one piece, finished in its block            -> nothing to do
        # 2: one piece, not finished (touched a face)    -> assembled
        # 3: finished in block 0 and met again in block 1 -> redone, assembled
        # 4: pieces in blocks 2 and 0                     -> assembled, pieces by block
        mf._write_block_pack(str(tmp_path), 0, [
            (1, b"1", 1, 1, 1), (3, b"33", 2, 2, 1), (4, b"444", 3, 3, 0),
        ])
        mf._write_block_pack(str(tmp_path), 1, [(2, b"2", 1, 1, 0), (3, b"3", 1, 1, 0)])
        mf._write_block_pack(str(tmp_path), 2, [(4, b"4", 1, 1, 0)])
        plan = mf._plan_assembly_from_manifests(mf._read_block_manifests(str(tmp_path)), 32)
        assert plan.finished_in_blocks == 1
        assert plan.redone == [3]
        assert [e.mesh_id for e in plan.estimates] == ["2", "3", "4"]
        assert plan.pieces[4] == ((0, 3, 3), (2, 0, 1))
        assert 1 not in plan.pieces
        by_id = {e.mesh_id: e for e in plan.estimates}
        assert (by_id["3"].num_files, by_id["3"].ply_bytes, by_id["3"].face_count) == (2, 3, 3)

    def test_no_manifests_means_an_empty_plan(self, tmp_path):
        plan = mf._plan_assembly_from_manifests(mf._read_block_manifests(str(tmp_path)), 32)
        assert (plan.estimates, plan.pieces, plan.finished_in_blocks, plan.redone) == ([], {}, 0, [])


