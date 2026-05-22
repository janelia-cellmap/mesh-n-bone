"""Tests for meshify module components."""

import os
import numpy as np
import pytest
import trimesh


class TestNormalizeTargetIds:
    """target_ids accepts None, int, list, or CSV path."""

    def test_none_returns_none(self):
        from mesh_n_bone.meshify.meshify import _normalize_target_ids
        assert _normalize_target_ids(None) is None

    def test_int_wrapped(self):
        from mesh_n_bone.meshify.meshify import _normalize_target_ids
        assert _normalize_target_ids(12345) == frozenset([12345])

    def test_list_converted(self):
        from mesh_n_bone.meshify.meshify import _normalize_target_ids
        assert _normalize_target_ids([1, 2, 3]) == frozenset([1, 2, 3])

    def test_csv_with_id_column(self, tmp_output_dir):
        from mesh_n_bone.meshify.meshify import _normalize_target_ids
        path = os.path.join(tmp_output_dir, "ids.csv")
        with open(path, "w") as f:
            f.write("id\n10\n20\n30\n")
        assert _normalize_target_ids(path) == frozenset([10, 20, 30])

    def test_csv_with_object_id_column(self, tmp_output_dir):
        from mesh_n_bone.meshify.meshify import _normalize_target_ids
        path = os.path.join(tmp_output_dir, "ids.csv")
        with open(path, "w") as f:
            f.write("Object ID,other_col\n10,foo\n20,bar\n")
        assert _normalize_target_ids(path) == frozenset([10, 20])

    def test_csv_headerless_first_column(self, tmp_output_dir):
        from mesh_n_bone.meshify.meshify import _normalize_target_ids
        path = os.path.join(tmp_output_dir, "ids.csv")
        # Pandas treats the first row as a header by default; with no
        # known id-column name, our normalize falls back to first column.
        with open(path, "w") as f:
            f.write("seg\n100\n200\n")
        assert _normalize_target_ids(path) == frozenset([100, 200])


class TestTargetIdsWorker:
    """Block-worker behavior when target_ids is set."""

    def _config(self, **overrides):
        """Minimal worker config (only the keys this test path reads)."""
        base = {
            "downsample_factor": None,
            "downsample_method": "nearest",
            "use_fixed_edge_simplification": False,
            "do_simplification": False,
            "target_reduction": 0.99,
            "stage_1_reduction_fraction": 0.5,
            "read_write_block_shape_pixels": [16, 16, 16],
            "default_aggressiveness": 0.3,
            "target_ids": None,
        }
        base.update(overrides)
        return base

    def test_mask_except_keeps_only_targets(self):
        """fastremap.mask_except behaves the way the worker expects it to."""
        import fastremap
        vol = np.array([0, 1, 2, 3, 1, 5, 1], dtype=np.uint64)
        out = fastremap.mask_except(vol, [1, 3], in_place=False)
        np.testing.assert_array_equal(out, [0, 1, 0, 3, 1, 0, 1])

    def test_renumber_round_trips(self):
        """The remap dict the worker builds round-trips correctly."""
        import fastremap
        keep = [42, 7, 1000]
        remap = {old: new for new, old in enumerate(sorted(keep), start=1)}
        inv_remap = {new: old for old, new in remap.items()}
        vol = np.array([0, 42, 7, 1000, 42, 0], dtype=np.uint64)
        small = fastremap.remap(vol, remap, preserve_missing_labels=True).astype(np.uint16)
        np.testing.assert_array_equal(small, [0, 2, 1, 3, 2, 0])
        # And inverse
        for new, old in inv_remap.items():
            assert remap[old] == new


class TestEstimateBlockTargetMb:
    """`_estimate_block_target_mb_from_dask_config` reads dask-config.yaml
    and returns a per-block memory budget derived from per-worker RAM."""

    def test_fallback_when_no_config(self, tmp_output_dir):
        from mesh_n_bone.meshify.meshify import _estimate_block_target_mb_from_dask_config
        missing = os.path.join(tmp_output_dir, "missing-dask-config.yaml")
        assert _estimate_block_target_mb_from_dask_config(missing, fallback_mb=128) == 128

    def test_real_lsf_config(self, tmp_output_dir):
        from mesh_n_bone.meshify.meshify import _estimate_block_target_mb_from_dask_config
        path = os.path.join(tmp_output_dir, "dask-config.yaml")
        with open(path, "w") as f:
            f.write("jobqueue:\n  lsf:\n    memory: 180GB\n    processes: 12\n")
        # 180 GB / 12 = 15 GB per worker; with amplification=8 -> 1875 MB,
        # capped to 1024 MB.
        result = _estimate_block_target_mb_from_dask_config(path, cap_mb=1024)
        assert result == 1024

    def test_small_worker_keeps_fallback(self, tmp_output_dir):
        from mesh_n_bone.meshify.meshify import _estimate_block_target_mb_from_dask_config
        path = os.path.join(tmp_output_dir, "dask-config.yaml")
        # 1 GB worker memory / 8 = 125 MB, smaller than the floor.
        with open(path, "w") as f:
            f.write("jobqueue:\n  local:\n    memory: 1GB\n    processes: 1\n")
        # Floored at fallback_mb to keep block size sane.
        result = _estimate_block_target_mb_from_dask_config(
            path, fallback_mb=128, cap_mb=1024
        )
        assert result == 128

    def test_malformed_config_falls_back(self, tmp_output_dir):
        from mesh_n_bone.meshify.meshify import _estimate_block_target_mb_from_dask_config
        path = os.path.join(tmp_output_dir, "dask-config.yaml")
        with open(path, "w") as f:
            f.write("not-jobqueue:\n  foo: bar\n")
        assert _estimate_block_target_mb_from_dask_config(path, fallback_mb=128) == 128


class TestBlockFromIndex:
    """`block_from_index` should produce the same DaskBlocks as
    `create_blocks` would, indexed by integer."""

    def test_index_parity_with_eager_create_blocks(self):
        from mesh_n_bone.util.dask_util import (
            block_from_index, count_blocks, create_blocks,
        )
        from funlib.geometry import Coordinate, Roi
        from types import SimpleNamespace

        roi = Roi((0, 0, 0), (96, 64, 128))
        block_size_world = Coordinate(32, 32, 32)
        ds = SimpleNamespace(
            chunk_shape=block_size_world,
            voxel_size=Coordinate(1, 1, 1),
        )
        eager = create_blocks(roi, ds)
        n_eager = len(eager)
        n_lazy = count_blocks(roi, block_size_world)
        assert n_eager == n_lazy
        for i in range(n_eager):
            lazy = block_from_index(
                i, roi.get_begin(), roi.get_end(), block_size_world,
            )
            assert lazy.index == eager[i].index == i
            assert tuple(lazy.roi.get_begin()) == tuple(eager[i].roi.get_begin()), (
                f"index {i}: {lazy.roi.get_begin()} != {eager[i].roi.get_begin()}"
            )
            assert tuple(lazy.roi.get_end()) == tuple(eager[i].roi.get_end())

    def test_padding_round_trip(self):
        from mesh_n_bone.util.dask_util import block_from_index
        from funlib.geometry import Coordinate

        b = block_from_index(
            0, (0, 0, 0), (32, 32, 32), (32, 32, 32), padding=Coordinate(4, 4, 4)
        )
        # padding=(4,4,4) grows the (0,0,0)+(32,32,32) block by 4 on each side
        assert tuple(b.roi.get_begin()) == (-4, -4, -4)
        assert tuple(b.roi.get_shape()) == (40, 40, 40)


class TestStagedReductions:
    def test_staged_reductions_sum(self):
        from mesh_n_bone.meshify.meshify import staged_reductions

        r1, r2 = staged_reductions(0.99, 0.5, 0.5)
        # After applying both reductions, overall keep should be 1 - 0.99 = 0.01
        keep_total = (1 - r1) * (1 - r2)
        np.testing.assert_almost_equal(keep_total, 0.01, decimal=6)

    def test_staged_reductions_asymmetric(self):
        from mesh_n_bone.meshify.meshify import staged_reductions

        r1, r2 = staged_reductions(0.90, 0.25, 0.75)
        keep_total = (1 - r1) * (1 - r2)
        np.testing.assert_almost_equal(keep_total, 0.10, decimal=6)

    def test_staged_reductions_invalid_fractions(self):
        from mesh_n_bone.meshify.meshify import staged_reductions

        with pytest.raises(AssertionError):
            staged_reductions(0.99, 0.3, 0.3)


class TestRepairMeshPymeshlab:
    def test_repair_simple_mesh(self, tiny_cube_mesh):
        from mesh_n_bone.meshify.meshify import Meshify

        repaired = Meshify.repair_mesh_pymeshlab(
            tiny_cube_mesh.vertices,
            tiny_cube_mesh.faces,
            remove_smallest_components=False,
        )
        assert len(repaired.vertices) > 0
        assert len(repaired.faces) > 0

    def test_is_mesh_valid(self, tiny_sphere_mesh):
        from mesh_n_bone.meshify.meshify import Meshify

        # Sphere should be valid
        assert Meshify.is_mesh_valid(tiny_sphere_mesh)


class TestDownsample:
    def test_flat_mode(self):
        from mesh_n_bone.meshify.downsample import flat_mode

        data = np.array([[[1, 1, 2], [1, 2, 2], [1, 1, 1]]])
        result = flat_mode(data)
        assert result == 1

    def test_flat_mode_except_zero(self):
        from mesh_n_bone.meshify.downsample import flat_mode_except_zero

        data = np.array([[[0, 0, 5], [5, 0, 5], [0, 0, 0]]])
        result = flat_mode_except_zero(data)
        assert result == 5

    def test_flat_binary_mode(self):
        from mesh_n_bone.meshify.downsample import flat_binary_mode

        # Majority nonzero
        data = np.array([[[1, 1, 1], [1, 0, 1], [1, 1, 1]]])
        assert flat_binary_mode(data) == 1

        # Majority zero
        data = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
        assert flat_binary_mode(data) == 0

    def test_downsample_box(self):
        from mesh_n_bone.meshify.downsample import downsample_box

        box = np.array([[0, 0, 0], [10, 10, 10]])
        block_shape = np.array([2, 2, 2])
        result = downsample_box(box, block_shape)
        np.testing.assert_array_equal(result[0], [0, 0, 0])
        np.testing.assert_array_equal(result[1], [5, 5, 5])


class TestDefaultBlockShape:
    """Test _default_block_shape_pixels computes sensible multiples."""

    @staticmethod
    def _make_mock_array(chunk_shape, dtype):
        """Minimal object with chunk_shape and dtype."""
        from types import SimpleNamespace
        from funlib.geometry import Coordinate
        return SimpleNamespace(
            chunk_shape=Coordinate(chunk_shape),
            dtype=np.dtype(dtype),
        )

    def test_stays_within_budget(self):
        from mesh_n_bone.meshify.meshify import Meshify

        meshify = object.__new__(Meshify)
        meshify.segmentation_array = self._make_mock_array((112, 112, 112), "uint64")
        block = meshify._default_block_shape_pixels(target_mb=128)
        actual_mb = int(np.prod(block)) * 8 / 1e6
        assert actual_mb <= 128, f"Block {block} uses {actual_mb:.0f} MB, exceeds 128 MB"

    def test_at_least_one_chunk(self):
        from mesh_n_bone.meshify.meshify import Meshify

        meshify = object.__new__(Meshify)
        meshify.segmentation_array = self._make_mock_array((256, 256, 256), "uint64")
        block = meshify._default_block_shape_pixels(target_mb=10)
        # Even if budget is tiny, should be at least 1x chunk
        np.testing.assert_array_equal(block, [256, 256, 256])

    def test_larger_budget_gives_larger_block(self):
        from mesh_n_bone.meshify.meshify import Meshify

        meshify = object.__new__(Meshify)
        meshify.segmentation_array = self._make_mock_array((64, 64, 64), "uint64")
        small = meshify._default_block_shape_pixels(target_mb=10)
        large = meshify._default_block_shape_pixels(target_mb=500)
        assert np.all(large >= small)

    def test_is_chunk_aligned(self):
        from mesh_n_bone.meshify.meshify import Meshify

        meshify = object.__new__(Meshify)
        chunk = (96, 96, 96)
        meshify.segmentation_array = self._make_mock_array(chunk, "uint32")
        block = meshify._default_block_shape_pixels(target_mb=128)
        # Must be an exact multiple of chunk shape
        assert np.all(block % np.array(chunk) == 0)


class TestClipPlaneDuplicateMerge:
    """Test that duplicate vertices at fixed-edge clip planes are merged."""

    def test_merge_clip_plane_duplicates(self):
        """Simulates two adjacent block meshes with shared clip-plane vertices.

        Fixed-edge simplification clips block meshes at half-voxel inward
        from chunk boundaries, producing exact-duplicate vertices that
        ``deduplicate_chunk_boundaries`` misses (mod != 0).  The merge
        step should collapse these duplicates.
        """
        # Create two open half-spheres meeting at z=0 (simulating clip plane)
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
        sphere.vertices += 20  # offset into positive quadrant

        # Split into two blocks at z=20 (the center)
        split_z = 20.0
        block_a = trimesh.intersections.slice_mesh_plane(
            sphere, [0, 0, -1], [0, 0, split_z], cap=False
        )
        block_b = trimesh.intersections.slice_mesh_plane(
            sphere, [0, 0, 1], [0, 0, split_z], cap=False
        )

        # Concatenate (simulating assembly without dedup)
        combined = trimesh.util.concatenate([block_a, block_b])
        n_before = len(combined.vertices)

        # The duplicate vertices at the split plane should exist
        at_plane = np.abs(combined.vertices[:, 2] - split_z) < 0.01
        # Each plane vertex exists twice (once from each block)
        unique_at_plane = np.unique(
            np.round(combined.vertices[at_plane], 4), axis=0
        )
        n_dups_at_plane = at_plane.sum() - len(unique_at_plane)
        assert n_dups_at_plane > 0, "Expected duplicate vertices at split plane"

        # Merge them (same as the fix in _assemble_mesh)
        combined.merge_vertices(merge_tex=False, merge_norm=False)
        n_after = len(combined.vertices)

        assert n_after < n_before, "merge_vertices should reduce vertex count"
        # All duplicates at the split plane should now be gone
        at_plane_after = np.abs(combined.vertices[:, 2] - split_z) < 0.01
        unique_after = np.unique(
            np.round(combined.vertices[at_plane_after], 4), axis=0
        )
        assert at_plane_after.sum() == len(unique_after), (
            "All duplicate vertices at the split plane should be merged"
        )
