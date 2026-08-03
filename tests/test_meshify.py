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
        # 180 GB / 12 = 15 GB per worker; amplification=6 -> ~2500 MB.
        result = _estimate_block_target_mb_from_dask_config(path)
        assert 2400 < result < 2600

    def test_huge_worker_is_unbounded(self, tmp_output_dir):
        from mesh_n_bone.meshify.meshify import _estimate_block_target_mb_from_dask_config
        path = os.path.join(tmp_output_dir, "dask-config.yaml")
        # 60 GB worker, no ceiling -> ~10000 MB.
        with open(path, "w") as f:
            f.write("jobqueue:\n  local:\n    memory: 60GB\n    processes: 1\n")
        result = _estimate_block_target_mb_from_dask_config(path)
        assert 9000 < result < 11000

    def test_small_worker_keeps_fallback(self, tmp_output_dir):
        from mesh_n_bone.meshify.meshify import _estimate_block_target_mb_from_dask_config
        path = os.path.join(tmp_output_dir, "dask-config.yaml")
        # 256 MB worker / 6 ~= 43 MB, smaller than the floor.
        with open(path, "w") as f:
            f.write("jobqueue:\n  local:\n    memory: 256MB\n    processes: 1\n")
        result = _estimate_block_target_mb_from_dask_config(path, fallback_mb=128)
        assert result == 128

    def test_malformed_config_falls_back(self, tmp_output_dir):
        from mesh_n_bone.meshify.meshify import _estimate_block_target_mb_from_dask_config
        path = os.path.join(tmp_output_dir, "dask-config.yaml")
        with open(path, "w") as f:
            f.write("not-jobqueue:\n  foo: bar\n")
        assert _estimate_block_target_mb_from_dask_config(path, fallback_mb=128) == 128


class TestAssemblyMemoryPlanning:
    def _write_binary_ply(self, path, vertices, faces):
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {vertices}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            f"element face {faces}\n"
            "property list int int vertex_indices\n"
            "end_header\n"
        ).encode("ascii")
        body = b"\0" * (vertices * 3 * 4 + faces * 4 * 4)
        with open(path, "wb") as f:
            f.write(header + body)

    def test_scans_ply_headers_for_mesh_estimates(self, tmp_output_dir):
        from mesh_n_bone.meshify.meshify import _scan_assembly_mesh_estimates

        mesh_dir = os.path.join(tmp_output_dir, "tmp_chunked", "42")
        os.makedirs(mesh_dir)
        self._write_binary_ply(os.path.join(mesh_dir, "block_0.ply"), 10, 20)
        self._write_binary_ply(os.path.join(mesh_dir, "block_1.ply"), 2, 4)

        estimates = _scan_assembly_mesh_estimates(
            os.path.join(tmp_output_dir, "tmp_chunked"),
            amplification=16,
        )
        assert len(estimates) == 1
        estimate = estimates[0]
        assert estimate.mesh_id == "42"
        assert estimate.num_files == 2
        assert estimate.vertex_count == 12
        assert estimate.face_count == 24
        assert estimate.raw_mesh_bytes == 12 * 3 * 8 + 24 * 3 * 4
        assert estimate.estimated_peak_bytes > estimate.raw_mesh_bytes

    @pytest.mark.parametrize(
        "num_workers",
        [
            1,   # sequential path: max_workers > 1 is false
            2,   # threaded path: total (10) > max_workers
            32,  # more workers requested than mesh dirs exist: falls back to
                 # sequential since total (10) > max_workers is false
        ],
    )
    def test_scan_result_is_independent_of_num_workers(
        self, tmp_output_dir, num_workers,
    ):
        from mesh_n_bone.meshify.meshify import _scan_assembly_mesh_estimates

        tmp_chunked = os.path.join(tmp_output_dir, "tmp_chunked")
        mesh_ids = [str(i) for i in range(10)]
        for mesh_id in mesh_ids:
            mesh_dir = os.path.join(tmp_chunked, mesh_id)
            os.makedirs(mesh_dir)
            self._write_binary_ply(os.path.join(mesh_dir, "block_0.ply"), 1, 1)

        estimates = _scan_assembly_mesh_estimates(
            tmp_chunked, amplification=16, num_workers=num_workers,
        )
        assert [e.mesh_id for e in estimates] == sorted(mesh_ids)

    def test_balanced_batches_leave_giant_mesh_alone(self):
        from mesh_n_bone.meshify.meshify import (
            AssemblyMeshEstimate,
            _balanced_assembly_batches,
        )

        estimates = [
            AssemblyMeshEstimate("giant", 1, 1000, 0, 0, 0, 1000),
            *[
                AssemblyMeshEstimate(f"small-{i}", 1, 1, 0, 0, 0, 1)
                for i in range(20)
            ],
        ]
        batches = _balanced_assembly_batches(estimates, max_batches=4)
        giant_batches = [batch for batch in batches if "giant" in batch]
        assert giant_batches == [["giant"]]
        assert sum(len(batch) for batch in batches) == len(estimates)

    def test_assembly_amplification_constants_cover_heavy_paths(self):
        from mesh_n_bone.meshify.meshify import _assembly_memory_amplification

        assert _assembly_memory_amplification(
            do_simplification=True,
            smooth_before_simplify=False,
            check_mesh_validity=False,
            has_custom_roi=False,
        ) == 24
        assert _assembly_memory_amplification(
            do_simplification=True,
            smooth_before_simplify=True,
            check_mesh_validity=True,
            has_custom_roi=False,
        ) == 36
        assert _assembly_memory_amplification(
            do_simplification=False,
            smooth_before_simplify=False,
            check_mesh_validity=False,
            has_custom_roi=False,
        ) == 20

    def test_plan_assembly_waves_lowers_processes_for_large_mesh(self):
        from mesh_n_bone.meshify.meshify import (
            AssemblyMeshEstimate,
            _plan_assembly_waves,
        )

        cfg = {
            "jobqueue": {
                "lsf": {
                    "ncpus": 12,
                    "processes": 12,
                    "cores": 12,
                    "memory": "180GB",
                }
            }
        }
        gib = 2 ** 30
        estimates = [
            AssemblyMeshEstimate("large", 1, 100, 0, 0, 0, 20 * gib),
            AssemblyMeshEstimate("small", 1, 1, 0, 0, 0, 1 * gib),
        ]

        waves = _plan_assembly_waves(estimates, requested_workers=576, config=cfg)
        assert [wave.processes for wave in waves] == [5, 12]
        large_wave = waves[0]
        assert large_wave.batches == [["large"]]
        assert large_wave.workers == 5
        assert large_wave.config["jobqueue"]["lsf"]["processes"] == 5
        assert large_wave.config["jobqueue"]["lsf"]["cores"] == 5
        assert large_wave.config["jobqueue"]["lsf"]["ncpus"] == 12
        assert cfg["jobqueue"]["lsf"]["processes"] == 12
        assert cfg["jobqueue"]["lsf"]["ncpus"] == 12


class TestTensorStoreReadTimeouts:
    class _FakeDataset:
        class _FakeDType:
            numpy_dtype = np.dtype("uint64")

        dtype = _FakeDType()

    def _slices_for_mib(self, mib):
        voxels = int(mib * 2**20 / np.dtype("uint64").itemsize)
        return (slice(0, voxels),)

    def test_small_reads_keep_old_five_second_timeout(self):
        from mesh_n_bone.util.image_data_interface import (
            _default_read_timeout_seconds,
        )

        slices = self._slices_for_mib(67)
        assert _default_read_timeout_seconds(
            self._FakeDataset(), slices, "/nrs/local.zarr",
        ) == 5.0
        assert _default_read_timeout_seconds(
            self._FakeDataset(), slices, "gs://bucket/volume",
        ) == 5.0

    def test_remote_timeout_scales_with_read_size(self):
        from mesh_n_bone.util.image_data_interface import (
            _default_read_timeout_seconds,
        )

        slices = self._slices_for_mib(512)
        assert _default_read_timeout_seconds(
            self._FakeDataset(), slices, "precomputed://gs://bucket/volume",
        ) == 32.0

    def test_local_timeout_scales_more_slowly_and_caps(self):
        from mesh_n_bone.util.image_data_interface import (
            _default_read_timeout_seconds,
        )

        slices = self._slices_for_mib(4096)
        assert _default_read_timeout_seconds(
            self._FakeDataset(), slices, "/nrs/local.zarr",
        ) == 16.0

        huge_slices = self._slices_for_mib(16384)
        assert _default_read_timeout_seconds(
            self._FakeDataset(), huge_slices, "/nrs/local.zarr",
        ) == 30.0


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

    def test_chunk_reduction_zero_when_fixed_edge_simplification_disabled(self):
        from mesh_n_bone.meshify.meshify import _chunk_stage_1_reduction

        assert _chunk_stage_1_reduction(
            {
                "use_fixed_edge_simplification": False,
                "target_reduction": 0.933,
                "stage_1_reduction_fraction": 0.25,
            }
        ) == 0.0

    def test_chunk_reduction_uses_stage_split_when_enabled(self):
        from mesh_n_bone.meshify.meshify import (
            _chunk_stage_1_reduction,
            staged_reductions,
        )

        config = {
            "use_fixed_edge_simplification": True,
            "target_reduction": 0.933,
            "stage_1_reduction_fraction": 0.25,
        }
        expected, _ = staged_reductions(0.933, 0.25, 0.75)
        assert _chunk_stage_1_reduction(config) == expected

    def test_assembly_uses_second_stage_reduction(self, tmp_output_dir, monkeypatch):
        from cloudvolume.mesh import Mesh as CloudVolumeMesh
        from mesh_n_bone.meshify.meshify import Meshify, staged_reductions

        block_dir = os.path.join(tmp_output_dir, "blocks", "1")
        output_dir = os.path.join(tmp_output_dir, "out")
        os.makedirs(block_dir)
        os.makedirs(os.path.join(output_dir, "meshes"))

        vertices = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        faces = np.array(
            [
                [0, 1, 2],
                [0, 1, 3],
                [0, 2, 3],
                [1, 2, 3],
            ],
            dtype=np.uint32,
        )
        mesh = CloudVolumeMesh(vertices, faces, normals=None)
        with open(os.path.join(block_dir, "block_0.ply"), "wb") as f:
            f.write(mesh.to_ply())

        reductions = []

        def fake_simplify_and_smooth_mesh(input_mesh, target_reduction, *args, **kwargs):
            reductions.append(target_reduction)
            return trimesh.Trimesh(
                vertices=input_mesh.vertices,
                faces=input_mesh.faces,
                process=False,
            )

        monkeypatch.setattr(
            Meshify,
            "simplify_and_smooth_mesh",
            staticmethod(fake_simplify_and_smooth_mesh),
        )

        meshify = object.__new__(Meshify)
        meshify.dirname = os.path.join(tmp_output_dir, "blocks")
        meshify.output_directory = output_dir
        meshify.max_num_blocks = 100
        meshify.check_mesh_validity = False
        meshify.has_custom_roi = False
        meshify.remove_smallest_components = False
        meshify.use_fixed_edge_simplification = True
        meshify.do_simplification = True
        meshify.target_reduction = 0.933
        meshify.stage_1_reduction_fraction = 0.25
        meshify.stage_2_reduction_fraction = 0.75
        meshify.n_smoothing_iter = 0
        meshify.default_aggressiveness = 0.3
        meshify.smooth_before_simplify = True
        meshify.true_voxel_size = np.array([1, 1, 1])
        meshify.output_voxel_size_funlib = np.array([1, 1, 1])
        meshify.do_legacy_neuroglancer = False
        meshify.do_singleres_multires_neuroglancer = False

        meshify._assemble_mesh("1")

        _, expected_stage_2 = staged_reductions(0.933, 0.25, 0.75)
        assert reductions == [expected_stage_2]

    def test_zero_reduction_skips_chunk_decimator(self, monkeypatch):
        from mesh_n_bone.meshify import fixed_edge

        mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)

        def fail_decimator(*args, **kwargs):
            raise AssertionError("decimator should not run")

        monkeypatch.setattr(fixed_edge, "pymeshlab_simplify", fail_decimator)
        out = fixed_edge.simplify_mesh(
            mesh,
            target_reduction=0.0,
            voxel_size=np.array([1, 1, 1]),
            block_size=None,
            fix_edges=True,
        )

        assert len(out.faces) > 0


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


class TestPerLodTargetReduction:
    """The per-LOD target_reduction formula used by the downsample multires
    strategy. Anchored on hemibrain face counts for seg 488231898: at
    target_reduction=0.933 / 16nm input / decimation_factor=6, the formula
    must yield reductions that produce a constant 6x face-count drop per LOD."""

    def _make_meshify(self, target_reduction=0.933, decimation_factor=6):
        from unittest.mock import MagicMock
        from mesh_n_bone.meshify.meshify import Meshify
        m = MagicMock(spec=Meshify)
        m.target_reduction = target_reduction
        m.decimation_factor = decimation_factor
        m._per_lod_target_reduction = (
            Meshify._per_lod_target_reduction.__get__(m, Meshify)
        )
        return m

    def test_lod_0_equals_input_target_reduction(self):
        m = self._make_meshify(target_reduction=0.933)
        assert m._per_lod_target_reduction(0) == pytest.approx(0.933, abs=1e-9)

    def test_lod_k_yields_constant_6x_face_drop(self):
        """For decimation_factor=6, face count at LOD k+1 should be 1/6 of
        LOD k's face count, given raw MC drops by 4x per scale step."""
        m = self._make_meshify(target_reduction=0.933, decimation_factor=6)
        raw_mc_0 = 76_500_000
        prev_faces = raw_mc_0 * (1 - m._per_lod_target_reduction(0))
        for k in range(1, 4):
            raw_mc_k = raw_mc_0 / (4 ** k)
            faces_k = raw_mc_k * (1 - m._per_lod_target_reduction(k))
            ratio = prev_faces / faces_k
            assert ratio == pytest.approx(6.0, rel=1e-6), (
                f"LOD {k-1}->{k} face ratio should be 6, got {ratio}"
            )
            prev_faces = faces_k

    def test_decimation_factor_4_yields_no_extra_reduction(self):
        """When decimation_factor == 4 (= raw MC scaling per LOD), the
        per-LOD reduction is constant (no extra decimation needed)."""
        m = self._make_meshify(target_reduction=0.9, decimation_factor=4)
        for k in range(4):
            assert m._per_lod_target_reduction(k) == pytest.approx(0.9, abs=1e-9)

    def test_hemibrain_anchor_values(self):
        """Formula values at the hemibrain-match config land within ~35%
        of the per-LOD values computed directly from hemibrain's measured
        face counts (hemibrain's per-LOD ratio isn't exactly 6 — varies
        5.13-7.14, hence the slack)."""
        m = self._make_meshify(target_reduction=0.933, decimation_factor=6)
        hb_faces = [5_219_990, 730_941, 119_321, 23_259]
        raw_mc_0 = 76_500_000
        for k in range(4):
            raw_k = raw_mc_0 / (4 ** k)
            tr_formula = m._per_lod_target_reduction(k)
            keep_formula = 1 - tr_formula
            faces_formula = raw_k * keep_formula
            ratio = faces_formula / hb_faces[k]
            assert 0.7 < ratio < 1.35, (
                f"LOD {k}: formula {faces_formula:.0f} vs hemibrain "
                f"{hb_faces[k]} (ratio {ratio:.2f}) out of expected band"
            )
