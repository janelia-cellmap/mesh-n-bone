"""Integration tests for the meshify pipeline: zarr → marching cubes → chunk assembly → mesh output.

Tests the full blockwise workflow: reading segmentation from zarr, generating
per-chunk meshes, assembling across chunk boundaries, simplification, and
neuroglancer format output.
"""

import numpy as np
import os
import pytest
import tempfile
import trimesh
import zarr

from funlib.geometry import Coordinate

from mesh_n_bone.meshify.meshify import Meshify


def _create_zarr_volume(tmpdir, vol, voxel_size=(8, 8, 8), chunk_shape=(16, 16, 16)):
    """Helper: write a labeled volume to zarr with metadata."""
    zarr_path = os.path.join(tmpdir, "test.zarr")
    root = zarr.open_group(zarr_path, mode="w")
    arr = root.create_array("labels/s0", data=vol, chunks=chunk_shape)
    arr.attrs["voxel_size"] = list(voxel_size)
    arr.attrs["offset"] = [0, 0, 0]
    return f"{zarr_path}/labels/s0"


class TestMeshifyFromZarr:
    """End-to-end tests: zarr segmentation → PLY meshes."""

    def test_two_separate_objects(self, tmp_output_dir):
        """Two non-overlapping objects should produce two separate mesh files."""
        vol = np.zeros((32, 32, 32), dtype=np.uint64)
        vol[2:14, 2:14, 2:14] = 1
        vol[18:30, 18:30, 18:30] = 2

        input_path = _create_zarr_volume(tmp_output_dir, vol)
        output_dir = os.path.join(tmp_output_dir, "output")

        m = Meshify(
            input_path=input_path,
            output_directory=output_dir,
            num_workers=1,
            do_analysis=False,
            check_mesh_validity=False,
            do_simplification=False,
            n_smoothing_iter=0,
            remove_smallest_components=False,
        )
        m.get_meshes()

        mesh_dir = os.path.join(output_dir, "meshes")
        meshes = sorted(os.listdir(mesh_dir))
        assert len(meshes) == 2
        assert "1.ply" in meshes
        assert "2.ply" in meshes

        for mesh_file in meshes:
            mesh = trimesh.load(os.path.join(mesh_dir, mesh_file))
            assert len(mesh.faces) > 0
            assert mesh.volume > 0

    def test_target_ids_filters_output(self, tmp_output_dir):
        """`target_ids=[1]` should produce ONLY mesh 1, even though
        the volume also has object 2."""
        vol = np.zeros((32, 32, 32), dtype=np.uint64)
        vol[2:14, 2:14, 2:14] = 1
        vol[18:30, 18:30, 18:30] = 2

        input_path = _create_zarr_volume(tmp_output_dir, vol)
        output_dir = os.path.join(tmp_output_dir, "output")

        m = Meshify(
            input_path=input_path,
            output_directory=output_dir,
            num_workers=1,
            do_analysis=False,
            check_mesh_validity=False,
            do_simplification=False,
            n_smoothing_iter=0,
            remove_smallest_components=False,
            target_ids=[1],
        )
        m.get_meshes()

        mesh_dir = os.path.join(output_dir, "meshes")
        meshes = sorted(os.listdir(mesh_dir))
        assert meshes == ["1.ply"], (
            f"target_ids=[1] should produce only 1.ply, got {meshes}"
        )
        mesh = trimesh.load(os.path.join(mesh_dir, "1.ply"))
        # Volume should match object 1 (12^3 voxels × 8^3 nm³/voxel)
        np.testing.assert_allclose(mesh.volume, 12**3 * 8**3, rtol=0.1)

    def test_target_ids_csv_input(self, tmp_output_dir):
        """`target_ids` set to a CSV file path should load the ids from
        the file and meshify exactly those."""
        vol = np.zeros((32, 32, 32), dtype=np.uint64)
        vol[2:10, 2:10, 2:10] = 1
        vol[12:20, 12:20, 12:20] = 2
        vol[22:30, 22:30, 22:30] = 3

        input_path = _create_zarr_volume(tmp_output_dir, vol)
        output_dir = os.path.join(tmp_output_dir, "output")

        ids_csv = os.path.join(tmp_output_dir, "ids.csv")
        with open(ids_csv, "w") as f:
            f.write("id\n1\n3\n")

        m = Meshify(
            input_path=input_path,
            output_directory=output_dir,
            num_workers=1,
            do_analysis=False,
            check_mesh_validity=False,
            do_simplification=False,
            n_smoothing_iter=0,
            remove_smallest_components=False,
            target_ids=ids_csv,
        )
        m.get_meshes()

        meshes = sorted(os.listdir(os.path.join(output_dir, "meshes")))
        assert meshes == ["1.ply", "3.ply"], (
            f"CSV [1,3] should produce 1.ply + 3.ply only, got {meshes}"
        )

    def test_cross_chunk_object_is_watertight(self, tmp_output_dir):
        """An object spanning multiple chunks should assemble into a watertight mesh."""
        vol = np.zeros((32, 32, 32), dtype=np.uint64)
        # Object spans all 8 chunks (chunk_shape=16)
        vol[2:30, 2:30, 2:30] = 1

        input_path = _create_zarr_volume(tmp_output_dir, vol)
        output_dir = os.path.join(tmp_output_dir, "output")

        m = Meshify(
            input_path=input_path,
            output_directory=output_dir,
            num_workers=1,
            do_analysis=False,
            check_mesh_validity=False,
            do_simplification=False,
            n_smoothing_iter=0,
            remove_smallest_components=False,
        )
        m.get_meshes()

        mesh = trimesh.load(os.path.join(output_dir, "meshes", "1.ply"))
        assert mesh.is_watertight
        # Volume should be close to expected (28^3 voxels * 8^3 nm^3/voxel)
        expected_vol = (28**3) * (8**3)
        np.testing.assert_allclose(mesh.volume, expected_vol, rtol=0.05)

    def test_cross_chunk_volume_accuracy(self, tmp_output_dir):
        """Chunk assembly should not lose or duplicate geometry at boundaries."""
        vol = np.zeros((32, 32, 32), dtype=np.uint64)
        vol[6:26, 6:26, 6:26] = 1  # 20^3 voxel cube crossing chunk boundary at 16

        input_path = _create_zarr_volume(tmp_output_dir, vol)
        output_dir = os.path.join(tmp_output_dir, "output")

        m = Meshify(
            input_path=input_path,
            output_directory=output_dir,
            num_workers=1,
            do_analysis=False,
            check_mesh_validity=False,
            do_simplification=False,
            n_smoothing_iter=0,
            remove_smallest_components=False,
        )
        m.get_meshes()

        mesh = trimesh.load(os.path.join(output_dir, "meshes", "1.ply"))
        expected_vol = (20**3) * (8**3)
        # Should be within 5% — marching cubes is approximate at boundaries
        np.testing.assert_allclose(mesh.volume, expected_vol, rtol=0.05)


class TestMeshifyWithSimplification:
    """Test meshify with simplification and repair enabled."""

    def test_simplification_reduces_faces(self, tmp_output_dir):
        """Simplification should produce fewer faces than the raw mesh."""
        vol = np.zeros((32, 32, 32), dtype=np.uint64)
        vol[2:30, 2:30, 2:30] = 1

        input_path = _create_zarr_volume(tmp_output_dir, vol)

        # First: no simplification
        output_raw = os.path.join(tmp_output_dir, "output_raw")
        m_raw = Meshify(
            input_path=input_path,
            output_directory=output_raw,
            num_workers=1,
            do_analysis=False,
            check_mesh_validity=False,
            do_simplification=False,
            n_smoothing_iter=0,
            remove_smallest_components=False,
        )
        m_raw.get_meshes()
        raw_mesh = trimesh.load(os.path.join(output_raw, "meshes", "1.ply"))

        # Second: with simplification
        output_simp = os.path.join(tmp_output_dir, "output_simp")
        m_simp = Meshify(
            input_path=input_path,
            output_directory=output_simp,
            num_workers=1,
            target_reduction=0.9,
            do_analysis=False,
            check_mesh_validity=False,
            do_simplification=True,
            n_smoothing_iter=5,
            remove_smallest_components=False,
        )
        m_simp.get_meshes()
        simp_mesh = trimesh.load(os.path.join(output_simp, "meshes", "1.ply"))

        assert len(simp_mesh.faces) < len(raw_mesh.faces)
        assert simp_mesh.volume > 0

    def test_simplification_with_validity_check(self, tmp_output_dir):
        """With check_mesh_validity=True, output should be watertight."""
        vol = np.zeros((32, 32, 32), dtype=np.uint64)
        vol[4:28, 4:28, 4:28] = 1

        input_path = _create_zarr_volume(tmp_output_dir, vol)
        output_dir = os.path.join(tmp_output_dir, "output")

        m = Meshify(
            input_path=input_path,
            output_directory=output_dir,
            num_workers=1,
            target_reduction=0.9,
            do_analysis=False,
            check_mesh_validity=True,
            do_simplification=True,
            n_smoothing_iter=5,
            remove_smallest_components=True,
        )
        m.get_meshes()

        mesh = trimesh.load(os.path.join(output_dir, "meshes", "1.ply"))
        assert Meshify.is_mesh_valid(mesh)


class TestMeshifyWithDownsampling:
    """Test meshify with on-the-fly downsampling."""

    def test_downsample_factor(self, tmp_output_dir):
        """Downsampling should produce a coarser but valid mesh."""
        vol = np.zeros((32, 32, 32), dtype=np.uint64)
        vol[2:30, 2:30, 2:30] = 1

        input_path = _create_zarr_volume(tmp_output_dir, vol)
        output_dir = os.path.join(tmp_output_dir, "output")

        m = Meshify(
            input_path=input_path,
            output_directory=output_dir,
            num_workers=1,
            downsample_factor=2,
            do_analysis=False,
            check_mesh_validity=False,
            do_simplification=False,
            n_smoothing_iter=0,
            remove_smallest_components=False,
        )
        m.get_meshes()

        mesh_dir = os.path.join(output_dir, "meshes")
        assert os.path.exists(os.path.join(mesh_dir, "1.ply"))
        mesh = trimesh.load(os.path.join(mesh_dir, "1.ply"))
        assert len(mesh.faces) > 0
        assert mesh.volume > 0


class TestMeshifyDownsampleStrategyExistingScales:
    """When the input zarr has OME-NGFF multiscales metadata exposing
    pre-computed coarser scales (s1, s2, ...), the downsample multires
    strategy should read directly from those scales instead of
    redundantly downsampling the input in-worker."""

    @staticmethod
    def _create_multiscale_zarr(tmpdir, base_voxel_size=(8, 8, 8), num_levels=3):
        """Write a zarr v3 group with OME-NGFF multiscales metadata and
        pre-computed s0..s{num_levels-1} arrays."""
        import json
        import tensorstore as ts

        zarr_path = os.path.join(tmpdir, "multiscale.zarr")
        os.makedirs(zarr_path, exist_ok=True)
        # Generate s0 volume (binary sphere) and progressively downsampled levels
        N = 64
        vol = np.zeros((N, N, N), dtype=np.uint8)
        zz, yy, xx = np.indices(vol.shape) - N // 2
        vol[(xx*xx + yy*yy + zz*zz) < 24*24] = 1

        datasets = []
        for level in range(num_levels):
            f = 2 ** level
            ds_name = f"s{level}"
            if level == 0:
                lvl_vol = vol
            else:
                # Strided downsample (preserves segment, simpler than mode for test)
                lvl_vol = vol[::f, ::f, ::f].copy()
            ds_dir = os.path.join(zarr_path, ds_name)
            os.makedirs(ds_dir, exist_ok=True)
            arr = ts.open({
                "driver": "zarr3",
                "kvstore": {"driver": "file", "path": ds_dir},
                "metadata": {
                    "shape": list(lvl_vol.shape),
                    "data_type": "uint8",
                    "chunk_grid": {
                        "name": "regular",
                        "configuration": {"chunk_shape": [16, 16, 16]},
                    },
                },
                "create": True, "delete_existing": True,
            }).result()
            arr.write(lvl_vol).result()
            datasets.append({
                "path": ds_name,
                "coordinateTransformations": [
                    {"type": "scale", "scale": [
                        base_voxel_size[0] * f,
                        base_voxel_size[1] * f,
                        base_voxel_size[2] * f,
                    ]},
                ],
            })

        with open(os.path.join(zarr_path, "zarr.json"), "w") as f:
            json.dump({
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {
                    "multiscales": [{
                        "version": "0.5",
                        "axes": [
                            {"name": "z", "type": "space", "unit": "nanometer"},
                            {"name": "y", "type": "space", "unit": "nanometer"},
                            {"name": "x", "type": "space", "unit": "nanometer"},
                        ],
                        "datasets": datasets,
                    }],
                },
            }, f)
        return f"{zarr_path}/s0", zarr_path

    def test_discover_existing_scales_finds_pre_computed_levels(self, tmp_output_dir):
        """_discover_existing_scales returns {1: s0, 2: s1, 4: s2}
        when input is s0 of a 3-level multiscale group."""
        s0_path, _ = self._create_multiscale_zarr(tmp_output_dir, num_levels=3)
        output_dir = os.path.join(tmp_output_dir, "out_discover")
        m = Meshify(
            input_path=s0_path,
            output_directory=output_dir,
            num_workers=1,
            do_multires=False,
            check_mesh_validity=False,
            do_analysis=False,
            do_simplification=False,
        )
        scales = m._discover_existing_scales()
        assert set(scales.keys()) == {1, 2, 4}, (
            f"Expected factors 1,2,4 but got {sorted(scales.keys())}"
        )
        for factor, path in scales.items():
            assert path.endswith(f"s{factor.bit_length() - 1}"), (
                f"Factor {factor} mapped to wrong path {path}"
            )

    def test_discover_returns_empty_for_non_multiscale_input(self, tmp_output_dir):
        """Input zarr without OME-NGFF multiscales metadata yields {}."""
        vol = np.zeros((32, 32, 32), dtype=np.uint64)
        vol[2:30, 2:30, 2:30] = 1
        input_path = _create_zarr_volume(tmp_output_dir, vol)
        m = Meshify(
            input_path=input_path,
            output_directory=os.path.join(tmp_output_dir, "out_no_ms"),
            num_workers=1,
            do_multires=False,
            check_mesh_validity=False,
            do_analysis=False,
            do_simplification=False,
        )
        assert m._discover_existing_scales() == {}

    def test_downsample_strategy_reads_existing_scales(self, tmp_output_dir):
        """End-to-end: when meshing with multires_strategy='downsample'
        on a multiscale input, the per-LOD log lines should mention the
        pre-existing s_k path being read directly — not 'downsample
        factor N' in-worker. Captured by patching _generate_meshes_at_scale
        to record what each LOD was called with."""
        s0_path, _ = self._create_multiscale_zarr(tmp_output_dir, num_levels=3)
        output_dir = os.path.join(tmp_output_dir, "out_downsample")
        m = Meshify(
            input_path=s0_path,
            output_directory=output_dir,
            num_workers=1,
            do_multires=True,
            num_lods=3,
            multires_strategy="downsample",
            target_faces_per_lod0_chunk=200,
            check_mesh_validity=False,
            do_analysis=False,
            do_simplification=True,
        )

        # Record per-LOD inputs to _generate_meshes_at_scale
        calls = []
        orig = m._generate_meshes_at_scale.__func__
        def _spy(self_, output_mesh_dir, downsample_factor=None,
                  target_reduction_override=None, input_dataset_path=None):
            calls.append({
                "output_mesh_dir": output_mesh_dir,
                "downsample_factor": downsample_factor,
                "input_dataset_path": input_dataset_path,
                "target_reduction_override": target_reduction_override,
            })
            return orig(self_, output_mesh_dir,
                         downsample_factor=downsample_factor,
                         target_reduction_override=target_reduction_override,
                         input_dataset_path=input_dataset_path)
        m._generate_meshes_at_scale = _spy.__get__(m, Meshify)

        m.get_meshes()

        # Each LOD should have been called with input_dataset_path pointing
        # to the matching pre-existing scale, NOT with a downsample_factor.
        assert len(calls) == 3, f"Expected 3 LOD calls, got {len(calls)}"
        for k, call in enumerate(calls):
            assert call["input_dataset_path"] is not None, (
                f"LOD {k} did NOT use pre-existing scale (called with "
                f"downsample_factor={call['downsample_factor']})"
            )
            assert call["input_dataset_path"].endswith(f"s{k}"), (
                f"LOD {k} read from {call['input_dataset_path']}, expected s{k}"
            )
            assert call["downsample_factor"] is None, (
                f"LOD {k} should have downsample_factor=None when reading "
                f"existing scale; got {call['downsample_factor']}"
            )


class TestMeshifyNeuroglancerOutput:
    """Test neuroglancer format output from meshify."""

    def test_legacy_neuroglancer_format(self, tmp_output_dir):
        """do_legacy_neuroglancer should write ngmesh files + metadata."""
        vol = np.zeros((32, 32, 32), dtype=np.uint64)
        vol[4:28, 4:28, 4:28] = 1

        input_path = _create_zarr_volume(tmp_output_dir, vol)
        output_dir = os.path.join(tmp_output_dir, "output")

        m = Meshify(
            input_path=input_path,
            output_directory=output_dir,
            num_workers=1,
            do_analysis=False,
            check_mesh_validity=False,
            do_simplification=False,
            n_smoothing_iter=0,
            remove_smallest_components=False,
            do_legacy_neuroglancer=True,
        )
        m.get_meshes()

        mesh_dir = os.path.join(output_dir, "meshes")
        # Should have mesh data file, fragment file, info, and segment_properties
        assert os.path.exists(os.path.join(mesh_dir, "1"))
        assert os.path.exists(os.path.join(mesh_dir, "1:0"))
        assert os.path.exists(os.path.join(mesh_dir, "info"))
        assert os.path.exists(os.path.join(mesh_dir, "segment_properties", "info"))

        import json
        with open(os.path.join(mesh_dir, "info")) as f:
            info = json.load(f)
        assert info["@type"] == "neuroglancer_legacy_mesh"

    def test_singleres_multires_format(self, tmp_output_dir):
        """do_singleres_multires_neuroglancer should write Draco files + index."""
        vol = np.zeros((32, 32, 32), dtype=np.uint64)
        vol[4:28, 4:28, 4:28] = 1

        input_path = _create_zarr_volume(tmp_output_dir, vol)
        output_dir = os.path.join(tmp_output_dir, "output")

        m = Meshify(
            input_path=input_path,
            output_directory=output_dir,
            num_workers=1,
            do_analysis=False,
            check_mesh_validity=False,
            do_simplification=False,
            n_smoothing_iter=0,
            remove_smallest_components=False,
            do_singleres_multires_neuroglancer=True,
        )
        m.get_meshes()

        mesh_dir = os.path.join(output_dir, "meshes")
        # Should have Draco mesh + .index file
        assert os.path.exists(os.path.join(mesh_dir, "1"))
        assert os.path.exists(os.path.join(mesh_dir, "1.index"))
        assert os.path.exists(os.path.join(mesh_dir, "info"))

        import json
        with open(os.path.join(mesh_dir, "info")) as f:
            info = json.load(f)
        assert info["@type"] == "neuroglancer_multilod_draco"


class TestMeshifyWithAnalysis:
    """Test meshify with built-in analysis."""

    def test_analysis_produces_csv(self, tmp_output_dir):
        """With do_analysis=True, a metrics CSV should be generated."""
        vol = np.zeros((32, 32, 32), dtype=np.uint64)
        vol[4:28, 4:28, 4:28] = 1

        input_path = _create_zarr_volume(tmp_output_dir, vol)
        output_dir = os.path.join(tmp_output_dir, "output")

        m = Meshify(
            input_path=input_path,
            output_directory=output_dir,
            num_workers=1,
            do_analysis=True,
            check_mesh_validity=False,
            do_simplification=False,
            n_smoothing_iter=0,
            remove_smallest_components=False,
        )
        m.get_meshes()

        csv_path = os.path.join(output_dir, "metrics", "mesh_metrics.csv")
        assert os.path.exists(csv_path)

        import pandas as pd
        df = pd.read_csv(csv_path)
        assert len(df) == 1
        assert "volume (nm^3)" in df.columns
        assert df["volume (nm^3)"].iloc[0] > 0


class TestMeshifyAnisotropicVoxels:
    """Test meshify with anisotropic voxel sizes."""

    def test_anisotropic_voxel_size(self, tmp_output_dir):
        """Anisotropic voxels should produce correctly scaled meshes."""
        vol = np.zeros((32, 32, 32), dtype=np.uint64)
        vol[4:28, 4:28, 4:28] = 1

        # Anisotropic: z is 2x coarser
        input_path = _create_zarr_volume(
            tmp_output_dir, vol, voxel_size=(8, 8, 16)
        )
        output_dir = os.path.join(tmp_output_dir, "output")

        m = Meshify(
            input_path=input_path,
            output_directory=output_dir,
            num_workers=1,
            do_analysis=False,
            check_mesh_validity=False,
            do_simplification=False,
            n_smoothing_iter=0,
            remove_smallest_components=False,
        )
        m.get_meshes()

        mesh = trimesh.load(os.path.join(output_dir, "meshes", "1.ply"))
        assert len(mesh.faces) > 0

        # Mesh extents should reflect anisotropic voxel size
        extents = mesh.bounds[1] - mesh.bounds[0]
        # z extent should be ~2x the x/y extents
        z_ratio = extents[0] / extents[2]  # vertices are xyz reversed from zyx
        # Should be roughly 0.5 (z is 2x larger voxels, so same voxel count = 2x physical)
        assert 0.3 < z_ratio < 0.7 or 1.5 < z_ratio < 2.5


class TestMeshifyFixedEdgeSimplification:
    """Test fixed-edge simplification across chunk boundaries.

    Verifies that per-block boundary clipping produces matching vertices
    at chunk boundaries so assembly yields watertight meshes.
    """

    def test_cross_chunk_fixed_edge_is_watertight(self, tmp_output_dir):
        """Fixed-edge simplification on a cross-chunk object should be watertight."""
        # Use small chunks (8 voxels) so a 28-voxel object crosses many boundaries
        vol = np.zeros((32, 32, 32), dtype=np.uint64)
        vol[2:30, 2:30, 2:30] = 1

        input_path = _create_zarr_volume(
            tmp_output_dir, vol, voxel_size=(4, 4, 4), chunk_shape=(8, 8, 8)
        )
        output_dir = os.path.join(tmp_output_dir, "output")

        m = Meshify(
            input_path=input_path,
            output_directory=output_dir,
            num_workers=1,
            do_analysis=False,
            check_mesh_validity=False,
            use_fixed_edge_simplification=True,
            do_simplification=True,
            target_reduction=0.9,
            n_smoothing_iter=5,
            remove_smallest_components=False,
        )
        m.get_meshes()

        mesh = trimesh.load(os.path.join(output_dir, "meshes", "1.ply"))
        assert mesh.is_watertight, (
            "Fixed-edge simplified mesh should be watertight after assembly"
        )
        assert mesh.volume > 0

    def test_cross_block_fixed_edge_is_watertight(self, tmp_output_dir):
        """Fixed-edge simplification across processing blocks is watertight.

        Forces the object to span multiple processing blocks by setting
        read_write_block_shape_pixels smaller than the object.  The
        sphere surface crosses block boundaries, exercising the
        boundary clipping and vertex deduplication path.
        """
        vol = np.zeros((32, 32, 32), dtype=np.uint64)
        center = np.array([16, 16, 16], dtype=float)
        for z in range(32):
            for y in range(32):
                for x in range(32):
                    if np.linalg.norm(np.array([z, y, x], dtype=float) - center) < 12:
                        vol[z, y, x] = 1

        input_path = _create_zarr_volume(
            tmp_output_dir, vol, voxel_size=(4, 4, 4), chunk_shape=(8, 8, 8)
        )
        output_dir = os.path.join(tmp_output_dir, "output")

        m = Meshify(
            input_path=input_path,
            output_directory=output_dir,
            num_workers=1,
            do_analysis=False,
            check_mesh_validity=False,
            use_fixed_edge_simplification=True,
            do_simplification=True,
            target_reduction=0.9,
            n_smoothing_iter=5,
            remove_smallest_components=False,
            read_write_block_shape_pixels=[8, 8, 8],
        )
        m.get_meshes()

        mesh = trimesh.load(os.path.join(output_dir, "meshes", "1.ply"))
        assert mesh.is_watertight, (
            "Fixed-edge simplified mesh should be watertight when object "
            "surface crosses processing block boundaries"
        )
        assert mesh.volume > 0

    def test_cross_chunk_fixed_edge_no_spikes(self, tmp_output_dir):
        """Fixed-edge simplification should not produce spike edges."""
        vol = np.zeros((32, 32, 32), dtype=np.uint64)
        vol[2:30, 2:30, 2:30] = 1

        input_path = _create_zarr_volume(
            tmp_output_dir, vol, voxel_size=(4, 4, 4), chunk_shape=(8, 8, 8)
        )
        output_dir = os.path.join(tmp_output_dir, "output")

        m = Meshify(
            input_path=input_path,
            output_directory=output_dir,
            num_workers=1,
            do_analysis=False,
            check_mesh_validity=False,
            use_fixed_edge_simplification=True,
            do_simplification=True,
            target_reduction=0.9,
            n_smoothing_iter=5,
            remove_smallest_components=False,
        )
        m.get_meshes()

        mesh = trimesh.load(os.path.join(output_dir, "meshes", "1.ply"))
        edges = mesh.edges_unique
        edge_lengths = np.linalg.norm(
            mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1
        )
        spike_ratio = edge_lengths.max() / np.median(edge_lengths)
        assert spike_ratio < 10, (
            f"Spike ratio {spike_ratio:.1f}x exceeds limit; "
            "boundary clipping may not align between blocks"
        )
