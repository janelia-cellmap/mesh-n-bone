"""Tests for ROI (Region of Interest) support in meshify and multires pipelines."""

import numpy as np
import os
import pytest
import trimesh

from mesh_n_bone.cli import _parse_roi_arg
from mesh_n_bone.multires.multires import _mesh_intersects_roi


class TestParseRoiArg:
    """Tests for CLI --roi argument parsing."""

    def test_valid_six_values(self):
        result = _parse_roi_arg("10,20,30,100,200,300")
        assert result == {"begin": [10.0, 20.0, 30.0], "end": [100.0, 200.0, 300.0]}

    def test_float_values(self):
        result = _parse_roi_arg("1.5,2.5,3.5,10.5,20.5,30.5")
        assert result["begin"] == [1.5, 2.5, 3.5]
        assert result["end"] == [10.5, 20.5, 30.5]

    def test_wrong_number_of_values(self):
        import argparse

        with pytest.raises(argparse.ArgumentTypeError):
            _parse_roi_arg("10,20,30")

    def test_negative_values(self):
        result = _parse_roi_arg("-10,-20,-30,100,200,300")
        assert result["begin"] == [-10.0, -20.0, -30.0]


class TestMeshIntersectsRoi:
    """Tests for spatial ROI filtering of meshes."""

    def test_mesh_inside_roi(self, tmp_output_dir):
        """Mesh fully inside ROI should intersect."""
        mesh = trimesh.creation.box(extents=[10, 10, 10])
        mesh.vertices += 50  # center at (50, 50, 50), bounds [45, 55]
        mesh_path = os.path.join(tmp_output_dir, "inside.ply")
        mesh.export(mesh_path)

        roi_begin = np.array([0, 0, 0])
        roi_end = np.array([100, 100, 100])
        assert _mesh_intersects_roi(mesh_path, roi_begin, roi_end)

    def test_mesh_outside_roi(self, tmp_output_dir):
        """Mesh completely outside ROI should not intersect."""
        mesh = trimesh.creation.box(extents=[10, 10, 10])
        mesh.vertices += 50
        mesh_path = os.path.join(tmp_output_dir, "outside.ply")
        mesh.export(mesh_path)

        roi_begin = np.array([200, 200, 200])
        roi_end = np.array([300, 300, 300])
        assert not _mesh_intersects_roi(mesh_path, roi_begin, roi_end)

    def test_mesh_partially_overlapping_roi(self, tmp_output_dir):
        """Mesh partially overlapping ROI should intersect."""
        mesh = trimesh.creation.box(extents=[10, 10, 10])
        mesh.vertices += 50
        mesh_path = os.path.join(tmp_output_dir, "partial.ply")
        mesh.export(mesh_path)

        # ROI overlaps the corner of the mesh
        roi_begin = np.array([52, 52, 52])
        roi_end = np.array([100, 100, 100])
        assert _mesh_intersects_roi(mesh_path, roi_begin, roi_end)

    def test_mesh_touching_roi_boundary(self, tmp_output_dir):
        """Mesh touching ROI boundary exactly should intersect."""
        mesh = trimesh.creation.box(extents=[10, 10, 10])
        mesh.vertices += 50  # bounds [45, 55]
        mesh_path = os.path.join(tmp_output_dir, "touching.ply")
        mesh.export(mesh_path)

        roi_begin = np.array([55, 55, 55])
        roi_end = np.array([100, 100, 100])
        assert _mesh_intersects_roi(mesh_path, roi_begin, roi_end)

    def test_nonexistent_mesh(self, tmp_output_dir):
        """Loading a mesh that doesn't exist should not intersect."""
        mesh_path = os.path.join(tmp_output_dir, "nonexistent.ply")
        roi_begin = np.array([0, 0, 0])
        roi_end = np.array([100, 100, 100])
        assert not _mesh_intersects_roi(mesh_path, roi_begin, roi_end)


class TestMeshifyRoiConfig:
    """Tests for ROI parsing in Meshify.__init__."""

    def test_roi_begin_end_dict(self, zarr_segmentation, tmp_output_dir):
        """Meshify should accept roi as a dict with begin+end."""
        from mesh_n_bone.meshify.meshify import Meshify

        m = Meshify(
            input_path=zarr_segmentation,
            output_directory=os.path.join(tmp_output_dir, "output"),
            roi={"begin": [0, 0, 0], "end": [16, 16, 16]},
            num_workers=1,
            do_analysis=False,
            check_mesh_validity=False,
            do_simplification=False,
            n_smoothing_iter=0,
            remove_smallest_components=False,
        )
        assert m.roi.get_begin() == (0, 0, 0)
        assert m.roi.get_end() == (16, 16, 16)

    def test_roi_offset_shape_dict(self, zarr_segmentation, tmp_output_dir):
        """Meshify should accept roi as a dict with offset+shape."""
        from mesh_n_bone.meshify.meshify import Meshify

        m = Meshify(
            input_path=zarr_segmentation,
            output_directory=os.path.join(tmp_output_dir, "output"),
            roi={"offset": [4, 4, 4], "shape": [20, 20, 20]},
            num_workers=1,
            do_analysis=False,
            check_mesh_validity=False,
            do_simplification=False,
            n_smoothing_iter=0,
            remove_smallest_components=False,
        )
        assert m.roi.get_begin() == (4, 4, 4)
        assert m.roi.get_end() == (24, 24, 24)

    def test_roi_invalid_dict_raises(self, zarr_segmentation, tmp_output_dir):
        """Invalid dict keys should raise ValueError."""
        from mesh_n_bone.meshify.meshify import Meshify

        with pytest.raises(ValueError, match="must have"):
            Meshify(
                input_path=zarr_segmentation,
                output_directory=os.path.join(tmp_output_dir, "output"),
                roi={"foo": [0, 0, 0], "bar": [10, 10, 10]},
                num_workers=1,
                do_analysis=False,
            )

    def test_roi_invalid_type_raises(self, zarr_segmentation, tmp_output_dir):
        """Non-dict, non-Roi roi should raise ValueError."""
        from mesh_n_bone.meshify.meshify import Meshify

        with pytest.raises(ValueError, match="must be a Roi"):
            Meshify(
                input_path=zarr_segmentation,
                output_directory=os.path.join(tmp_output_dir, "output"),
                roi=[0, 0, 0, 10, 10, 10],
                num_workers=1,
                do_analysis=False,
            )


class TestMeshifyRoiIntegration:
    """Integration tests for meshify with ROI restriction."""

    def test_roi_restricts_to_one_object(self, tmp_output_dir):
        """When ROI covers only one of two objects, only that object should be meshed."""
        from funlib.persistence import prepare_ds
        from funlib.geometry import Coordinate

        vol = np.zeros((32, 32, 32), dtype=np.uint64)
        vol[2:14, 2:14, 2:14] = 1  # object 1 in [2:14]
        vol[18:30, 18:30, 18:30] = 2  # object 2 in [18:30]

        zarr_path = os.path.join(tmp_output_dir, "test.zarr")
        vs = Coordinate(1, 1, 1)
        ds = prepare_ds(
            f"{zarr_path}/labels/s0",
            shape=Coordinate(vol.shape),
            offset=Coordinate(0, 0, 0),
            voxel_size=vs,
            dtype=vol.dtype,
            chunk_shape=Coordinate(16, 16, 16),
        )
        ds[ds.roi] = vol
        input_path = f"{zarr_path}/labels/s0"

        output_dir = os.path.join(tmp_output_dir, "output")
        from mesh_n_bone.meshify.meshify import Meshify

        m = Meshify(
            input_path=input_path,
            output_directory=output_dir,
            roi={"begin": [0, 0, 0], "end": [16, 16, 16]},
            num_workers=1,
            do_analysis=False,
            check_mesh_validity=False,
            do_simplification=False,
            n_smoothing_iter=0,
            remove_smallest_components=False,
        )
        m.get_meshes()

        mesh_dir = os.path.join(output_dir, "meshes")
        meshes = os.listdir(mesh_dir)
        # Only object 1 should be present (object 2 is outside ROI)
        assert "1.ply" in meshes
        assert "2.ply" not in meshes

    def test_roi_covering_full_volume_gets_both_objects(self, tmp_output_dir):
        """ROI covering the full volume should produce all objects."""
        from funlib.persistence import prepare_ds
        from funlib.geometry import Coordinate

        vol = np.zeros((32, 32, 32), dtype=np.uint64)
        vol[2:14, 2:14, 2:14] = 1
        vol[18:30, 18:30, 18:30] = 2

        zarr_path = os.path.join(tmp_output_dir, "test.zarr")
        vs = Coordinate(1, 1, 1)
        ds = prepare_ds(
            f"{zarr_path}/labels/s0",
            shape=Coordinate(vol.shape),
            offset=Coordinate(0, 0, 0),
            voxel_size=vs,
            dtype=vol.dtype,
            chunk_shape=Coordinate(16, 16, 16),
        )
        ds[ds.roi] = vol
        input_path = f"{zarr_path}/labels/s0"

        output_dir = os.path.join(tmp_output_dir, "output")
        from mesh_n_bone.meshify.meshify import Meshify

        m = Meshify(
            input_path=input_path,
            output_directory=output_dir,
            roi={"begin": [0, 0, 0], "end": [32, 32, 32]},
            num_workers=1,
            do_analysis=False,
            check_mesh_validity=False,
            do_simplification=False,
            n_smoothing_iter=0,
            remove_smallest_components=False,
        )
        m.get_meshes()

        mesh_dir = os.path.join(output_dir, "meshes")
        meshes = os.listdir(mesh_dir)
        assert "1.ply" in meshes
        assert "2.ply" in meshes


class TestMultiresRoiFilter:
    """Tests for ROI-based filtering in the multires pipeline."""

    def test_roi_filters_meshes(self, tmp_output_dir):
        """Multires ROI filter should exclude meshes outside the region."""
        # Create two mesh files at different spatial locations
        input_dir = os.path.join(tmp_output_dir, "input")
        os.makedirs(input_dir)

        # Mesh 1: centered at (50, 50, 50)
        mesh1 = trimesh.creation.box(extents=[10, 10, 10])
        mesh1.vertices += 50
        mesh1.export(os.path.join(input_dir, "1.ply"))

        # Mesh 2: centered at (200, 200, 200)
        mesh2 = trimesh.creation.box(extents=[10, 10, 10])
        mesh2.vertices += 200
        mesh2.export(os.path.join(input_dir, "2.ply"))

        # ROI only covers mesh 1
        roi_begin = np.array([0, 0, 0])
        roi_end = np.array([100, 100, 100])

        mesh1_path = os.path.join(input_dir, "1.ply")
        mesh2_path = os.path.join(input_dir, "2.ply")

        assert _mesh_intersects_roi(mesh1_path, roi_begin, roi_end)
        assert not _mesh_intersects_roi(mesh2_path, roi_begin, roi_end)

    def test_roi_in_multires_config(self, tmp_output_dir):
        """ROI in multires config should be parsed correctly."""
        import yaml
        from mesh_n_bone.config import read_multires_config

        config_dir = tmp_output_dir
        config = {
            "required_settings": {
                "input_path": "/tmp/in",
                "output_path": "/tmp/out",
                "num_lods": 3,
            },
            "optional_decimation_settings": {
                "roi": {
                    "begin": [10, 20, 30],
                    "end": [100, 200, 300],
                },
            },
        }
        with open(os.path.join(config_dir, "run-config.yaml"), "w") as f:
            yaml.dump(config, f)

        _, optional, _ = read_multires_config(config_dir)
        assert optional["roi"] is not None
        assert optional["roi"]["begin"] == [10, 20, 30]
        assert optional["roi"]["end"] == [100, 200, 300]

    def test_no_roi_in_config_defaults_to_none(self, tmp_output_dir):
        """Without roi in config, it should default to None."""
        import yaml
        from mesh_n_bone.config import read_multires_config

        config_dir = tmp_output_dir
        config = {
            "required_settings": {
                "input_path": "/tmp/in",
                "output_path": "/tmp/out",
                "num_lods": 2,
            },
        }
        with open(os.path.join(config_dir, "run-config.yaml"), "w") as f:
            yaml.dump(config, f)

        _, optional, _ = read_multires_config(config_dir)
        assert optional["roi"] is None


class TestMeshifyRoiBoundaryProtection:
    """Tests for ROI boundary edge protection during simplification."""

    def test_has_custom_roi_flag(self, zarr_segmentation, tmp_output_dir):
        """Meshify should set has_custom_roi when ROI is provided."""
        from mesh_n_bone.meshify.meshify import Meshify

        m_with = Meshify(
            input_path=zarr_segmentation,
            output_directory=os.path.join(tmp_output_dir, "with"),
            roi={"begin": [0, 0, 0], "end": [16, 16, 16]},
            num_workers=1,
            do_analysis=False,
        )
        assert m_with.has_custom_roi is True

        m_without = Meshify(
            input_path=zarr_segmentation,
            output_directory=os.path.join(tmp_output_dir, "without"),
            num_workers=1,
            do_analysis=False,
        )
        assert m_without.has_custom_roi is False

    def test_repair_skips_hole_closing_with_zero_max_hole_size(self):
        """repair_mesh_pymeshlab with max_hole_size=0 should not close holes."""
        from mesh_n_bone.meshify.meshify import Meshify

        # Create an open mesh (box with top removed)
        vertices = np.array([
            [0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0],
            [0, 0, 10], [10, 0, 10], [10, 10, 10], [0, 10, 10],
        ], dtype=np.float64)
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom
            [0, 1, 5], [0, 5, 4],  # front
            [1, 2, 6], [1, 6, 5],  # right
            [2, 3, 7], [2, 7, 6],  # back
            [3, 0, 4], [3, 4, 7],  # left
        ], dtype=np.int32)

        # With hole closing (default)
        repaired_closed = Meshify.repair_mesh_pymeshlab(
            vertices.copy(), faces.copy(),
            remove_smallest_components=False, max_hole_size=30,
        )

        # Without hole closing
        repaired_open = Meshify.repair_mesh_pymeshlab(
            vertices.copy(), faces.copy(),
            remove_smallest_components=False, max_hole_size=0,
        )

        # The closed version should have more faces (hole was filled)
        assert len(repaired_closed.faces) >= len(repaired_open.faces)
        # The open version should still be open (not watertight)
        assert not repaired_open.is_watertight

    def test_roi_with_simplification_produces_mesh(self, tmp_output_dir):
        """Meshify with ROI + simplification should produce valid output."""
        from funlib.persistence import prepare_ds
        from funlib.geometry import Coordinate
        from mesh_n_bone.meshify.meshify import Meshify

        vol = np.zeros((32, 32, 32), dtype=np.uint64)
        vol[2:30, 2:30, 2:30] = 1  # object spans full volume

        zarr_path = os.path.join(tmp_output_dir, "test.zarr")
        vs = Coordinate(1, 1, 1)
        ds = prepare_ds(
            f"{zarr_path}/labels/s0",
            shape=Coordinate(vol.shape),
            offset=Coordinate(0, 0, 0),
            voxel_size=vs,
            dtype=vol.dtype,
            chunk_shape=Coordinate(16, 16, 16),
        )
        ds[ds.roi] = vol

        output_dir = os.path.join(tmp_output_dir, "output")
        m = Meshify(
            input_path=f"{zarr_path}/labels/s0",
            output_directory=output_dir,
            roi={"begin": [0, 0, 0], "end": [16, 16, 16]},
            num_workers=1,
            do_analysis=False,
            do_simplification=True,
            target_reduction=0.5,
            n_smoothing_iter=5,
            remove_smallest_components=False,
        )
        m.get_meshes()

        mesh_dir = os.path.join(output_dir, "meshes")
        assert os.path.exists(os.path.join(mesh_dir, "1.ply"))
        mesh = trimesh.load(os.path.join(mesh_dir, "1.ply"))
        assert len(mesh.faces) > 0


class TestNeuroglancerCoordinateUnits:
    """Tests for coordinate_units in neuroglancer output."""

    def test_annotations_default_nm(self, tmp_output_dir):
        """Default coordinate_units should be nm."""
        from mesh_n_bone.util.neuroglancer import write_precomputed_annotations
        import json

        output_dir = os.path.join(tmp_output_dir, "annotations")
        coords = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        write_precomputed_annotations(
            output_dir, "point", [1, 2], coords, {"val": np.array([0.1, 0.2])},
        )
        with open(os.path.join(output_dir, "info")) as f:
            info = json.load(f)
        assert info["dimensions"]["x"] == [1, "nm"]

    def test_annotations_custom_units(self, tmp_output_dir):
        """coordinate_units='um' should write um in dimensions."""
        from mesh_n_bone.util.neuroglancer import write_precomputed_annotations
        import json

        output_dir = os.path.join(tmp_output_dir, "annotations_um")
        coords = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        write_precomputed_annotations(
            output_dir, "point", [1, 2], coords, {"val": np.array([0.1, 0.2])},
            coordinate_units="um",
        )
        with open(os.path.join(output_dir, "info")) as f:
            info = json.load(f)
        assert info["dimensions"]["x"] == [1, "um"]
        assert info["dimensions"]["y"] == [1, "um"]
        assert info["dimensions"]["z"] == [1, "um"]

    def test_meshify_stores_coordinate_units(self, zarr_segmentation, tmp_output_dir):
        """Meshify should store coordinate_units from config."""
        from mesh_n_bone.meshify.meshify import Meshify

        m = Meshify(
            input_path=zarr_segmentation,
            output_directory=os.path.join(tmp_output_dir, "output"),
            num_workers=1,
            do_analysis=False,
            coordinate_units="um",
        )
        assert m.coordinate_units == "um"


class TestCliRoiHelp:
    """Test that --roi flag appears in CLI help."""

    def test_meshify_help_shows_roi(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "mesh_n_bone.cli", "meshify", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--roi" in result.stdout

    def test_to_neuroglancer_help_shows_roi(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "mesh_n_bone.cli", "to-neuroglancer", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--roi" in result.stdout
