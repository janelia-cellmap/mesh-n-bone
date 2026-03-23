"""Tests for multires pipeline components."""

import numpy as np
import os
import pytest
import trimesh

from mesh_n_bone.multires.decomposition import my_slice_faces_plane
from mesh_n_bone.config import read_multires_config


class TestSliceFacesPlane:
    def test_slice_retains_vertices(self):
        mesh = trimesh.creation.box(extents=[10, 10, 10])
        vertices = mesh.vertices + 5  # shift to positive octant
        faces = mesh.faces

        plane_normal = np.array([1, 0, 0])
        plane_origin = np.array([3, 0, 0])

        v_out, f_out = my_slice_faces_plane(vertices, faces, plane_normal, plane_origin)
        assert len(v_out) > 0
        assert len(f_out) > 0
        # All retained vertices should be >= plane
        assert np.all(v_out[:, 0] >= 3 - 1e-6)

    def test_slice_empty_mesh(self):
        v_out, f_out = my_slice_faces_plane(
            np.empty((0, 3)), np.empty((0, 3), dtype=int),
            np.array([1, 0, 0]), np.array([0, 0, 0])
        )
        assert len(v_out) == 0
        assert len(f_out) == 0

    def test_slice_all_on_one_side(self):
        """When the entire mesh is on one side of the plane, should return all."""
        mesh = trimesh.creation.box(extents=[2, 2, 2])
        vertices = mesh.vertices + 10  # all positive, far from origin
        faces = mesh.faces

        plane_normal = np.array([1, 0, 0])
        plane_origin = np.array([0, 0, 0])

        v_out, f_out = my_slice_faces_plane(vertices, faces, plane_normal, plane_origin)
        assert len(v_out) > 0


class TestMultiresConfig:
    def test_read_config_missing_dir(self):
        with pytest.raises(FileNotFoundError):
            read_multires_config("/nonexistent/path")

    def test_box_size_scalar(self, tmp_output_dir):
        import yaml
        config_dir = tmp_output_dir
        config = {
            "required_settings": {
                "input_path": "/tmp/in",
                "output_path": "/tmp/out",
                "num_lods": 3,
            },
            "optional_decimation_settings": {
                "box_size": 100,
            },
        }
        with open(os.path.join(config_dir, "run-config.yaml"), "w") as f:
            yaml.dump(config, f)

        required, optional, _ = read_multires_config(config_dir)
        assert required["num_lods"] == 3
        np.testing.assert_array_equal(optional["box_size"], [100.0, 100.0, 100.0])

    def test_box_size_per_axis(self, tmp_output_dir):
        import yaml
        config_dir = tmp_output_dir
        config = {
            "required_settings": {
                "input_path": "/tmp/in",
                "output_path": "/tmp/out",
                "num_lods": 2,
            },
            "optional_decimation_settings": {
                "box_size": [20, 40, 60],
            },
        }
        with open(os.path.join(config_dir, "run-config.yaml"), "w") as f:
            yaml.dump(config, f)

        _, optional, _ = read_multires_config(config_dir)
        np.testing.assert_array_equal(optional["box_size"], [20.0, 40.0, 60.0])

    def test_box_size_scalar_broadcasts_to_3d(self, tmp_output_dir):
        import yaml
        config_dir = tmp_output_dir
        config = {
            "required_settings": {
                "input_path": "/tmp/in",
                "output_path": "/tmp/out",
                "num_lods": 2,
            },
            "optional_decimation_settings": {
                "box_size": 50,
            },
        }
        with open(os.path.join(config_dir, "run-config.yaml"), "w") as f:
            yaml.dump(config, f)

        _, optional, _ = read_multires_config(config_dir)
        np.testing.assert_array_equal(optional["box_size"], [50.0, 50.0, 50.0])
