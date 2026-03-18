"""Shared fixtures for mesh-n-bone tests."""

import json
import numpy as np
import pytest
import trimesh
import os
import tempfile
import zarr


@pytest.fixture
def tiny_cube_mesh():
    """A small cube mesh (box) for basic testing."""
    mesh = trimesh.creation.box(extents=[10, 10, 10])
    mesh.vertices += 50  # offset so not at origin
    return mesh


@pytest.fixture
def tiny_sphere_mesh():
    """A small sphere mesh for testing."""
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=5.0)
    mesh.vertices += 50
    return mesh


@pytest.fixture
def watertight_sphere_mesh():
    """A watertight sphere mesh with enough faces for simplification testing."""
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=20.0)
    mesh.vertices += 100
    return mesh


@pytest.fixture
def tmp_output_dir():
    """Temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_ply_dir(tiny_cube_mesh, tmp_output_dir):
    """Directory containing a sample PLY mesh file."""
    mesh_dir = os.path.join(tmp_output_dir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)
    tiny_cube_mesh.export(os.path.join(mesh_dir, "1.ply"))
    return mesh_dir


@pytest.fixture
def sample_vertices_and_faces():
    """Simple triangle mesh vertices and faces."""
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
    )
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
    return vertices, faces


@pytest.fixture
def multires_mesh_dir(tmp_output_dir):
    """Directory set up with LOD meshes for multires pipeline testing.
    Creates a sphere mesh at s0 and a decimated version at s1."""
    output_path = os.path.join(tmp_output_dir, "multires_test")
    mesh_lods = os.path.join(output_path, "mesh_lods")

    # LOD 0: original mesh
    s0_dir = os.path.join(mesh_lods, "s0")
    os.makedirs(s0_dir, exist_ok=True)
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=50.0)
    mesh.vertices += 100  # offset to positive coordinates
    mesh.export(os.path.join(s0_dir, "1.ply"))

    # LOD 1: simplified mesh (fewer faces)
    import pyfqmr
    simplifier = pyfqmr.Simplify()
    simplifier.setMesh(mesh.vertices, mesh.faces)
    target = max(len(mesh.faces) // 4, 4)
    simplifier.simplify_mesh(target_count=target, aggressiveness=7, preserve_border=False, verbose=False)
    v_dec, f_dec, _ = simplifier.getMesh()
    s1_dir = os.path.join(mesh_lods, "s1")
    os.makedirs(s1_dir, exist_ok=True)
    decimated = trimesh.Trimesh(v_dec, f_dec)
    decimated.export(os.path.join(s1_dir, "1.ply"))

    return output_path


@pytest.fixture
def labeled_volume_3d():
    """A small 3D labeled volume with two objects for downsampling tests."""
    vol = np.zeros((8, 8, 8), dtype=np.uint32)
    # Object 1: fills a 4x4x4 block
    vol[0:4, 0:4, 0:4] = 1
    # Object 2: fills a 4x4x4 block
    vol[4:8, 4:8, 4:8] = 2
    # Scattered voxels of label 3 (thin structure, should vanish with suppress_zero)
    vol[0, 4, 4] = 3
    return vol


@pytest.fixture
def zarr_segmentation(tmp_output_dir):
    """Create a zarr segmentation volume with two labeled objects.

    Returns the path to the zarr dataset (e.g. /tmp/.../test.zarr/seg/s0).
    Volume is 32x32x32 with voxel_size=[1,1,1], containing:
    - Label 1: a solid 20x20x20 cube at [4:24, 4:24, 4:24]
    - Label 2: a solid 8x8x8 cube at [24:32, 24:32, 24:32]
    """
    zarr_path = os.path.join(tmp_output_dir, "test.zarr")
    store = zarr.DirectoryStore(zarr_path)
    root = zarr.open(store, mode="w")
    vol = np.zeros((32, 32, 32), dtype=np.uint32)
    vol[4:24, 4:24, 4:24] = 1
    vol[24:32, 24:32, 24:32] = 2
    root.create_dataset("seg/s0", data=vol, chunks=(16, 16, 16))

    # Write attrs for funlib.persistence
    zattrs_path = os.path.join(zarr_path, "seg", "s0", ".zattrs")
    with open(zattrs_path, "w") as f:
        json.dump(
            {"voxel_size": [1, 1, 1], "offset": [0, 0, 0], "axis_names": ["z", "y", "x"]},
            f,
        )

    return f"{zarr_path}/seg/s0"
