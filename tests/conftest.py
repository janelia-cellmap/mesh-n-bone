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
    root = zarr.open_group(zarr_path, mode="w")
    vol = np.zeros((32, 32, 32), dtype=np.uint32)
    vol[4:24, 4:24, 4:24] = 1
    vol[24:32, 24:32, 24:32] = 2
    arr = root.create_array("seg/s0", data=vol, chunks=(16, 16, 16))
    arr.attrs["voxel_size"] = [1, 1, 1]
    arr.attrs["offset"] = [0, 0, 0]
    arr.attrs["axis_names"] = ["z", "y", "x"]

    return f"{zarr_path}/seg/s0"


def _make_zarr_cube(tmp_dir, voxel_size, offset, vol_shape, cube_slice, label=1,
                    chunk_shape=None):
    """Helper: create a zarr volume with a single labeled cube at known position.

    Args:
        tmp_dir: parent directory for zarr
        voxel_size: [vz, vy, vx] in ZYX order
        offset: [oz, oy, ox] in ZYX order
        vol_shape: (nz, ny, nx)
        cube_slice: tuple of 3 slices defining the cube in voxel indices
        label: label value to assign
        chunk_shape: zarr chunk shape (defaults to vol_shape)

    Returns:
        (zarr_path_str, expected_bounds_xyz, expected_center_xyz)
        where expected values account for marching cubes half-voxel offset.
    """
    if chunk_shape is None:
        chunk_shape = vol_shape
    zarr_path = os.path.join(tmp_dir, "cube.zarr")
    root = zarr.open_group(zarr_path, mode="w")
    vol = np.zeros(vol_shape, dtype=np.uint32)
    vol[cube_slice] = label
    arr = root.create_array("seg/s0", data=vol, chunks=chunk_shape)
    arr.attrs["voxel_size"] = list(voxel_size)
    arr.attrs["offset"] = list(offset)
    arr.attrs["axis_names"] = ["z", "y", "x"]

    # Compute expected mesh bounds in XYZ.
    vs = np.array(voxel_size, dtype=float)  # ZYX
    off = np.array(offset, dtype=float)  # ZYX
    starts = np.array([s.start for s in cube_slice], dtype=float)
    stops = np.array([s.stop for s in cube_slice], dtype=float)

    expected_min_zyx = off + (starts - 0.5) * vs
    expected_max_zyx = off + (stops - 0.5) * vs
    expected_center_zyx = (expected_min_zyx + expected_max_zyx) / 2

    expected_min_xyz = expected_min_zyx[::-1]
    expected_max_xyz = expected_max_zyx[::-1]
    expected_center_xyz = expected_center_zyx[::-1]

    return f"{zarr_path}/seg/s0", expected_min_xyz, expected_max_xyz, expected_center_xyz


@pytest.fixture
def zarr_cube_with_offset(tmp_output_dir):
    """Zarr volume with a cube at known position with non-zero offset and non-unit voxel size.

    Volume: 64x64x64, voxel_size=[4,4,4], offset=[100,200,300] (ZYX).
    Cube: label 1 at voxels [8:48, 8:48, 8:48].
    Chunks: 16x16x16 so data spans 4x4x4=64 zarr chunks and mesh blocks,
    ensuring blockwise assembly is tested.
    """
    return _make_zarr_cube(
        tmp_output_dir,
        voxel_size=[4, 4, 4],
        offset=[100, 200, 300],
        vol_shape=(64, 64, 64),
        cube_slice=(slice(8, 48), slice(8, 48), slice(8, 48)),
        chunk_shape=(16, 16, 16),
    )


@pytest.fixture
def zarr_sphere(tmp_output_dir):
    """Zarr volume containing a voxelized sphere for volume verification.

    Volume: 64x64x64, voxel_size=[2,2,2], offset=[0,0,0].
    Sphere: label 1, radius 20 voxels, center at voxel [32,32,32].
    """
    zarr_path = os.path.join(tmp_output_dir, "sphere.zarr")
    root = zarr.open_group(zarr_path, mode="w")

    vol = np.zeros((64, 64, 64), dtype=np.uint32)
    center = np.array([32, 32, 32])
    radius = 20  # voxels
    zz, yy, xx = np.mgrid[0:64, 0:64, 0:64]
    dist = np.sqrt((zz - center[0])**2 + (yy - center[1])**2 + (xx - center[2])**2)
    vol[dist <= radius] = 1
    arr = root.create_array("seg/s0", data=vol, chunks=(16, 16, 16))
    arr.attrs["voxel_size"] = [2, 2, 2]
    arr.attrs["offset"] = [0, 0, 0]
    arr.attrs["axis_names"] = ["z", "y", "x"]

    expected_center_xyz = np.array([64.0, 64.0, 64.0])
    expected_radius_world = 40.0
    expected_volume = (4.0 / 3.0) * np.pi * expected_radius_world**3

    return f"{zarr_path}/seg/s0", expected_center_xyz, expected_radius_world, expected_volume


def _make_zarr_cube_ome_ngff(tmp_dir, voxel_size, offset, vol_shape, cube_slice,
                              label=1, chunk_shape=None):
    """Like _make_zarr_cube but stores metadata in OME-NGFF multiscales format
    (parent .zattrs) instead of per-dataset .zattrs. This mimics how most
    cellmap/OME-Zarr datasets store their coordinate transforms."""
    if chunk_shape is None:
        chunk_shape = vol_shape
    zarr_path = os.path.join(tmp_dir, "ome.zarr")
    root = zarr.open_group(zarr_path, mode="w")
    vol = np.zeros(vol_shape, dtype=np.uint32)
    vol[cube_slice] = label
    root.create_array("seg/s0", data=vol, chunks=chunk_shape)

    # Write OME-NGFF multiscales metadata on the PARENT group
    seg_group = root["seg"]
    seg_group.attrs["multiscales"] = [{
        "axes": [
            {"name": "z", "type": "space", "unit": "nanometer"},
            {"name": "y", "type": "space", "unit": "nanometer"},
            {"name": "x", "type": "space", "unit": "nanometer"},
        ],
        "datasets": [{
            "coordinateTransformations": [
                {"scale": list(voxel_size), "type": "scale"},
                {"translation": list(offset), "type": "translation"},
            ],
            "path": "s0",
        }],
        "version": "0.4",
    }]

    # NO attrs on the dataset itself — open_dataset will see voxel_size=(1,1,1)

    vs = np.array(voxel_size, dtype=float)
    off = np.array(offset, dtype=float)
    starts = np.array([s.start for s in cube_slice], dtype=float)
    stops = np.array([s.stop for s in cube_slice], dtype=float)
    expected_min_zyx = off + (starts - 0.5) * vs
    expected_max_zyx = off + (stops - 0.5) * vs
    expected_center_zyx = (expected_min_zyx + expected_max_zyx) / 2

    return (
        f"{zarr_path}/seg/s0",
        expected_min_zyx[::-1],
        expected_max_zyx[::-1],
        expected_center_zyx[::-1],
    )


@pytest.fixture
def zarr_cube_ome_ngff(tmp_output_dir):
    """Zarr volume with OME-NGFF multiscales metadata (no per-dataset .zattrs).

    Volume: 64x64x64, voxel_size=[8,8,8], offset=[100,100,100] (ZYX).
    Cube: label 1 at voxels [8:48, 8:48, 8:48].
    Chunks: 16x16x16 for multi-block testing.
    """
    return _make_zarr_cube_ome_ngff(
        tmp_output_dir,
        voxel_size=[8, 8, 8],
        offset=[100, 100, 100],
        vol_shape=(64, 64, 64),
        cube_slice=(slice(8, 48), slice(8, 48), slice(8, 48)),
        chunk_shape=(16, 16, 16),
    )
