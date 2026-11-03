"""Volumes stored in another axis order (an x, y, z OME-Zarr, as
TensorSwitch writes) are read z, y, x and mesh where the z, y, x store
of the same data does."""

import os

import numpy as np
import pytest
import trimesh
import zarr

from mesh_n_bone.meshify.meshify import Meshify
from mesh_n_bone.util.zarr_io import open_dataset


def _write_ome_group(path, levels, axes="zyx"):
    """A zarr v3 OME-NGFF 0.5 group whose arrays are stored in ``axes``
    order. ``levels`` are ``(name, data, voxel_size, translation)`` in
    z, y, x; an ``"xyz"`` group stores each array transposed, with its
    ``dimension_names``, scale and translation in that order."""
    perm = {"zyx": (0, 1, 2), "xyz": (2, 1, 0)}[axes]
    root = zarr.open_group(path, mode="w", zarr_format=3)
    datasets = []
    for name, data, voxel_size, translation in levels:
        stored = np.ascontiguousarray(np.transpose(data, perm))
        arr = root.create_array(
            name, shape=stored.shape, dtype=stored.dtype,
            chunks=tuple(min(8, s) for s in stored.shape),
            dimension_names=list(axes),
        )
        arr[:] = stored
        datasets.append({
            "path": name,
            "coordinateTransformations": [
                {"type": "scale", "scale": [float(voxel_size[i]) for i in perm]},
                {"type": "translation",
                 "translation": [float(translation[i]) for i in perm]},
            ],
        })
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [{
            "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in axes],
            "datasets": datasets,
        }],
    }
    return path


# An asymmetric cube, so a transposed read would show: voxel size
# (10, 4, 4) nm (z, y, x), voxel 0's corner at the origin, label at
# voxels z [2, 6), y [8, 20), x [4, 28).
VOXEL = np.array([10.0, 4.0, 4.0])
CUBE = (slice(2, 6), slice(8, 20), slice(4, 28))
TRUTH_MIN_XYZ = (16.0, 32.0, 20.0)
TRUTH_MAX_XYZ = (112.0, 80.0, 60.0)


def _cube_levels():
    data = np.zeros((16, 32, 32), dtype=np.uint32)
    data[CUBE] = 1
    return [("s0", data, VOXEL, VOXEL / 2)]


def _mesh(path, out):
    m = Meshify(
        input_path=path, output_directory=out, num_workers=1,
        do_multires=False, do_simplification=False, check_mesh_validity=False,
        do_analysis=False, n_smoothing_iter=0,
    )
    m.get_meshes()
    return trimesh.load(os.path.join(out, "meshes", "1.ply"))


class TestStoredAxisOrder:
    def test_xyz_store_opens_zyx(self, tmp_output_dir):
        path = _write_ome_group(
            os.path.join(tmp_output_dir, "xyz.zarr"), _cube_levels(), axes="xyz",
        )
        ds = open_dataset(path)
        assert ds.shape == (16, 32, 32)
        assert tuple(ds.voxel_size) == (10.0, 4.0, 4.0)
        assert tuple(ds.translation) == (5.0, 2.0, 2.0)
        assert ds.chunk_shape == (8, 8, 8)

    @pytest.mark.parametrize("axes", ["zyx", "xyz"])
    def test_cube_meshes_where_it_is_whatever_the_stored_order(
        self, tmp_output_dir, axes,
    ):
        path = _write_ome_group(
            os.path.join(tmp_output_dir, f"{axes}.zarr"), _cube_levels(), axes=axes,
        )
        mesh = _mesh(path, os.path.join(tmp_output_dir, f"out_{axes}"))
        tol = 0.1 * VOXEL.min()
        np.testing.assert_allclose(mesh.bounds[0], TRUTH_MIN_XYZ, atol=tol)
        np.testing.assert_allclose(mesh.bounds[1], TRUTH_MAX_XYZ, atol=tol)

    def test_xyz_pyramid_levels_are_keyed_zyx(self, tmp_output_dir):
        """A TensorSwitch-style pyramid: s0 at 16, 16, 45 nm (x, y, z),
        then the finer axes halved, then every axis. Its levels are the
        first LODs, with per-axis factors in z, y, x."""
        s0 = np.zeros((8, 32, 32), dtype=np.uint32)
        s0[2:6, 8:24, 8:24] = 1
        vs = np.array([45.0, 16.0, 16.0])
        levels = [
            ("s0", s0, vs, vs / 2),
            ("s1", s0[:, ::2, ::2], vs * (1, 2, 2), vs * (1, 2, 2) / 2),
            ("s2", s0[::2, ::4, ::4], vs * (2, 4, 4), vs * (2, 4, 4) / 2),
        ]
        path = _write_ome_group(
            os.path.join(tmp_output_dir, "pyr.zarr"), levels, axes="xyz",
        )
        m = Meshify(
            input_path=path, output_directory=os.path.join(tmp_output_dir, "out"),
            num_workers=1, do_multires=False, do_simplification=False,
            check_mesh_validity=False, do_analysis=False,
        )
        assert tuple(m.true_voxel_size) == (45.0, 16.0, 16.0)
        scales = m._discover_existing_scales()
        assert {f: p.rsplit("/", 1)[-1] for f, p in scales.items()} == {
            (1, 1, 1): "s0", (1, 2, 2): "s1", (2, 4, 4): "s2",
        }
        assert m._lod_factors(4) == [(1, 1, 1), (1, 2, 2), (2, 4, 4), (4, 8, 8)]

    def test_two_dimensional_array_is_refused(self, tmp_output_dir):
        path = os.path.join(tmp_output_dir, "flat.zarr")
        root = zarr.open_group(path, mode="w", zarr_format=3)
        arr = root.create_array("s0", shape=(8, 8), dtype="uint8", chunks=(8, 8))
        arr[:] = 0
        with pytest.raises(ValueError, match="three spatial axes"):
            open_dataset(f"{path}/s0")
