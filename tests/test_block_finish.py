"""Segments that lie wholly inside one block are finished by that block's
worker, and come out exactly as assembly would have made them."""

import logging
import os

import numpy as np
import pytest
import zarr
from zmesh import Mesher

from mesh_n_bone.meshify import meshify as meshify_module
from mesh_n_bone.meshify.meshify import Meshify


def _blob_volume(tmp_path):
    """48^3 voxels in 27 blocks of 16^3: many small blobs, some crossing
    block faces, one segment in two separate parts in different blocks,
    and one large blob wholly inside a block."""
    rng = np.random.default_rng(0)
    vol = np.zeros((48, 48, 48), dtype=np.uint32)
    label = 1
    for _ in range(140):
        size = rng.integers(2, 5, size=3)
        corner = rng.integers(0, 48 - size, size=3)
        region = tuple(slice(c, c + s) for c, s in zip(corner, size))
        if not vol[region].any():
            vol[region] = label
            label += 1
    # Segment 500: two parts, each inside its own block (blocks 0 and 26).
    vol[4:8, 4:8, 4:8] = 500
    vol[36:40, 36:40, 36:40] = 500
    # Segment 600: a large blob wholly inside the centre block.
    zz, yy, xx = np.mgrid[16:32, 16:32, 16:32]
    vol[16:32, 16:32, 16:32][((zz - 24) ** 2 + (yy - 24) ** 2 + (xx - 24) ** 2) < 5 ** 2] = 600
    root = zarr.open_group(str(tmp_path / "blobs.zarr"), mode="w")
    arr = root.create_array("labels/s0", data=vol, chunks=(16, 16, 16))
    arr.attrs["voxel_size"] = [8, 8, 8]
    arr.attrs["offset"] = [0, 0, 0]
    return str(tmp_path / "blobs.zarr" / "labels" / "s0")


def _mesh_all(path, out, **kwargs):
    settings = dict(
        input_path=path, output_directory=out, num_workers=1,
        do_multires=False, do_analysis=False, check_mesh_validity=False,
        read_write_block_shape_pixels=[16, 16, 16],
        target_reduction=0.5, n_smoothing_iter=2,
    )
    settings.update(kwargs)
    Meshify(**settings).get_meshes()
    meshes = os.path.join(out, "meshes")
    contents = {}
    for root, _, files in os.walk(meshes):
        for name in files:
            path = os.path.join(root, name)
            with open(path, "rb") as f:
                contents[os.path.relpath(path, meshes)] = f.read()
    return contents


MODES = {
    "default": {},
    "fixed_edge": {"use_fixed_edge_simplification": True},
    "no_simplification": {"do_simplification": False, "n_smoothing_iter": 0},
    "legacy_neuroglancer": {"do_legacy_neuroglancer": True},
    "validity_checks": {"check_mesh_validity": True},
}


@pytest.mark.parametrize("mode", sorted(MODES))
def test_block_finished_segments_match_assembled_ones(tmp_path, monkeypatch, caplog, mode):
    path = _blob_volume(tmp_path)
    # Small enough that segment 600 is too big to finish in its block.
    monkeypatch.setattr(meshify_module, "LOCAL_FINISH_MAX_FACES", 400)
    with caplog.at_level(logging.INFO):
        finished_in_blocks = _mesh_all(path, str(tmp_path / "blocks"), **MODES[mode])
    summary = [r.message for r in caplog.records if r.message.startswith("Assembly: ")]
    caplog.clear()

    monkeypatch.setattr(meshify_module, "LOCAL_FINISH_MAX_FACES", 0)
    with caplog.at_level(logging.INFO):
        all_assembled = _mesh_all(path, str(tmp_path / "assembly"), **MODES[mode])
    summary_off = [r.message for r in caplog.records if r.message.startswith("Assembly: ")]

    assert sorted(finished_in_blocks) == sorted(all_assembled)
    differing = [name for name in all_assembled if finished_in_blocks[name] != all_assembled[name]]
    assert not differing, f"{len(differing)} outputs differ, e.g. {differing[:5]}"

    finished = int(summary[0].split()[1])
    assert finished > 50, summary
    assert "1 of them also finished early in a block" in summary[0]  # segment 500
    assert summary_off[0].startswith("Assembly: 0 segments were finished")


def test_a_segment_in_two_blocks_keeps_both_parts(tmp_path, monkeypatch):
    import trimesh

    path = _blob_volume(tmp_path)
    out = str(tmp_path / "out")
    _mesh_all(path, out, target_reduction=0.0, n_smoothing_iter=0,
              remove_smallest_components=False)
    mesh = trimesh.load(os.path.join(out, "meshes", "500.ply"), process=False)
    parts = mesh.split(only_watertight=False)
    assert len(parts) == 2
    # The parts are the two 4^3 cubes, 32 voxels apart on every axis.
    centres = sorted(tuple(p.bounds.mean(axis=0) / 8) for p in parts)
    np.testing.assert_allclose(np.subtract(centres[1], centres[0]), (32, 32, 32))
    assert all(np.allclose(p.extents / 8, 4) for p in parts)


class TestInteriorTest:
    """A segment counts as interior exactly when no neighbour's read box
    holds any of its voxels."""

    @staticmethod
    def _vertices(core, index):
        """zmesh vertices of one voxel at read-box index *index* on every
        axis, in a read box of *core* + 2 voxels."""
        block = np.zeros((core + 2,) * 3, dtype=np.uint32)
        block[index, index, index] = 1
        mesher = Mesher((1.0, 1.0, 1.0))
        mesher.mesh(block, close=False)
        return mesher.get_mesh(1).vertices

    @pytest.mark.parametrize("index, interior", [
        (1, False),   # the block's first voxel: the low neighbour reads it
        (2, True),
        (15, True),
        (16, False),  # the block's last voxel: the high neighbour reads it
    ])
    def test_one_voxel(self, index, interior):
        vertices = self._vertices(16, index)
        assert meshify_module._segment_is_interior(vertices, (16, 16, 16), (1.0, 1.0, 1.0)) is interior

    def test_no_vertices_is_not_interior(self):
        assert not meshify_module._segment_is_interior(np.empty((0, 3)), (16,) * 3, (1.0,) * 3)


def test_an_earlier_runs_packs_are_not_read(tmp_path):
    """A run into a directory an earlier, killed run left behind meshes
    from scratch: the old packs, manifests and assembly markers go."""
    path = _blob_volume(tmp_path)
    clean = _mesh_all(path, str(tmp_path / "clean"))

    out = tmp_path / "reused"
    stale = out / "tmp_chunked"
    meshify_module._write_block_pack(str(stale), 99, [(77777, b"not a ply", 3, 1, 0)])
    (stale / meshify_module._ASSEMBLED_SUBDIR).mkdir()
    for k in range(8):
        (stale / meshify_module._ASSEMBLED_SUBDIR / f"wave_1_batch_{k}").write_text("")
    assert _mesh_all(path, str(out)) == clean
