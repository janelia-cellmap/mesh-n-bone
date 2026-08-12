"""Tests for the OME-NGFF multiscale pyramid auto-builder.

Focus areas:
- Per-LOD per-axis factor calculation for anisotropic voxels.
- ROI alignment (snap mode + halo mode).
- OME-NGFF metadata correctness (scale + translation per level).
- Fence-post correctness: per-block downsamples match a global single-pass
  downsample bit-exact when blocks are 2^k-aligned.
"""

import json
import os

import numpy as np
import pytest

from mesh_n_bone.meshify.downsample import (
    downsample_labels_3d,
    downsample_binary_3d_suppress_zero,
)
from mesh_n_bone.util.pyramid_builder import (
    align_roi_voxels,
    build_missing_pyramid_levels,
    build_multiscales_metadata,
    downsample_super_chunk,
    per_lod_factors_for_anisotropy,
    write_multiscales_metadata,
)


class TestAnisotropyFactors:
    def test_isotropic_doubles_every_axis(self):
        f = per_lod_factors_for_anisotropy(np.array([8.0, 8.0, 8.0]), num_lods=4)
        assert f == [(1, 1, 1), (2, 2, 2), (4, 4, 4), (8, 8, 8)]

    def test_anisotropic_2to1_z_axis(self):
        # voxel z=20, xy=8: z is coarsest → downsample xy first
        f = per_lod_factors_for_anisotropy(np.array([20.0, 8.0, 8.0]), num_lods=4)
        # LOD 0: (1,1,1)         voxel = [20, 8, 8]
        # LOD 1: only axes with 8<=1.5*8=12 qualify → y,x → step (1,2,2)
        #   voxel = [20, 16, 16]
        # LOD 2: 16<=1.5*16=24, 20<=24 → all qualify → step (2,2,2)
        #   voxel = [40, 32, 32]
        # LOD 3: 32<=48, 40<=48 → all → step (2,2,2)
        #   voxel = [80, 64, 64]
        assert f[0] == (1, 1, 1)
        assert f[1] == (1, 2, 2)
        assert f[2] == (2, 4, 4)
        assert f[3] == (4, 8, 8)

    def test_extreme_anisotropy_multiple_xy_steps(self):
        # voxel z=40, xy=8: z is 5x — needs two xy downsamples before z
        f = per_lod_factors_for_anisotropy(np.array([40.0, 8.0, 8.0]), num_lods=4)
        # LOD 0: (1,1,1) voxel=[40,8,8]
        # LOD 1: step (1,2,2) → [40,16,16]
        # LOD 2: 16<=1.5*16=24, 40>24 → step (1,2,2) → [40,32,32]
        # LOD 3: 32<=48, 40<=48 → step (2,2,2) → [80,64,64]
        assert f[1] == (1, 2, 2)
        assert f[2] == (1, 4, 4)
        assert f[3] == (2, 8, 8)


class TestRoiAlignment:
    def test_aligned_roi_snap_is_identity(self):
        origin = np.array([0, 0, 0])
        shape = np.array([64, 64, 64])
        out_o, out_s, _, _ = align_roi_voxels(
            origin, shape, np.array([8, 8, 8]), "snap",
        )
        assert out_o.tolist() == [0, 0, 0]
        assert out_s.tolist() == [64, 64, 64]

    def test_unaligned_roi_snap_drops_edges(self):
        # origin 3 → snap up to 8 (drop 5 voxels off the start)
        # extent 70 → end 73 → snap down to 72 (drop 1 voxel off the end)
        # net snapped: origin 8, extent 64
        origin = np.array([3, 3, 3])
        shape = np.array([70, 70, 70])
        out_o, out_s, _, _ = align_roi_voxels(
            origin, shape, np.array([8, 8, 8]), "snap",
        )
        assert out_o.tolist() == [8, 8, 8]
        assert out_s.tolist() == [64, 64, 64]

    def test_unaligned_roi_halo_rounds_outward(self):
        # origin 3 → halo round DOWN to 0
        # extent 70 → end 73 → round UP to 80
        # net: output spans 0..80
        origin = np.array([3, 3, 3])
        shape = np.array([70, 70, 70])
        out_o, out_s, read_o, read_s = align_roi_voxels(
            origin, shape, np.array([8, 8, 8]), "halo",
        )
        assert out_o.tolist() == [0, 0, 0]
        assert out_s.tolist() == [80, 80, 80]
        # In halo mode, read_* matches out_* (caller clips to dataset bounds)
        assert read_o.tolist() == [0, 0, 0]
        assert read_s.tolist() == [80, 80, 80]

    def test_anisotropic_factor(self):
        origin = np.array([3, 5, 5])
        shape = np.array([20, 32, 32])
        # max factor (4, 8, 8)
        out_o, out_s, _, _ = align_roi_voxels(
            origin, shape, np.array([4, 8, 8]), "snap",
        )
        # z: snap 3 -> 4, end 23 -> 20, shape 16
        # y: snap 5 -> 8, end 37 -> 32, shape 24
        # x: snap 5 -> 8, end 37 -> 32, shape 24
        assert out_o.tolist() == [4, 8, 8]
        assert out_s.tolist() == [16, 24, 24]


class TestMultiscalesMetadata:
    def test_isotropic_translations(self):
        # s0_vs = 8, s0_tr = 4 (= corner + vs/2 of an origin at 0)
        md = build_multiscales_metadata(
            s0_voxel_size_zyx=[8.0, 8.0, 8.0],
            s0_translation_zyx=[4.0, 4.0, 4.0],
            per_lod_factors=[(1, 1, 1), (2, 2, 2), (4, 4, 4), (8, 8, 8)],
        )
        ds = md["multiscales"][0]["datasets"]
        # s0: scale=8, tr=4
        assert ds[0]["coordinateTransformations"][0]["scale"] == [8.0, 8.0, 8.0]
        assert ds[0]["coordinateTransformations"][1]["translation"] == [4.0, 4.0, 4.0]
        # s1: scale=16, tr = 4 + 0.5*8*(2-1) = 8
        assert ds[1]["coordinateTransformations"][0]["scale"] == [16.0, 16.0, 16.0]
        assert ds[1]["coordinateTransformations"][1]["translation"] == [8.0, 8.0, 8.0]
        # s2: scale=32, tr = 4 + 0.5*8*(4-1) = 16
        assert ds[2]["coordinateTransformations"][0]["scale"] == [32.0, 32.0, 32.0]
        assert ds[2]["coordinateTransformations"][1]["translation"] == [16.0, 16.0, 16.0]
        # s3: scale=64, tr = 4 + 0.5*8*(8-1) = 32
        assert ds[3]["coordinateTransformations"][0]["scale"] == [64.0, 64.0, 64.0]
        assert ds[3]["coordinateTransformations"][1]["translation"] == [32.0, 32.0, 32.0]

    def test_anisotropic_translations_per_axis(self):
        # s0_vs = [20, 8, 8], s0_tr = [10, 4, 4]
        # LOD 1 cumulative factor (1, 2, 2) → vs=[20,16,16], tr=[10, 4+0.5*8*1, 4+0.5*8*1]=[10, 8, 8]
        md = build_multiscales_metadata(
            s0_voxel_size_zyx=[20.0, 8.0, 8.0],
            s0_translation_zyx=[10.0, 4.0, 4.0],
            per_lod_factors=[(1, 1, 1), (1, 2, 2), (2, 4, 4)],
        )
        ds = md["multiscales"][0]["datasets"]
        assert ds[1]["coordinateTransformations"][0]["scale"] == [20.0, 16.0, 16.0]
        assert ds[1]["coordinateTransformations"][1]["translation"] == [10.0, 8.0, 8.0]
        # LOD 2 factor (2, 4, 4) → vs=[40,32,32], tr=[10+0.5*20*1, 4+0.5*8*3, 4+0.5*8*3]=[20, 16, 16]
        assert ds[2]["coordinateTransformations"][0]["scale"] == [40.0, 32.0, 32.0]
        assert ds[2]["coordinateTransformations"][1]["translation"] == [20.0, 16.0, 16.0]

    def test_axes_metadata(self):
        md = build_multiscales_metadata(
            s0_voxel_size_zyx=[1.0, 1.0, 1.0],
            s0_translation_zyx=[0.0, 0.0, 0.0],
            per_lod_factors=[(1, 1, 1)],
        )
        axes = md["multiscales"][0]["axes"]
        assert [a["name"] for a in axes] == ["z", "y", "x"]
        assert all(a["type"] == "space" and a["unit"] == "nanometer" for a in axes)


class TestFencepostCorrectness:
    """Block-by-block downsample of an aligned volume must equal a
    single-pass global downsample, voxel-by-voxel."""

    def _build_synthetic_volume(self, shape=(64, 64, 64), num_labels=8, seed=42):
        rng = np.random.default_rng(seed)
        return rng.integers(low=0, high=num_labels, size=shape, dtype=np.uint8)

    def test_super_chunk_matches_global_isotropic(self, tmp_path):
        vol = self._build_synthetic_volume(shape=(32, 32, 32))
        per_lod = [(1, 1, 1), (2, 2, 2), (4, 4, 4)]
        # Global single-pass reference
        ref_s1, _ = downsample_labels_3d(vol, (2, 2, 2))
        ref_s2, _ = downsample_labels_3d(vol, (4, 4, 4))

        # Build per-chunk via the super-chunk worker on 16-voxel super-chunks
        super_chunk_shape = np.array([16, 16, 16])  # = out_chunk(4)*max_factor(4)
        out_chunk = np.array([4, 4, 4])
        result_s1 = np.zeros((16, 16, 16), dtype=vol.dtype)
        result_s2 = np.zeros((8, 8, 8), dtype=vol.dtype)
        for z0 in range(0, 32, 16):
            for y0 in range(0, 32, 16):
                for x0 in range(0, 32, 16):
                    s0_block = vol[z0:z0+16, y0:y0+16, x0:x0+16]
                    sc_origin = np.array([z0, y0, x0])
                    downs = downsample_super_chunk(
                        s0_block, sc_origin, per_lod, downsample_labels_3d, out_chunk,
                    )
                    s1_block, s1_origin = downs[1]
                    s2_block, s2_origin = downs[2]
                    result_s1[
                        s1_origin[0]:s1_origin[0]+s1_block.shape[0],
                        s1_origin[1]:s1_origin[1]+s1_block.shape[1],
                        s1_origin[2]:s1_origin[2]+s1_block.shape[2],
                    ] = s1_block
                    result_s2[
                        s2_origin[0]:s2_origin[0]+s2_block.shape[0],
                        s2_origin[1]:s2_origin[1]+s2_block.shape[1],
                        s2_origin[2]:s2_origin[2]+s2_block.shape[2],
                    ] = s2_block

        np.testing.assert_array_equal(result_s1, ref_s1)
        np.testing.assert_array_equal(result_s2, ref_s2)

    def test_super_chunk_matches_global_anisotropic(self):
        vol = self._build_synthetic_volume(shape=(16, 64, 64))
        per_lod = [(1, 1, 1), (1, 2, 2), (2, 4, 4)]
        ref_s1, _ = downsample_labels_3d(vol, (1, 2, 2))
        ref_s2, _ = downsample_labels_3d(vol, (2, 4, 4))

        # Super-chunk size = out_chunk * max per-axis factor = (8*2, 8*4, 8*4) = (16, 32, 32)
        super_chunk = np.array([16, 32, 32])
        out_chunk = np.array([8, 8, 8])
        result_s1 = np.zeros((16, 32, 32), dtype=vol.dtype)
        result_s2 = np.zeros((8, 16, 16), dtype=vol.dtype)
        for z0 in range(0, 16, 16):
            for y0 in range(0, 64, 32):
                for x0 in range(0, 64, 32):
                    s0_block = vol[z0:z0+16, y0:y0+32, x0:x0+32]
                    downs = downsample_super_chunk(
                        s0_block, np.array([z0, y0, x0]),
                        per_lod, downsample_labels_3d, out_chunk,
                    )
                    s1b, s1o = downs[1]
                    s2b, s2o = downs[2]
                    result_s1[
                        s1o[0]:s1o[0]+s1b.shape[0],
                        s1o[1]:s1o[1]+s1b.shape[1],
                        s1o[2]:s1o[2]+s1b.shape[2],
                    ] = s1b
                    result_s2[
                        s2o[0]:s2o[0]+s2b.shape[0],
                        s2o[1]:s2o[1]+s2b.shape[1],
                        s2o[2]:s2o[2]+s2b.shape[2],
                    ] = s2b
        np.testing.assert_array_equal(result_s1, ref_s1)
        np.testing.assert_array_equal(result_s2, ref_s2)


class TestEndToEndPyramidBuild:
    """Drive ``build_missing_pyramid_levels`` end-to-end against a
    synthetic volume backed by an in-memory s0_reader. Verify the on-disk
    output matches a single-pass reference downsample."""

    def test_build_isotropic_pyramid(self, tmp_path):
        rng = np.random.default_rng(7)
        vol = rng.integers(low=0, high=4, size=(32, 32, 32), dtype=np.uint8)

        def s0_reader(origin, shape):
            z, y, x = origin.tolist()
            sz, sy, sx = shape.tolist()
            return vol[z:z+sz, y:y+sy, x:x+sx].copy()

        out_path = str(tmp_path / "pyramid.zarr")
        result = build_missing_pyramid_levels(
            s0_reader=s0_reader,
            s0_dataset_shape_voxels=np.array(vol.shape),
            s0_voxel_size_zyx=[8.0, 8.0, 8.0],
            s0_translation_zyx=[4.0, 4.0, 4.0],
            dtype=vol.dtype,
            num_lods=3,
            existing_factors=set(),
            output_zarr_path=out_path,
            downsample_func=downsample_labels_3d,
            out_chunk_shape_voxels=(4, 4, 4),
        )

        assert result == out_path
        # OME metadata file present
        assert os.path.exists(os.path.join(out_path, ".zattrs"))
        with open(os.path.join(out_path, ".zattrs")) as f:
            md = json.load(f)
        assert len(md["multiscales"][0]["datasets"]) == 3

        # s1 and s2 zarrs exist and match the global downsample
        import zarr
        s1 = zarr.open_array(os.path.join(out_path, "s1"), mode="r")
        s2 = zarr.open_array(os.path.join(out_path, "s2"), mode="r")
        ref_s1, _ = downsample_labels_3d(vol, (2, 2, 2))
        ref_s2, _ = downsample_labels_3d(vol, (4, 4, 4))
        np.testing.assert_array_equal(s1[:], ref_s1)
        np.testing.assert_array_equal(s2[:], ref_s2)

    def test_symlink_s0_when_local(self, tmp_path):
        # Set up a dummy "source" s0 directory
        s0_src = tmp_path / "source_s0"
        s0_src.mkdir()
        (s0_src / "marker.txt").write_text("hi")

        rng = np.random.default_rng(0)
        vol = rng.integers(low=0, high=2, size=(16, 16, 16), dtype=np.uint8)

        out_path = str(tmp_path / "pyramid.zarr")
        build_missing_pyramid_levels(
            s0_reader=lambda o, s: vol[o[0]:o[0]+s[0], o[1]:o[1]+s[1], o[2]:o[2]+s[2]].copy(),
            s0_dataset_shape_voxels=np.array(vol.shape),
            s0_voxel_size_zyx=[1.0, 1.0, 1.0],
            s0_translation_zyx=[0.5, 0.5, 0.5],
            dtype=vol.dtype,
            num_lods=2,
            existing_factors=set(),
            output_zarr_path=out_path,
            downsample_func=downsample_labels_3d,
            out_chunk_shape_voxels=(4, 4, 4),
            s0_source_path=str(s0_src),
        )

        s0_link = os.path.join(out_path, "s0")
        assert os.path.islink(s0_link)
        assert os.path.exists(os.path.join(s0_link, "marker.txt"))

    def test_no_symlink_when_source_is_zarr_v3(self, tmp_path):
        """This pyramid's own group metadata and its s1+ arrays are
        written in whatever ``zarr_format`` the caller asks for
        (default 2). A zarr v3 source symlinked in as s0 when the
        pyramid is v2 (the default) would be format-inconsistent —
        generic OME-zarr readers (e.g. neuroglancer) resolve the group
        by its declared format then fail to find matching metadata
        inside the differently-formatted s0 array. s0 must stay absent
        instead (see test_symlinks_when_zarr_format_matches_source for
        the case where the caller matches the format up front)."""
        s0_src = tmp_path / "source_s0"
        s0_src.mkdir()
        (s0_src / "zarr.json").write_text('{"zarr_format": 3, "node_type": "array"}')

        rng = np.random.default_rng(0)
        vol = rng.integers(low=0, high=2, size=(16, 16, 16), dtype=np.uint8)

        out_path = str(tmp_path / "pyramid.zarr")
        build_missing_pyramid_levels(
            s0_reader=lambda o, s: vol[o[0]:o[0]+s[0], o[1]:o[1]+s[1], o[2]:o[2]+s[2]].copy(),
            s0_dataset_shape_voxels=np.array(vol.shape),
            s0_voxel_size_zyx=[1.0, 1.0, 1.0],
            s0_translation_zyx=[0.5, 0.5, 0.5],
            dtype=vol.dtype,
            num_lods=2,
            existing_factors=set(),
            output_zarr_path=out_path,
            downsample_func=downsample_labels_3d,
            out_chunk_shape_voxels=(4, 4, 4),
            s0_source_path=str(s0_src),
            # zarr_format defaults to 2 — mismatched against the v3 source above.
        )

        s0_link = os.path.join(out_path, "s0")
        assert not os.path.exists(s0_link)
        assert not os.path.islink(s0_link)
        # s1 (a genuine v2 array this builder wrote) is unaffected.
        assert os.path.isdir(os.path.join(out_path, "s1"))

    def test_symlinks_when_zarr_format_matches_source(self, tmp_path):
        """When the caller matches zarr_format to the source's real
        format (zarr v3 here), s0 CAN be symlinked in safely, and s1/s2
        are themselves real, readable v3 arrays — the whole point of
        detecting the source format up front (Meshify does this via
        self._driver) instead of always hardcoding v2."""
        import tensorstore as ts

        rng = np.random.default_rng(0)
        vol = rng.integers(low=0, high=4, size=(16, 16, 16), dtype=np.uint8)
        s0_src = tmp_path / "source_s0"
        ts.open({
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(s0_src)},
            "metadata": {
                "shape": list(vol.shape),
                "data_type": "uint8",
                "chunk_grid": {"name": "regular",
                               "configuration": {"chunk_shape": [4, 4, 4]}},
            },
            "create": True, "delete_existing": True,
        }).result().write(vol).result()

        out_path = str(tmp_path / "pyramid.zarr")
        build_missing_pyramid_levels(
            s0_reader=lambda o, s: vol[o[0]:o[0]+s[0], o[1]:o[1]+s[1], o[2]:o[2]+s[2]].copy(),
            s0_dataset_shape_voxels=np.array(vol.shape),
            s0_voxel_size_zyx=[1.0, 1.0, 1.0],
            s0_translation_zyx=[0.5, 0.5, 0.5],
            dtype=vol.dtype,
            num_lods=3,
            existing_factors=set(),
            output_zarr_path=out_path,
            downsample_func=downsample_labels_3d,
            out_chunk_shape_voxels=(4, 4, 4),
            s0_source_path=str(s0_src),
            zarr_format=3,
        )

        s0_link = os.path.join(out_path, "s0")
        assert os.path.islink(s0_link)
        assert os.path.isfile(os.path.join(s0_link, "zarr.json"))
        # Group metadata is zarr.json (v3), not .zattrs/.zgroup (v2).
        assert os.path.isfile(os.path.join(out_path, "zarr.json"))
        assert not os.path.exists(os.path.join(out_path, ".zattrs"))

        s1 = ts.open({
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": os.path.join(out_path, "s1")},
            "open": True,
        }).result().read().result()
        ref_s1, _ = downsample_labels_3d(vol, (2, 2, 2))
        np.testing.assert_array_equal(s1, ref_s1)

    def test_existing_factor_skipped(self, tmp_path):
        """If a factor is already present (existing_factors), it shouldn't
        be re-built."""
        rng = np.random.default_rng(0)
        vol = rng.integers(low=0, high=4, size=(16, 16, 16), dtype=np.uint8)
        out_path = str(tmp_path / "pyramid.zarr")
        build_missing_pyramid_levels(
            s0_reader=lambda o, s: vol[o[0]:o[0]+s[0], o[1]:o[1]+s[1], o[2]:o[2]+s[2]].copy(),
            s0_dataset_shape_voxels=np.array(vol.shape),
            s0_voxel_size_zyx=[1.0, 1.0, 1.0],
            s0_translation_zyx=[0.5, 0.5, 0.5],
            dtype=vol.dtype,
            num_lods=3,
            # Pretend s1 already exists
            existing_factors={(2, 2, 2)},
            output_zarr_path=out_path,
            downsample_func=downsample_labels_3d,
            out_chunk_shape_voxels=(4, 4, 4),
        )
        # s1 should NOT have been written, but s2 should
        assert not os.path.isdir(os.path.join(out_path, "s1"))
        assert os.path.isdir(os.path.join(out_path, "s2"))


class TestCascadeDownsampling:
    """``cascade=True`` builds each missing level from the immediately
    preceding one instead of always downsampling straight from s0."""

    def test_cascade_matches_direct_for_associative_reducer(self, tmp_path):
        """np.any-based downsampling is associative — cascade (s1 then
        s2-from-s1) must match direct (s2-from-s0) bit-for-bit."""
        rng = np.random.default_rng(3)
        vol = rng.integers(low=0, high=2, size=(32, 32, 32), dtype=np.uint8)

        def reader(o, s):
            return vol[o[0]:o[0]+s[0], o[1]:o[1]+s[1], o[2]:o[2]+s[2]].copy()

        common = dict(
            s0_reader=reader,
            s0_dataset_shape_voxels=np.array(vol.shape),
            s0_voxel_size_zyx=[1.0, 1.0, 1.0],
            s0_translation_zyx=[0.5, 0.5, 0.5],
            dtype=vol.dtype,
            num_lods=3,
            existing_factors=set(),
            downsample_func=downsample_binary_3d_suppress_zero,
            out_chunk_shape_voxels=(4, 4, 4),
        )
        direct_path = str(tmp_path / "direct.zarr")
        cascade_path = str(tmp_path / "cascade.zarr")
        build_missing_pyramid_levels(output_zarr_path=direct_path, cascade=False, **common)
        build_missing_pyramid_levels(output_zarr_path=cascade_path, cascade=True, **common)

        import zarr
        for lvl in ("s1", "s2"):
            direct_arr = zarr.open_array(os.path.join(direct_path, lvl), mode="r")[:]
            cascade_arr = zarr.open_array(os.path.join(cascade_path, lvl), mode="r")[:]
            np.testing.assert_array_equal(direct_arr, cascade_arr)

    def test_cascade_approximates_mode_reducer(self, tmp_path):
        """Mode (majority-vote) downsampling is NOT associative — s1
        (a single step, nothing composed yet) still matches exactly, but
        s2 (mode-of-modes under cascade) is only a close approximation of
        the direct/global reference. Uses a block-structured label volume
        (contiguous 4-voxel regions) since that's representative of real
        segmentation data — per-voxel random labels disagree almost
        everywhere and wouldn't demonstrate the "small fraction of
        boundary voxels" caveat this test documents."""
        rng = np.random.default_rng(11)
        coarse = rng.integers(low=0, high=6, size=(8, 8, 8), dtype=np.uint8)
        vol = np.kron(coarse, np.ones((4, 4, 4), dtype=np.uint8))

        def reader(o, s):
            return vol[o[0]:o[0]+s[0], o[1]:o[1]+s[1], o[2]:o[2]+s[2]].copy()

        common = dict(
            s0_reader=reader,
            s0_dataset_shape_voxels=np.array(vol.shape),
            s0_voxel_size_zyx=[1.0, 1.0, 1.0],
            s0_translation_zyx=[0.5, 0.5, 0.5],
            dtype=vol.dtype,
            num_lods=3,
            existing_factors=set(),
            downsample_func=downsample_labels_3d,
            out_chunk_shape_voxels=(4, 4, 4),
        )
        direct_path = str(tmp_path / "direct.zarr")
        cascade_path = str(tmp_path / "cascade.zarr")
        build_missing_pyramid_levels(output_zarr_path=direct_path, cascade=False, **common)
        build_missing_pyramid_levels(output_zarr_path=cascade_path, cascade=True, **common)

        import zarr
        direct_s1 = zarr.open_array(os.path.join(direct_path, "s1"), mode="r")[:]
        cascade_s1 = zarr.open_array(os.path.join(cascade_path, "s1"), mode="r")[:]
        np.testing.assert_array_equal(direct_s1, cascade_s1)

        direct_s2 = zarr.open_array(os.path.join(direct_path, "s2"), mode="r")[:]
        cascade_s2 = zarr.open_array(os.path.join(cascade_path, "s2"), mode="r")[:]
        agreement = np.mean(direct_s2 == cascade_s2)
        assert agreement > 0.8, f"expected high agreement, got {agreement:.2%}"

    def test_cascade_resets_chain_at_existing_gap(self, tmp_path):
        """If s1 is 'existing' (skipped, so cascade has no reader for its
        contents), s2 must be built directly from s0 rather than
        incorrectly chaining through a non-existent s1 array."""
        rng = np.random.default_rng(5)
        vol = rng.integers(low=0, high=4, size=(16, 16, 16), dtype=np.uint8)
        out_path = str(tmp_path / "pyramid.zarr")
        build_missing_pyramid_levels(
            s0_reader=lambda o, s: vol[o[0]:o[0]+s[0], o[1]:o[1]+s[1], o[2]:o[2]+s[2]].copy(),
            s0_dataset_shape_voxels=np.array(vol.shape),
            s0_voxel_size_zyx=[1.0, 1.0, 1.0],
            s0_translation_zyx=[0.5, 0.5, 0.5],
            dtype=vol.dtype,
            num_lods=3,
            existing_factors={(2, 2, 2)},  # pretend s1 already exists
            output_zarr_path=out_path,
            downsample_func=downsample_labels_3d,
            out_chunk_shape_voxels=(4, 4, 4),
            cascade=True,
        )
        assert not os.path.isdir(os.path.join(out_path, "s1"))

        import zarr
        s2 = zarr.open_array(os.path.join(out_path, "s2"), mode="r")[:]
        ref_s2, _ = downsample_labels_3d(vol, (4, 4, 4))
        np.testing.assert_array_equal(s2, ref_s2)


class TestSymlinkSafeCleanup:
    """The pyramid's ``s0`` is a symlink to the user's input data.
    Cleanup (whether via shutil.rmtree or our explicit walk) must
    unlink the symlink — NEVER follow it into the input."""

    def test_safe_remove_pyramid_preserves_symlink_target(self, tmp_path):
        from mesh_n_bone.meshify.meshify import _safely_remove_pyramid

        # Build a fake "source" directory with critical data
        src = tmp_path / "important_source"
        src.mkdir()
        (src / "DO_NOT_DELETE.txt").write_text("user's input data")
        (src / "subdir").mkdir()
        (src / "subdir" / "deeper.txt").write_text("also critical")

        # Build a fake pyramid with s0 -> src symlink
        pyramid = tmp_path / "pyramid.zarr"
        pyramid.mkdir()
        (pyramid / ".zattrs").write_text("{}")
        os.symlink(str(src), str(pyramid / "s0"))
        (pyramid / "s1").mkdir()
        (pyramid / "s1" / "data.bin").write_text("downsampled")

        # Sanity: symlink works
        assert (pyramid / "s0" / "DO_NOT_DELETE.txt").exists()

        # Tear down via our helper
        _safely_remove_pyramid(str(pyramid))

        # Pyramid is gone
        assert not pyramid.exists()
        # Source SURVIVES with all contents
        assert src.exists()
        assert (src / "DO_NOT_DELETE.txt").read_text() == "user's input data"
        assert (src / "subdir" / "deeper.txt").read_text() == "also critical"

    def test_safe_remove_handles_many_chunk_files(self, tmp_path):
        """Cleanup must scale to many chunk-like files (zarr v2 flat layout)
        — each pyramid sk array is a single directory containing thousands
        of small files. _safely_remove_pyramid partitions them across the
        thread pool so chunk-file unlinks are parallel, not serialised."""
        from mesh_n_bone.meshify.meshify import _safely_remove_pyramid

        pyramid = tmp_path / "pyramid.zarr"
        pyramid.mkdir()
        (pyramid / ".zattrs").write_text("{}")
        # Make a "fake s1 array" with many flat chunk files (like zarr v2)
        s1 = pyramid / "s1"
        s1.mkdir()
        n_files = 500
        for i in range(n_files):
            (s1 / f"0.0.{i}").write_text(str(i))
        # And a "fake s2 array" with fewer files
        s2 = pyramid / "s2"
        s2.mkdir()
        for i in range(50):
            (s2 / f"0.0.{i}").write_text(str(i))

        _safely_remove_pyramid(str(pyramid), num_workers=8)
        # Everything gone
        assert not pyramid.exists()

    def test_safe_remove_refuses_to_delete_when_root_is_symlink(self, tmp_path):
        from mesh_n_bone.meshify.meshify import _safely_remove_pyramid

        src = tmp_path / "real_data"
        src.mkdir()
        (src / "data.txt").write_text("real")

        # Hostile setup: pyramid_path itself is a symlink
        pyramid_link = tmp_path / "pyramid.zarr"
        os.symlink(str(src), str(pyramid_link))

        # Our helper refuses to delete a symlinked root
        _safely_remove_pyramid(str(pyramid_link))

        # Real data still exists
        assert (src / "data.txt").read_text() == "real"
        # The symlink itself may or may not be removed, but the target survives.


class TestUnalignedRoiHalo:
    """When ROI is unaligned and alignment_mode='halo', the pyramid
    builder reads beyond the ROI to complete boundary cubes. Output
    covers the ROI rounded OUTWARD to factor boundaries — no data loss
    at unaligned ROI edges."""

    def test_halo_mode_reads_beyond_roi(self, tmp_path):
        """With ROI [3, 37) (size 34) and max factor 4, halo mode should
        round outward to [0, 40) and produce s1=20 voxels, s2=10 voxels.

        The downsampled values must match a global single-pass downsample
        of vol[0:40, 0:40, 0:40] (the halo-extended region) — proving the
        halo reads actually pulled the surrounding voxels."""
        rng = np.random.default_rng(0)
        vol = rng.integers(low=0, high=4, size=(40, 40, 40), dtype=np.uint8)

        # Reader tracks what voxel ranges it was asked for, so we can
        # assert the halo actually fetched outside the ROI.
        read_log = []
        def reader(o, s):
            read_log.append((tuple(o.tolist()), tuple(s.tolist())))
            return vol[o[0]:o[0]+s[0], o[1]:o[1]+s[1], o[2]:o[2]+s[2]].copy()

        out_path = str(tmp_path / "pyramid_halo.zarr")
        build_missing_pyramid_levels(
            s0_reader=reader,
            s0_dataset_shape_voxels=np.array(vol.shape),
            s0_voxel_size_zyx=[1.0, 1.0, 1.0],
            s0_translation_zyx=[0.5, 0.5, 0.5],
            dtype=vol.dtype,
            num_lods=3,                       # max factor 4
            existing_factors=set(),
            output_zarr_path=out_path,
            downsample_func=downsample_labels_3d,
            out_chunk_shape_voxels=(4, 4, 4),
            roi_origin_voxels=np.array([3, 3, 3]),
            roi_shape_voxels=np.array([34, 34, 34]),
            alignment_mode="halo",
        )

        # Output region: ROI rounded OUTWARD to factor 4 → [0:40, 0:40, 0:40].
        # Downsampled by 2 → 20³; by 4 → 10³.
        import zarr
        s1 = zarr.open_array(os.path.join(out_path, "s1"), mode="r")
        s2 = zarr.open_array(os.path.join(out_path, "s2"), mode="r")
        assert s1.shape == (20, 20, 20)
        assert s2.shape == (10, 10, 10)

        # Match global single-pass downsample of the halo-extended region
        ref_s1, _ = downsample_labels_3d(vol[0:40, 0:40, 0:40], (2, 2, 2))
        ref_s2, _ = downsample_labels_3d(vol[0:40, 0:40, 0:40], (4, 4, 4))
        np.testing.assert_array_equal(s1[:], ref_s1)
        np.testing.assert_array_equal(s2[:], ref_s2)

        # The reader log should show reads starting at voxel 0 — outside
        # the original ROI which starts at voxel 3. If snap mode had been
        # used, reads would never go below voxel 4.
        min_read_origin = min(o for o, _ in read_log)
        assert min_read_origin == (0, 0, 0), (
            f"halo mode should read from voxel 0 (outside original ROI); "
            f"min read origin was {min_read_origin}"
        )

    def test_halo_clipped_at_dataset_bounds(self, tmp_path):
        """When the halo would extend past the dataset, the read is clipped
        and zero-padding (or partial cubes) take over. The output should
        still complete cleanly — no exceptions, output array fully written."""
        # 18-voxel dataset, ROI [3, 18) (size 15), max factor 4 → halo
        # rounds out to [0, 20). Dataset bounds clip the read at 18.
        rng = np.random.default_rng(1)
        vol = rng.integers(low=0, high=4, size=(18, 18, 18), dtype=np.uint8)
        out_path = str(tmp_path / "pyramid_halo_clip.zarr")
        build_missing_pyramid_levels(
            s0_reader=lambda o, s: vol[o[0]:o[0]+s[0], o[1]:o[1]+s[1], o[2]:o[2]+s[2]].copy(),
            s0_dataset_shape_voxels=np.array(vol.shape),
            s0_voxel_size_zyx=[1.0, 1.0, 1.0],
            s0_translation_zyx=[0.5, 0.5, 0.5],
            dtype=vol.dtype,
            num_lods=3,
            existing_factors=set(),
            output_zarr_path=out_path,
            downsample_func=downsample_labels_3d,
            out_chunk_shape_voxels=(4, 4, 4),
            roi_origin_voxels=np.array([3, 3, 3]),
            roi_shape_voxels=np.array([15, 15, 15]),
            alignment_mode="halo",
        )
        # Output region: [0, 20) rounded outward, but dataset only has 18.
        # Builder must handle this without crashing.
        import zarr
        s1 = zarr.open_array(os.path.join(out_path, "s1"), mode="r")
        # s1 covers output region 20 voxels / 2 = 10 voxels
        assert s1.shape == (10, 10, 10)


class TestUnalignedRoiSnap:
    """When the ROI is unaligned, snap mode drops up to max_factor-1
    voxels per edge; downsampled output covers the snapped region only."""

    def test_snap_mode_unaligned_roi(self, tmp_path):
        rng = np.random.default_rng(0)
        vol = rng.integers(low=0, high=4, size=(40, 40, 40), dtype=np.uint8)

        def reader(o, s):
            return vol[o[0]:o[0]+s[0], o[1]:o[1]+s[1], o[2]:o[2]+s[2]].copy()

        out_path = str(tmp_path / "pyramid.zarr")
        build_missing_pyramid_levels(
            s0_reader=reader,
            s0_dataset_shape_voxels=np.array(vol.shape),
            s0_voxel_size_zyx=[1.0, 1.0, 1.0],
            s0_translation_zyx=[0.5, 0.5, 0.5],
            dtype=vol.dtype,
            num_lods=3,                          # max factor 4
            existing_factors=set(),
            output_zarr_path=out_path,
            downsample_func=downsample_labels_3d,
            out_chunk_shape_voxels=(4, 4, 4),
            roi_origin_voxels=np.array([3, 3, 3]),
            roi_shape_voxels=np.array([34, 34, 34]),
            alignment_mode="snap",
        )
        # Snap: origin -> 4, end -> 36 -> shape 32
        # s1: 16x16x16 from vol[4:36, 4:36, 4:36] downsampled by 2
        import zarr
        s1 = zarr.open_array(os.path.join(out_path, "s1"), mode="r")
        assert s1.shape == (16, 16, 16)
        # Compare to global downsample of the snapped region
        ref_s1, _ = downsample_labels_3d(vol[4:36, 4:36, 4:36], (2, 2, 2))
        np.testing.assert_array_equal(s1[:], ref_s1)


class TestExistingPyramidScalesBothFormats:
    """``_existing_pyramid_scales`` (the "reuse a prior run's pyramid"
    shortcut) must parse group metadata from EITHER a zarr v2 pyramid
    (.zattrs) or a zarr v3 one (zarr.json, attributes nested under
    "attributes") — otherwise a v3-formatted pyramid (built once the
    source's own format is matched) would silently never be recognized
    for reuse on a second run, always rebuilding from scratch."""

    def _build(self, tmp_path, zarr_format):
        from mesh_n_bone.meshify.meshify import _existing_pyramid_scales

        rng = np.random.default_rng(1)
        vol = rng.integers(low=0, high=4, size=(16, 16, 16), dtype=np.uint8)
        out_path = str(tmp_path / f"pyramid_v{zarr_format}.zarr")
        build_missing_pyramid_levels(
            s0_reader=lambda o, s: vol[o[0]:o[0]+s[0], o[1]:o[1]+s[1], o[2]:o[2]+s[2]].copy(),
            s0_dataset_shape_voxels=np.array(vol.shape),
            s0_voxel_size_zyx=[1.0, 1.0, 1.0],
            s0_translation_zyx=[0.5, 0.5, 0.5],
            dtype=vol.dtype,
            num_lods=3,
            existing_factors=set(),
            output_zarr_path=out_path,
            downsample_func=downsample_labels_3d,
            out_chunk_shape_voxels=(4, 4, 4),
            zarr_format=zarr_format,
        )
        return out_path, _existing_pyramid_scales(out_path)

    def test_parses_zarr_v2_pyramid(self, tmp_path):
        out_path, result = self._build(tmp_path, zarr_format=2)
        assert os.path.isfile(os.path.join(out_path, ".zattrs"))
        assert result == {1: 2, 2: 4}

    def test_parses_zarr_v3_pyramid(self, tmp_path):
        out_path, result = self._build(tmp_path, zarr_format=3)
        assert os.path.isfile(os.path.join(out_path, "zarr.json"))
        assert result == {1: 2, 2: 4}
