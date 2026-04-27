"""Unit tests for OME-NGFF metadata helpers in mesh_n_bone.util.zarr_io."""

import logging
import os

import numpy as np
import pytest
import zarr

from mesh_n_bone.util.zarr_io import (
    ArrayMetadata,
    _extract_ome_scale_translation,
    _get_multiscales,
    _read_funlib_voxel_offset,
    _read_transforms,
    _read_voxel_size_offset,
    open_dataset,
    read_raw_voxel_size,
)


def _multiscales_block(scale, translation):
    return [{
        "axes": [
            {"name": "z", "type": "space", "unit": "nanometer"},
            {"name": "y", "type": "space", "unit": "nanometer"},
            {"name": "x", "type": "space", "unit": "nanometer"},
        ],
        "datasets": [{
            "coordinateTransformations": [
                {"scale": list(scale), "type": "scale"},
                {"translation": list(translation), "type": "translation"},
            ],
            "path": "s0",
        }],
        "version": "test",
    }]


class TestGetMultiscales:
    """``_get_multiscales`` must accept v0.4 and v0.5 layouts."""

    def test_v04_top_level(self):
        attrs = {"multiscales": _multiscales_block([1, 2, 3], [0, 0, 0])}
        assert _get_multiscales(attrs) is attrs["multiscales"]

    def test_v05_ome_namespaced(self):
        ms = _multiscales_block([1, 2, 3], [0, 0, 0])
        attrs = {"ome": {"multiscales": ms, "version": "0.5"}}
        assert _get_multiscales(attrs) is ms

    def test_missing_returns_none(self):
        assert _get_multiscales({}) is None
        assert _get_multiscales({"ome": {}}) is None
        assert _get_multiscales(None) is None

    def test_v04_takes_precedence(self):
        # If both layouts are present, top-level wins (it's the legacy path
        # and matches existing cellmap datasets).
        v04 = _multiscales_block([4, 4, 4], [0, 0, 0])
        v05 = _multiscales_block([8, 8, 8], [0, 0, 0])
        attrs = {"multiscales": v04, "ome": {"multiscales": v05}}
        assert _get_multiscales(attrs) is v04


class TestExtractOmeScaleTranslation:
    """Reads scale and translation in either OME layout, preserving floats."""

    def test_v04(self):
        attrs = {"multiscales": _multiscales_block([8.0, 8.0, 8.0], [100, 100, 100])}
        scale, trans = _extract_ome_scale_translation(attrs)
        assert scale == (8.0, 8.0, 8.0)
        assert trans == (100.0, 100.0, 100.0)

    def test_v05(self):
        ms = _multiscales_block([32.0, 32.0, 32.0], [12.0, 12.0, 12.0])
        attrs = {"ome": {"multiscales": ms, "version": "0.5"}}
        scale, trans = _extract_ome_scale_translation(attrs)
        assert scale == (32.0, 32.0, 32.0)
        assert trans == (12.0, 12.0, 12.0)

    def test_non_integer_voxel_size_preserved(self):
        # Sub-nanometer / non-integer scales must round-trip as floats so
        # callers like read_raw_voxel_size don't lose precision.
        attrs = {"multiscales": _multiscales_block([8.5, 8.5, 8.5], [0, 0, 0])}
        scale, _ = _extract_ome_scale_translation(attrs)
        assert scale == (8.5, 8.5, 8.5)

    def test_missing_translation(self):
        ms = [{
            "axes": [],
            "datasets": [{
                "coordinateTransformations": [
                    {"scale": [4, 4, 4], "type": "scale"},
                ],
                "path": "s0",
            }],
        }]
        scale, trans = _extract_ome_scale_translation({"multiscales": ms})
        assert scale == (4.0, 4.0, 4.0)
        assert trans is None

    def test_no_multiscales(self):
        scale, trans = _extract_ome_scale_translation({})
        assert scale is None
        assert trans is None


@pytest.fixture
def _ome_zarr_factory(tmp_output_dir):
    """Build a tiny zarr v3 group with multiscales metadata in either layout."""

    def _make(ome_version, scale=(8, 16, 32), translation=(10, 20, 30)):
        zarr_path = os.path.join(tmp_output_dir, f"ome_{ome_version}.zarr")
        root = zarr.open_group(zarr_path, mode="w")
        root.create_array(
            "seg/s0",
            data=np.zeros((4, 4, 4), dtype=np.uint32),
            chunks=(4, 4, 4),
        )
        ms = _multiscales_block(scale, translation)
        if ome_version == "0.5":
            root["seg"].attrs["ome"] = {"multiscales": ms, "version": "0.5"}
        else:
            root["seg"].attrs["multiscales"] = ms
        return f"{zarr_path}/seg/s0"

    return _make


class TestOpenDatasetOmeFallback:
    """``open_dataset`` must pick up voxel_size/offset from either layout."""

    @pytest.mark.parametrize("ome_version", ["0.4", "0.5"])
    def test_open_dataset_picks_up_ome(self, _ome_zarr_factory, ome_version):
        path = _ome_zarr_factory(ome_version)
        ds = open_dataset(*os.path.split(path))
        assert tuple(ds.voxel_size) == (8, 16, 32)
        # offset propagates into the ROI begin in physical units.
        assert tuple(ds.roi.begin) == (10, 20, 30)

    @pytest.mark.parametrize("ome_version", ["0.4", "0.5"])
    def test_read_raw_voxel_size_picks_up_ome(self, _ome_zarr_factory, ome_version):
        # Floats must survive the parent-traversal path used by read_raw_voxel_size.
        path = _ome_zarr_factory(ome_version)
        ds = open_dataset(*os.path.split(path))
        raw = read_raw_voxel_size(ds)
        assert raw == (8.0, 16.0, 32.0)


def _empty_meta(attrs=None):
    return ArrayMetadata(
        shape=(4, 4, 4), dtype=np.uint32, chunks=(4, 4, 4), attrs=attrs or {},
    )


class TestReadVoxelSizeOffsetParentFallback:
    """The ``parent_attrs`` parameter feeds the OME multiscales fallback."""

    def test_array_attrs_take_precedence_over_parent(self):
        data = _empty_meta({"voxel_size": [2, 2, 2], "offset": [1, 1, 1]})
        parent_attrs = {"multiscales": _multiscales_block([8, 8, 8], [9, 9, 9])}
        vs, off = _read_voxel_size_offset(data, parent_attrs=parent_attrs)
        assert tuple(vs) == (2, 2, 2)
        assert tuple(off) == (1, 1, 1)

    def test_parent_fallback_when_array_empty(self):
        data = _empty_meta()
        parent_attrs = {"ome": {"multiscales": _multiscales_block([8, 8, 8], [9, 9, 9])}}
        vs, off = _read_voxel_size_offset(data, parent_attrs=parent_attrs)
        assert tuple(vs) == (8, 8, 8)
        assert tuple(off) == (9, 9, 9)

    def test_parent_funlib_fallback(self):
        # #7: some N5 multiscales setups put resolution on the parent
        # group instead of each scale array.
        data = _empty_meta()
        parent_attrs = {"voxel_size": [16, 16, 16], "offset": [4, 4, 4]}
        vs, off = _read_voxel_size_offset(data, parent_attrs=parent_attrs)
        assert tuple(vs) == (16, 16, 16)
        assert tuple(off) == (4, 4, 4)

    def test_parent_funlib_pixelResolution(self):
        data = _empty_meta()
        parent_attrs = {
            "pixelResolution": {"dimensions": [3, 2, 1], "unit": "nm"},
        }
        vs, _ = _read_voxel_size_offset(data, parent_attrs=parent_attrs)
        # Reversed XYZ → ZYX
        assert tuple(vs) == (1, 2, 3)

    def test_no_parent_attrs_falls_back_to_unit(self):
        data = _empty_meta()
        vs, off = _read_voxel_size_offset(data, parent_attrs=None)
        assert tuple(vs) == (1, 1, 1)
        assert tuple(off) == (0, 0, 0)

    def test_non_integer_voxel_size_warns(self, caplog):
        data = _empty_meta()
        parent_attrs = {"multiscales": _multiscales_block([8.5, 8.5, 8.5], [0, 0, 0])}
        with caplog.at_level(logging.WARNING, logger="mesh_n_bone.util.zarr_io"):
            vs, _ = _read_voxel_size_offset(data, parent_attrs=parent_attrs)
        assert tuple(vs) == (8, 8, 8)
        assert any("non-integer voxel_size" in r.message for r in caplog.records)


class TestPathMatching:
    """``_extract_ome_scale_translation`` matches by dataset path (#1)."""

    def test_matches_named_dataset(self):
        ms = [{
            "axes": [
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": [
                {
                    "path": "s0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [4, 4, 4]},
                    ],
                },
                {
                    "path": "s1",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [8, 8, 8]},
                    ],
                },
            ],
        }]
        scale, _ = _extract_ome_scale_translation({"multiscales": ms}, dataset_name="s1")
        assert scale == (8.0, 8.0, 8.0)

    def test_falls_back_to_first_when_path_unknown(self):
        ms = [{
            "datasets": [
                {
                    "path": "s0",
                    "coordinateTransformations": [{"type": "scale", "scale": [4, 4, 4]}],
                },
            ],
        }]
        scale, _ = _extract_ome_scale_translation({"multiscales": ms}, dataset_name="missing")
        assert scale == (4.0, 4.0, 4.0)


class TestTransformOrdering:
    """``_read_transforms`` composes in document order regardless of layout (#2)."""

    def test_scale_then_translation(self):
        # physical = scale * voxel + translation  (the conventional order).
        s, t = _read_transforms([
            {"type": "scale", "scale": [2, 2, 2]},
            {"type": "translation", "translation": [10, 10, 10]},
        ])
        assert s == [2.0, 2.0, 2.0]
        assert t == [10.0, 10.0, 10.0]

    def test_translation_then_scale(self):
        # If translation is applied first, it lives in voxel space and
        # must be scaled along with the rest:
        # physical = scale * (translation + voxel) = scale * voxel + scale * translation.
        s, t = _read_transforms([
            {"type": "translation", "translation": [10, 10, 10]},
            {"type": "scale", "scale": [2, 2, 2]},
        ])
        assert s == [2.0, 2.0, 2.0]
        assert t == [20.0, 20.0, 20.0]

    def test_identity_ignored(self):
        s, t = _read_transforms([
            {"type": "identity"},
            {"type": "scale", "scale": [3, 3, 3]},
        ])
        assert s == [3.0, 3.0, 3.0]
        assert t is None


class TestAxesPermutation:
    """OME-NGFF axes ordering and 5D filtering (#3)."""

    def test_xyz_axes_reordered_to_zyx(self):
        ms = [{
            "axes": [
                {"name": "x", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "z", "type": "space"},
            ],
            "datasets": [{
                "path": "s0",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [10, 20, 30]},  # x,y,z
                    {"type": "translation", "translation": [1, 2, 3]},
                ],
            }],
        }]
        scale, trans = _extract_ome_scale_translation({"multiscales": ms})
        # ZYX-ordered output
        assert scale == (30.0, 20.0, 10.0)
        assert trans == (3.0, 2.0, 1.0)

    def test_5d_axes_filtered_to_spatial_zyx(self):
        ms = [{
            "axes": [
                {"name": "t", "type": "time"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": [{
                "path": "s0",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1, 1, 4, 4, 4]},
                ],
            }],
        }]
        scale, _ = _extract_ome_scale_translation({"multiscales": ms})
        assert scale == (4.0, 4.0, 4.0)

    def test_no_axes_assumes_zyx(self):
        ms = [{
            "datasets": [{
                "path": "s0",
                "coordinateTransformations": [{"type": "scale", "scale": [1, 2, 3]}],
            }],
        }]
        scale, _ = _extract_ome_scale_translation({"multiscales": ms})
        assert scale == (1.0, 2.0, 3.0)


class TestRootLevelComposition:
    """Multiscales root-level coordinateTransformations compose with per-dataset (#4)."""

    def test_root_scale_composes(self):
        ms = [{
            "axes": [
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": [{
                "path": "s0",
                "coordinateTransformations": [{"type": "scale", "scale": [2, 2, 2]}],
            }],
            "coordinateTransformations": [
                {"type": "scale", "scale": [4, 4, 4]},
            ],
        }]
        scale, _ = _extract_ome_scale_translation({"multiscales": ms})
        assert scale == (8.0, 8.0, 8.0)

    def test_root_translation_after_dataset_scale(self):
        # physical = root_translation + root_scale * dataset_translation
        #            + (root_scale * dataset_scale) * voxel
        ms = [{
            "datasets": [{
                "path": "s0",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [2, 2, 2]},
                    {"type": "translation", "translation": [5, 5, 5]},
                ],
            }],
            "coordinateTransformations": [
                {"type": "translation", "translation": [100, 100, 100]},
            ],
        }]
        scale, trans = _extract_ome_scale_translation({"multiscales": ms})
        assert scale == (2.0, 2.0, 2.0)
        assert trans == (105.0, 105.0, 105.0)


class TestForwardCompatWarning:
    """Unrecognized ``ome`` block layout should warn (#6)."""

    def test_warns_on_ome_without_multiscales(self, caplog):
        attrs = {"ome": {"version": "0.6", "schema_url": "..."}}
        with caplog.at_level(logging.WARNING, logger="mesh_n_bone.util.zarr_io"):
            assert _get_multiscales(attrs) is None
        assert any(
            "ome" in r.message and "multiscales" in r.message for r in caplog.records
        )


class TestFunlibHelper:
    def test_returns_none_when_empty(self):
        assert _read_funlib_voxel_offset({}) == (None, None)
        assert _read_funlib_voxel_offset(None) == (None, None)

    def test_voxel_size_takes_precedence_over_pixelResolution(self):
        attrs = {
            "voxel_size": [4, 4, 4],
            "pixelResolution": {"dimensions": [99, 99, 99]},
        }
        vs, _ = _read_funlib_voxel_offset(attrs)
        assert vs == [4.0, 4.0, 4.0]
