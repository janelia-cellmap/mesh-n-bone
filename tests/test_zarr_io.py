"""Unit tests for OME-NGFF metadata helpers in mesh_n_bone.util.zarr_io."""

import os

import numpy as np
import pytest
import zarr

from mesh_n_bone.util.zarr_io import (
    _extract_ome_scale_translation,
    _get_multiscales,
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


class TestReadVoxelSizeOffsetParentFallback:
    """The ``parent_attrs`` parameter feeds the OME multiscales fallback."""

    def test_array_attrs_take_precedence_over_parent(self):
        # If the array carries explicit voxel_size, the OME parent fallback
        # must not override it.
        from mesh_n_bone.util.zarr_io import ArrayMetadata

        data = ArrayMetadata(
            shape=(4, 4, 4),
            dtype=np.uint32,
            chunks=(4, 4, 4),
            attrs={"voxel_size": [2, 2, 2], "offset": [1, 1, 1]},
        )
        parent_attrs = {"multiscales": _multiscales_block([8, 8, 8], [9, 9, 9])}
        vs, off = _read_voxel_size_offset(data, parent_attrs=parent_attrs)
        assert tuple(vs) == (2, 2, 2)
        assert tuple(off) == (1, 1, 1)

    def test_parent_fallback_when_array_empty(self):
        from mesh_n_bone.util.zarr_io import ArrayMetadata

        data = ArrayMetadata(
            shape=(4, 4, 4), dtype=np.uint32, chunks=(4, 4, 4), attrs={},
        )
        parent_attrs = {"ome": {"multiscales": _multiscales_block([8, 8, 8], [9, 9, 9])}}
        vs, off = _read_voxel_size_offset(data, parent_attrs=parent_attrs)
        assert tuple(vs) == (8, 8, 8)
        assert tuple(off) == (9, 9, 9)

    def test_no_parent_attrs_falls_back_to_unit(self):
        from mesh_n_bone.util.zarr_io import ArrayMetadata

        data = ArrayMetadata(
            shape=(4, 4, 4), dtype=np.uint32, chunks=(4, 4, 4), attrs={},
        )
        vs, off = _read_voxel_size_offset(data, parent_attrs=None)
        assert tuple(vs) == (1, 1, 1)
        assert tuple(off) == (0, 0, 0)
