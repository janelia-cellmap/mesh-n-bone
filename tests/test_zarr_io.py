"""Tests for opening a segmentation volume through worldarray."""

import functools
import http.server
import os
import threading

import numpy as np
import pytest
import zarr

from mesh_n_bone.util.zarr_io import open_dataset


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


@pytest.fixture
def _http_directory_server(tmp_output_dir):
    """Serve *tmp_output_dir* over localhost HTTP for TensorStore tests."""

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

    handler = functools.partial(QuietHandler, directory=tmp_output_dir)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


class TestOpenDatasetOmeFallback:
    """``open_dataset`` must pick up voxel_size/offset from either layout."""

    @pytest.mark.parametrize("ome_version", ["0.4", "0.5"])
    def test_open_dataset_picks_up_ome(self, _ome_zarr_factory, ome_version):
        path = _ome_zarr_factory(ome_version)
        ds = open_dataset(path)
        assert tuple(ds.voxel_size) == (8, 16, 32)
        # offset propagates into the ROI begin in physical units.
        assert tuple(ds.roi.begin) == (10, 20, 30)

    @pytest.mark.parametrize("ome_version", ["0.4", "0.5"])
    def test_true_voxel_size_picks_up_ome(self, _ome_zarr_factory, ome_version):
        # Floats must survive the parent-group lookup.
        path = _ome_zarr_factory(ome_version)
        ds = open_dataset(path)
        assert tuple(ds.true_voxel_size) == (8.0, 16.0, 32.0)

    @pytest.mark.parametrize("ome_version", ["0.4", "0.5"])
    def test_true_offset_preserves_fractional_translation(
        self, tmp_output_dir, ome_version,
    ):
        # OME translation with sub-unit precision must survive the
        # parent-group lookup; the rounded
        # ds.roi.offset would otherwise drop the fraction.
        zarr_path = os.path.join(tmp_output_dir, f"frac_trans_{ome_version}.zarr")
        root = zarr.open_group(zarr_path, mode="w", zarr_format=2)
        ds_path = "s0"
        root.create_array(
            ds_path,
            data=np.zeros((4, 4, 4), dtype=np.uint32),
            chunks=(4, 4, 4),
        )
        ms = _multiscales_block([8.0, 8.0, 8.0], [10.5, 20.25, 30.125])
        if ome_version == "0.4":
            root.attrs["multiscales"] = ms
        else:
            root.attrs["ome"] = {"multiscales": ms, "version": "0.5"}

        ds = open_dataset(os.path.join(zarr_path, ds_path))
        # ds.roi.offset is the rounded Coordinate (integer truncation/round).
        # The raw accessor must keep the floats verbatim.
        assert tuple(ds.true_offset) == (10.5, 20.25, 30.125)

    def test_open_http_zarr_without_extension(
        self, tmp_output_dir, _http_directory_server,
    ):
        zarr_path = os.path.join(tmp_output_dir, "crop04_AIPv2")
        root = zarr.open_group(zarr_path, mode="w")
        root.create_array(
            "seg/s0",
            data=np.zeros((4, 4, 4), dtype=np.uint32),
            chunks=(4, 4, 4),
        )
        root["seg"].attrs["multiscales"] = _multiscales_block(
            [8, 16, 32], [10, 20, 30],
        )

        url = f"{_http_directory_server}/crop04_AIPv2/seg/s0"
        ds = open_dataset(url)

        assert ds.shape == (4, 4, 4)
        assert tuple(ds.voxel_size) == (8, 16, 32)
        assert tuple(ds.roi.begin) == (10, 20, 30)

    def test_open_http_ome_group_without_extension_selects_first_dataset(
        self, tmp_output_dir, _http_directory_server,
    ):
        zarr_path = os.path.join(tmp_output_dir, "crop04_group")
        root = zarr.open_group(zarr_path, mode="w")
        root.create_array(
            "seg/s0",
            data=np.zeros((4, 4, 4), dtype=np.uint32),
            chunks=(4, 4, 4),
        )
        ms = _multiscales_block([8.5, 16.5, 32.5], [10, 20, 30])
        ms[0]["datasets"][0]["path"] = "seg/s0"
        root.attrs["multiscales"] = ms

        url = f"{_http_directory_server}/crop04_group"
        ds = open_dataset(url)

        assert ds.shape == (4, 4, 4)
        assert ds._dataset_path == f"{url}/seg/s0"
        assert tuple(ds.true_voxel_size) == (8.5, 16.5, 32.5)


def _precomputed_scales(path, scales, num_channels=1, data=None):
    """Write a precomputed volume with one scale per ``(key, resolution,
    size)``; *data* (x, y, z, c) fills the first scale."""
    import tensorstore as ts

    for i, (key, resolution, size) in enumerate(scales):
        spec = {
            "driver": "neuroglancer_precomputed",
            "kvstore": {"driver": "file", "path": path},
            "scale_metadata": {"resolution": resolution, "size": size,
                               "chunk_size": size, "encoding": "raw", "key": key},
        }
        if i == 0:
            spec["multiscale_metadata"] = {"type": "segmentation", "data_type": "uint32",
                                           "num_channels": num_channels}
        store = ts.open(spec, create=True).result()
        if i == 0 and data is not None:
            store.write(data).result()


class TestOpenDatasetLevels:
    def test_precomputed_scale_key_opens_that_scale(self, tmp_output_dir):
        path = os.path.join(tmp_output_dir, "pc")
        _precomputed_scales(path, [("4nm", [4, 4, 4], [8, 8, 8]),
                                   ("8nm", [8, 8, 8], [4, 4, 4])])

        ds = open_dataset(f"{path}/8nm")

        assert tuple(ds.true_voxel_size) == (8, 8, 8)
        assert ds._dataset_path.endswith("8nm")
        # The volume root still opens at its first scale.
        assert tuple(open_dataset(path).true_voxel_size) == (4, 4, 4)

    def test_multichannel_precomputed_reads_channel_zero(self, tmp_output_dir):
        from funlib.geometry import Roi
        from mesh_n_bone.util.image_data_interface import open_volume, read_block

        path = os.path.join(tmp_output_dir, "pc2")
        data = np.zeros((8, 8, 8, 2), dtype=np.uint32)
        data[..., 0] = 1
        data[..., 1] = 2
        _precomputed_scales(path, [("s0", [4, 4, 4], [8, 8, 8])],
                            num_channels=2, data=data)

        ds = open_dataset(path)
        block = read_block(
            open_volume(path), Roi(ds.roi.offset, (16, 16, 16)),
            ds.voxel_size, ds.roi.offset,
        )

        assert block.shape == (4, 4, 4)
        assert np.all(block == 1)
        assert len(ds.chunk_shape) == 3
