"""Unit tests for the neuroglancer precomputed input reader.

These tests cover URL parsing, kvstore-spec construction, and scale
resolution. They monkey-patch the ``info``-fetching helper so the tests
don't need network access.
"""

import json

import pytest

from mesh_n_bone.util import precomputed_io


SAMPLE_INFO = {
    "type": "segmentation",
    "data_type": "uint64",
    "num_channels": 1,
    "scales": [
        {
            "key": "8.0x8.0x8.0",
            "resolution": [8, 8, 8],
            "size": [34432, 39552, 41408],
            "voxel_offset": [0, 0, 0],
            "chunk_sizes": [[64, 64, 64]],
            "encoding": "compressed_segmentation",
        },
        {
            "key": "16.0x16.0x16.0",
            "resolution": [16, 16, 16],
            "size": [17216, 19776, 20704],
            "chunk_sizes": [[64, 64, 64]],
            "encoding": "compressed_segmentation",
        },
    ],
}


@pytest.fixture
def patched_fetch_info(monkeypatch):
    """Replace _fetch_info so calls to scale-key subpaths get a 404."""
    def fake_fetch(kvstore_spec, base_path):
        normalized = base_path.rstrip("/")
        # Pretend the info file only exists at the dataset root.
        if normalized.endswith("/segmentation") or normalized == "v1.0/segmentation":
            return SAMPLE_INFO
        raise FileNotFoundError(f"info not found under {base_path!r}")
    monkeypatch.setattr(precomputed_io, "_fetch_info", fake_fetch)


class TestIsPrecomputedPath:
    def test_recognized(self):
        assert precomputed_io.is_precomputed_path(
            "precomputed://gs://bucket/path"
        )

    def test_not_recognized(self):
        assert not precomputed_io.is_precomputed_path("/local/path/seg.zarr/s0")
        assert not precomputed_io.is_precomputed_path("https://example.com/data")
        assert not precomputed_io.is_precomputed_path(None)


class TestKvstoreForUrl:
    def test_gcs_url(self):
        kv, path = precomputed_io._kvstore_for_url(
            "gs://my-bucket/data/segmentation"
        )
        assert kv == {"driver": "gcs", "bucket": "my-bucket"}
        assert path == "data/segmentation"

    def test_s3_url(self, monkeypatch):
        # With no AWS env vars set, the S3 kvstore now defaults to anonymous
        # credentials to avoid the multi-second IMDS / credential-chain
        # timeout on public buckets like OpenOrganelle. See
        # ``_should_use_anonymous_s3`` in zarr_io.
        for k in ("AWS_ACCESS_KEY_ID", "AWS_PROFILE", "AWS_ENDPOINT_URL"):
            monkeypatch.delenv(k, raising=False)
        kv, path = precomputed_io._kvstore_for_url(
            "s3://my-bucket/path/to/seg"
        )
        assert kv == {
            "driver": "s3", "bucket": "my-bucket",
            "aws_credentials": {"type": "anonymous"},
        }
        assert path == "path/to/seg"

    def test_http_url(self):
        kv, path = precomputed_io._kvstore_for_url(
            "https://example.com/data/seg"
        )
        assert kv == {"driver": "http", "base_url": "https://example.com"}
        assert path == "data/seg"

    def test_file_scheme(self):
        kv, path = precomputed_io._kvstore_for_url("file:///mnt/data/seg")
        assert kv == {"driver": "file"}
        assert path == "/mnt/data/seg"

    def test_unsupported_scheme(self):
        with pytest.raises(ValueError, match="Unsupported"):
            precomputed_io._kvstore_for_url("ftp://example.com/data")


class TestParsePrecomputedPath:
    def test_default_scale_when_no_suffix(self, patched_fetch_info):
        url = "precomputed://gs://bucket/v1.0/segmentation"
        kvstore, base_path, info, scale_index = (
            precomputed_io.parse_precomputed_path(url)
        )
        assert kvstore["driver"] == "gcs"
        assert kvstore["bucket"] == "bucket"
        assert base_path == "v1.0/segmentation"
        assert info == SAMPLE_INFO
        assert scale_index == 0

    def test_scale_suffix_resolved(self, patched_fetch_info):
        url = "precomputed://gs://bucket/v1.0/segmentation/16.0x16.0x16.0"
        kvstore, base_path, info, scale_index = (
            precomputed_io.parse_precomputed_path(url)
        )
        assert base_path == "v1.0/segmentation"
        assert scale_index == 1

    def test_unknown_scale_key_errors(self, patched_fetch_info):
        url = "precomputed://gs://bucket/v1.0/segmentation/999.0x999.0x999.0"
        with pytest.raises(ValueError, match="scale key"):
            precomputed_io.parse_precomputed_path(url)

    def test_accepts_bare_url_without_prefix(self, patched_fetch_info):
        # The precomputed:// prefix is no longer required — bare URLs
        # work the same way (format will have been auto-detected upstream).
        url = "gs://bucket/v1.0/segmentation"
        kvstore, base_path, info, scale_index = (
            precomputed_io.parse_precomputed_path(url)
        )
        assert kvstore["driver"] == "gcs"
        assert base_path == "v1.0/segmentation"
        assert info == SAMPLE_INFO
        assert scale_index == 0


class TestPrecomputedArrayMetadata:
    def test_default_scale_zyx_orientation(self, patched_fetch_info):
        url = "precomputed://gs://bucket/v1.0/segmentation"
        meta = precomputed_io.precomputed_array_metadata(url)
        # XYZ in info → ZYX in meta
        assert meta["shape"] == (41408, 39552, 34432)
        assert meta["chunks"] == (64, 64, 64)
        assert meta["voxel_size"] == [8.0, 8.0, 8.0]
        assert meta["offset"] == [0, 0, 0]
        assert meta["dtype"] == "uint64"
        assert meta["scale_index"] == 0

    def test_specific_scale(self, patched_fetch_info):
        url = "precomputed://gs://bucket/v1.0/segmentation/16.0x16.0x16.0"
        meta = precomputed_io.precomputed_array_metadata(url)
        assert meta["shape"] == (20704, 19776, 17216)
        assert meta["voxel_size"] == [16.0, 16.0, 16.0]
        assert meta["scale_index"] == 1

    def test_voxel_offset_converts_to_physical_units(
        self, monkeypatch, patched_fetch_info
    ):
        # Override info to add a non-zero voxel_offset
        info_with_offset = json.loads(json.dumps(SAMPLE_INFO))
        info_with_offset["scales"][0]["voxel_offset"] = [10, 20, 30]
        monkeypatch.setattr(
            precomputed_io,
            "_fetch_info",
            lambda kv, p: info_with_offset,
        )
        url = "precomputed://gs://bucket/v1.0/segmentation"
        meta = precomputed_io.precomputed_array_metadata(url)
        # XYZ offset [10, 20, 30] * resolution [8, 8, 8] = [80, 160, 240]
        # then reversed to ZYX = [240, 160, 80]
        assert meta["offset"] == [240, 160, 80]
