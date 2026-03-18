"""Integration tests for downsampling methods on 3D volumes."""

import numpy as np
import pytest

from mesh_n_bone.meshify.downsample import (
    downsample_box,
    downsample_labels_3d,
    downsample_labels_3d_suppress_zero,
    downsample_binary_3d,
    downsample_binary_3d_suppress_zero,
    flat_mode,
    flat_mode_except_zero,
    flat_binary_mode,
    make_blockwise_reducer_3d,
)


class TestBlockwiseDownsampling:
    """Test blockwise downsampling on realistic 3D volumes."""

    def test_mode_downsampling_preserves_majority_label(self, labeled_volume_3d):
        """Mode downsampling should keep the most frequent label per block."""
        result, result_box = downsample_labels_3d(
            labeled_volume_3d, block_shape=np.array([2, 2, 2])
        )
        assert result.shape == (4, 4, 4)
        # Block [0:2, 0:2, 0:2] is all label 1 → should be 1
        assert result[0, 0, 0] == 1
        # Block [2:4, 2:4, 2:4] is all label 2 → should be 2
        assert result[2, 2, 2] == 2

    def test_mode_suppress_zero_preserves_thin_structures(self):
        """mode_suppress_zero should prefer nonzero labels over zero."""
        vol = np.zeros((4, 4, 4), dtype=np.uint32)
        # One voxel of label 5 surrounded by zeros
        vol[0, 0, 0] = 5
        result, _ = downsample_labels_3d_suppress_zero(
            vol, block_shape=np.array([4, 4, 4])
        )
        assert result.shape == (1, 1, 1)
        # Should pick 5 (the only nonzero) instead of 0
        assert result[0, 0, 0] == 5

    def test_mode_suppress_zero_all_zeros(self):
        """When all values are zero, suppress_zero should return 0."""
        vol = np.zeros((4, 4, 4), dtype=np.uint32)
        result, _ = downsample_labels_3d_suppress_zero(
            vol, block_shape=np.array([4, 4, 4])
        )
        assert result[0, 0, 0] == 0

    def test_mode_suppress_zero_multiple_nonzero(self):
        """With multiple nonzero labels, suppress_zero picks the mode among them."""
        vol = np.zeros((4, 4, 4), dtype=np.uint32)
        vol[0, 0, 0] = 3
        vol[0, 0, 1] = 3
        vol[0, 0, 2] = 7
        vol[0, 0, 3] = 0
        result, _ = downsample_labels_3d_suppress_zero(
            vol, block_shape=np.array([4, 4, 4])
        )
        # Label 3 appears twice, label 7 once → mode is 3
        assert result[0, 0, 0] == 3

    def test_binary_mode_majority_vote(self):
        """Binary mode should return 1 if majority of voxels are nonzero."""
        vol = np.zeros((4, 4, 4), dtype=np.uint32)
        # Fill more than half with nonzero
        vol[0:3, :, :] = 1  # 48 out of 64
        result, _ = downsample_binary_3d(vol, block_shape=np.array([4, 4, 4]))
        assert result[0, 0, 0] == 1

    def test_binary_mode_minority_zero(self):
        """Binary mode should return 0 if majority of voxels are zero."""
        vol = np.zeros((4, 4, 4), dtype=np.uint32)
        vol[0, 0, 0] = 1  # only 1 out of 64
        result, _ = downsample_binary_3d(vol, block_shape=np.array([4, 4, 4]))
        assert result[0, 0, 0] == 0

    def test_binary_suppress_zero_any_nonzero(self):
        """binary_suppress_zero uses np.any — any nonzero voxel means 1."""
        vol = np.zeros((4, 4, 4), dtype=np.uint32)
        vol[0, 0, 0] = 1  # single voxel
        result, _ = downsample_binary_3d_suppress_zero(
            vol, block_shape=np.array([4, 4, 4])
        )
        assert result[0, 0, 0] == 1

    def test_downsample_preserves_data_box(self):
        """Downsampling with a custom data_box should adjust coordinates correctly."""
        vol = np.ones((8, 8, 8), dtype=np.uint32) * 5
        data_box = np.array([[10, 10, 10], [18, 18, 18]])
        result, result_box = downsample_labels_3d(
            vol, block_shape=np.array([2, 2, 2]), data_box=data_box
        )
        assert result.shape == (4, 4, 4)
        np.testing.assert_array_equal(result_box[0], [5, 5, 5])
        np.testing.assert_array_equal(result_box[1], [9, 9, 9])

    def test_identity_downsample(self):
        """Block shape of (1,1,1) should return the original data."""
        vol = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint32)
        result, _ = downsample_labels_3d(vol, block_shape=np.array([1, 1, 1]))
        np.testing.assert_array_equal(result, vol)

    def test_anisotropic_downsample(self):
        """Anisotropic block shape should downsample differently per axis."""
        vol = np.ones((8, 8, 8), dtype=np.uint32) * 42
        result, _ = downsample_labels_3d(vol, block_shape=np.array([4, 2, 1]))
        assert result.shape == (2, 4, 8)
        assert np.all(result == 42)


class TestDownsampleBox:
    def test_basic(self):
        box = np.array([[0, 0, 0], [10, 10, 10]])
        result = downsample_box(box, np.array([2, 2, 2]))
        np.testing.assert_array_equal(result[0], [0, 0, 0])
        np.testing.assert_array_equal(result[1], [5, 5, 5])

    def test_non_divisible(self):
        """Non-divisible dimensions should round up."""
        box = np.array([[0, 0, 0], [7, 7, 7]])
        result = downsample_box(box, np.array([2, 2, 2]))
        np.testing.assert_array_equal(result[1], [4, 4, 4])  # ceil(7/2) = 4

    def test_offset_box(self):
        box = np.array([[4, 4, 4], [12, 12, 12]])
        result = downsample_box(box, np.array([4, 4, 4]))
        np.testing.assert_array_equal(result[0], [1, 1, 1])
        np.testing.assert_array_equal(result[1], [3, 3, 3])
