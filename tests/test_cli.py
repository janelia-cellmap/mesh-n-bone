"""Tests for the CLI module."""

import subprocess
import sys
import pytest


class TestCLI:
    def test_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "mesh_n_bone.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "mesh-n-bone" in result.stdout
        assert "meshify" in result.stdout
        assert "multires" in result.stdout
        assert "skeletonize" in result.stdout
        assert "analyze" in result.stdout

    def test_no_args_shows_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "mesh_n_bone.cli"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1

    def test_skeletonize_single_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "mesh_n_bone.cli", "skeletonize-single", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "input_file" in result.stdout
        assert "output_file" in result.stdout
        assert "--subdivisions" in result.stdout
        assert "--neuroglancer" in result.stdout
