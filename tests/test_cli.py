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
        assert "to-neuroglancer" in result.stdout
        assert "skeletonize" in result.stdout
        assert "analyze" in result.stdout

    def test_no_args_shows_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "mesh_n_bone.cli"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1

    def test_parse_ids_arg_comma_list(self):
        from mesh_n_bone.cli import _parse_ids_arg
        assert _parse_ids_arg("123,456,789") == [123, 456, 789]

    def test_parse_ids_arg_single_int(self):
        from mesh_n_bone.cli import _parse_ids_arg
        assert _parse_ids_arg("42") == [42]

    def test_parse_ids_arg_csv_path(self):
        from mesh_n_bone.cli import _parse_ids_arg
        # No commas, not all-digits → treated as a path string,
        # passed through to Meshify which knows how to read it.
        path = "/some/path/ids.csv"
        assert _parse_ids_arg(path) == path

    def test_meshify_help_shows_ids(self):
        result = subprocess.run(
            [sys.executable, "-m", "mesh_n_bone.cli", "meshify", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--ids" in result.stdout
        assert "segment IDs" in result.stdout or "segment ids" in result.stdout.lower()

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
