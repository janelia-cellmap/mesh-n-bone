"""Create an example zarr segmentation volume for the meshify demo.

Generates a 256x256x256 uint8 volume with two labeled objects:
  - Label 1: a sphere (radius 80 voxels)
  - Label 2: a cube (80x80x80 voxels)

Output: examples/data/example.zarr/seg/s0
"""

import os
import numpy as np
import zarr

script_dir = os.path.dirname(os.path.abspath(__file__))
zarr_path = os.path.join(script_dir, "data", "example.zarr")

vol = np.zeros((256, 256, 256), dtype=np.uint8)

# Label 1: sphere centered at (100, 128, 128), radius 80
zz, yy, xx = np.mgrid[0:256, 0:256, 0:256]
dist = np.sqrt((zz - 100) ** 2 + (yy - 128) ** 2 + (xx - 128) ** 2)
vol[dist <= 80] = 1

# Label 2: cube
vol[160:240, 160:240, 160:240] = 2

root = zarr.open_group(zarr_path, mode="w")
arr = root.create_array("seg/s0", data=vol, chunks=(64, 64, 64))
arr.attrs["voxel_size"] = [1, 1, 1]
arr.attrs["offset"] = [0, 0, 0]
arr.attrs["axis_names"] = ["z", "y", "x"]

print(f"Created example volume at {zarr_path}/seg/s0")
print(f"  Shape: {vol.shape}, dtype: {vol.dtype}, Labels: {np.unique(vol[vol > 0]).tolist()}")
