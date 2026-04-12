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

zz, yy, xx = np.mgrid[0:256, 0:256, 0:256]

# Label 1: sphere centered at (128, 64, 64), radius 50
dist_sphere = np.sqrt((zz - 128) ** 2 + (yy - 64) ** 2 + (xx - 64) ** 2)
vol[dist_sphere <= 50] = 1

# Label 2: torus centered at (128, 192, 192), ring in the y-z plane
# major radius R=40 (distance from center to ring), minor radius r=20 (tube)
dist_to_ring = (np.sqrt((yy - 192) ** 2 + (zz - 128) ** 2) - 40) ** 2 + (xx - 192) ** 2
vol[dist_to_ring <= 20 ** 2] = 2

root = zarr.open_group(zarr_path, mode="w")
seg_group = root.require_group("seg")

# OME-NGFF multiscale metadata (z, y, x axes at 1 nm resolution)
seg_group.attrs["multiscales"] = [
    {
        "version": "0.5",
        "axes": [
            {"name": "z", "type": "space", "unit": "nanometer"},
            {"name": "y", "type": "space", "unit": "nanometer"},
            {"name": "x", "type": "space", "unit": "nanometer"},
        ],
        "datasets": [
            {
                "path": "s0",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 1.0, 1.0]}
                ],
            }
        ],
    }
]

arr = seg_group.create_array("s0", data=vol, chunks=(64, 64, 64))

print(f"Created example volume at {zarr_path}/seg/s0")
print(f"  Shape: {vol.shape}, dtype: {vol.dtype}, Labels: {np.unique(vol[vol > 0]).tolist()}")
