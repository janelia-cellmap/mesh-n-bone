"""Create an example zarr segmentation volume for the meshify demo.

Generates a 256x256x256 uint8 volume with two labeled objects:
  - Label 1: a sphere (radius 50 voxels)
  - Label 2: a torus (R=40, r=20)

Output: examples/data/example.zarr/seg/s0 (zarr v3, OME-NGFF multiscales)
"""

import json
import os
import shutil

import numpy as np
import tensorstore as ts

script_dir = os.path.dirname(os.path.abspath(__file__))
zarr_path = os.path.join(script_dir, "data", "example.zarr")
seg_path = os.path.join(zarr_path, "seg")
array_path = os.path.join(seg_path, "s0")

if os.path.exists(zarr_path):
    shutil.rmtree(zarr_path)

vol = np.zeros((256, 256, 256), dtype=np.uint8)
zz, yy, xx = np.mgrid[0:256, 0:256, 0:256]

dist_sphere = np.sqrt((zz - 128) ** 2 + (yy - 64) ** 2 + (xx - 64) ** 2)
vol[dist_sphere <= 50] = 1

dist_to_ring = (np.sqrt((yy - 192) ** 2 + (zz - 128) ** 2) - 40) ** 2 + (xx - 192) ** 2
vol[dist_to_ring <= 20 ** 2] = 2

multiscales = [
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

os.makedirs(seg_path, exist_ok=True)
with open(os.path.join(zarr_path, "zarr.json"), "w") as f:
    json.dump({"zarr_format": 3, "node_type": "group", "attributes": {}}, f)
with open(os.path.join(seg_path, "zarr.json"), "w") as f:
    json.dump(
        {"zarr_format": 3, "node_type": "group", "attributes": {"multiscales": multiscales}},
        f,
    )

arr = ts.open(
    {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": os.path.abspath(array_path)},
        "metadata": {
            "shape": list(vol.shape),
            "data_type": "uint8",
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": [64, 64, 64]},
            },
        },
        "create": True,
        "delete_existing": True,
    }
).result()
arr.write(vol).result()

print(f"Created example volume at {array_path}")
print(f"  Shape: {vol.shape}, dtype: {vol.dtype}, Labels: {np.unique(vol[vol > 0]).tolist()}")
