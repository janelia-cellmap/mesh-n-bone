"""Create an example zarr segmentation volume for the meshify demo.

Generates a 256x256x256 uint8 volume with three labeled objects:
  - Label 1: a sphere (radius 50 voxels)
  - Label 2: a torus (R=40, r=20)
  - Label 3: a ridged blob with high-frequency surface detail

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


def reset_directory(path):
    if not os.path.exists(path):
        return
    try:
        shutil.rmtree(path)
    except OSError:
        stale_path = f"{path}.stale-{os.getpid()}"
        os.rename(path, stale_path)
        shutil.rmtree(stale_path, ignore_errors=True)


if os.path.exists(zarr_path):
    reset_directory(zarr_path)

vol = np.zeros((256, 256, 256), dtype=np.uint8)


def object_box(center, radius):
    if np.isscalar(radius):
        radius = (radius, radius, radius)
    begin = [max(0, int(c - r)) for c, r in zip(center, radius)]
    end = [min(s, int(c + r + 1)) for c, r, s in zip(center, radius, vol.shape)]
    return tuple(slice(b, e) for b, e in zip(begin, end))


def box_coords(box):
    z, y, x = box
    return np.ogrid[z.start:z.stop, y.start:y.stop, x.start:x.stop]


# Objects are positioned so each one straddles the y=128 and x=128 center
# planes (z=128 was already crossed by all three). That way the neuroglancer
# cross-section panels at the volume center show pieces of each object.

sphere_box = object_box((128, 96, 96), 50)
zz, yy, xx = box_coords(sphere_box)
sphere = vol[sphere_box]
sphere[(zz - 128) ** 2 + (yy - 96) ** 2 + (xx - 96) ** 2 <= 50**2] = 1

torus_box = object_box((128, 160, 128), (60, 60, 20))
zz, yy, xx = box_coords(torus_box)
dist_to_ring = (
    np.sqrt((yy - 160) ** 2 + (zz - 128) ** 2) - 40
) ** 2 + (xx - 128) ** 2
torus = vol[torus_box]
torus[dist_to_ring <= 20**2] = 2

blob_box = object_box((128, 160, 96), 48)
zz, yy, xx = box_coords(blob_box)
dz = zz - 128
dy = yy - 160
dx = xx - 96
r = np.sqrt(dx**2 + dy**2 + dz**2)
azimuth = np.arctan2(dy, dx)
polar = np.arctan2(dz, np.sqrt(dx**2 + dy**2))
ridged_surface = (
    34
    + 6 * np.sin(7 * azimuth + 2 * np.sin(3 * polar))
    + 4 * np.sin(9 * polar)
    + 2 * np.sin(0.27 * dx + 0.17 * dy + 0.21 * dz)
)
blob = vol[blob_box]
blob[r <= ridged_surface] = 3

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
