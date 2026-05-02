# mesh-n-bone examples

End-to-end walkthrough: create a segmentation volume, generate meshes, and view both the volume and meshes in neuroglancer.

## Run in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/janelia-cellmap/mesh-n-bone/blob/master/examples/example.ipynb)

The [example notebook](example.ipynb) does the steps below in one click — install, build the zarr volume, run meshify, and launch a local neuroglancer viewer.

## Prerequisites

```bash
git clone https://github.com/janelia-cellmap/mesh-n-bone.git
cd mesh-n-bone
pixi install
```

## Steps

### 1. Create an example zarr volume

```bash
pixi run python examples/create_example_volume.py
```

This creates a 256x256x256 uint8 segmentation volume with a sphere (label 1) and a cube (label 2) at `examples/data/example.zarr/seg/s0`.

### 2. Generate meshes and multiresolution output

```bash
pixi run mesh-n-bone meshify examples/meshify-config -n 1
```

This runs the full pipeline: marching cubes, simplification, and multiresolution Draco-compressed output. Results are written to `examples/output/`.

### 3. View in neuroglancer

```bash
pixi run mesh-n-bone serve examples --zarr data/example.zarr/seg/s0 --meshes output/multires
```

This starts a local HTTP server with CORS headers and prints a neuroglancer URL. Open the URL in your browser to see both the segmentation volume and the generated meshes.
