# Fileglancer integration

mesh-n-bone is a [Fileglancer](https://github.com/JaneliaSciComp/fileglancer) app: `runnables.yaml` at the repo root describes a form, Fileglancer renders it, and submitting it runs `mesh-n-bone meshify` as a cluster job.

## Files

- [`../runnables.yaml`](../runnables.yaml): the Fileglancer manifest. One runnable, `meshify`, with form fields for the input and output, mesh generation, simplification, multiresolution output, ROI and segment selection, segment properties and compute.
- [`run_meshify.py`](run_meshify.py): the wrapper the job runs. Fileglancer passes the form as flags; the wrapper decides which runs to make and where their meshes go, writes a `run-config.yaml` and `dask-config.yaml` per run under `$FG_WORK_DIR/meshify-config/`, and invokes `mesh-n-bone meshify` on each.
- [`dask-config.yaml`](dask-config.yaml): the dask-jobqueue settings of the worker jobs the driver launches, used instead of `lsf-config/dask-config.yaml`.

## Registering the app

In Fileglancer, open **Apps**, then **My Apps**, then **Add from URL**, and give the repository URL:

```
https://github.com/janelia-cellmap/mesh-n-bone
```

A branch or tag can be chosen; a release tag (`v0.9.0`) pins the version. Fileglancer clones the repo at that revision and, on the first launch, `pixi run` builds the environment from `pixi.lock` (a few minutes, once per revision). The job needs `pixi` on the cluster node and access to conda-forge and PyPI for that first build.

## Where the meshes go

The **Label volume** field takes a label group of an OME-Zarr, an array, an N5 or precomputed volume, or a zarr root. Leave **Output directory** blank and the wrapper places the meshes by the OME-Zarr layout:

```
crop-001.zarr/
  labels/
    mitochondria/        <- Label volume
    mitochondria_mesh/   <- written here
    nucleus/
    nucleus_mesh/
    zarr.json
  raw/
  zarr.json
```

- A label group (`.../labels/<name>`) or an array in it is meshed to `.../labels/<name>_mesh`.
- A zarr root whose `labels/` group lists labels (`attributes.ome.labels` in zarr v3, `labels` in v2) is meshed label by label, each to its own `_mesh` folder, in one submission.
- The suffix is the **Mesh folder suffix** field.
- Any other input, including a remote URL, needs an explicit output directory. Fileglancer creates it if it does not exist.

Each mesh folder is a Neuroglancer precomputed multiresolution mesh source with an `info` file and, by default, shards. To view it, add it to the label's segmentation layer as a second source, `precomputed://<url of the _mesh folder>`, where `<url>` is the Fileglancer data link of the folder.

## Multiscale inputs

A label group with its own pyramid (`s0`, `s1`, ...) is meshed from its finest level, and the coarser levels are the levels of detail, whatever factors they were built with, including anisotropic ones. Levels that are missing are built once under the output directory and deleted afterwards unless **Delete the levels built for this run** is off. Volumes stored x, y, z (as TensorSwitch writes them) are read z, y, x; a ROI is always given z, y, x in world units.

## Compute

Fileglancer's job is the driver (the **Resources** defaults: 2 cpus, 30 GB, 8 hours). The driver reads [`dask-config.yaml`](dask-config.yaml) from this directory, puts the form's **LSF project** into it, and launches the worker jobs with dask-jobqueue. The two compute fields are:

- **Dask workers**: worker processes in total. With 8 processes per child job, 48 workers are 6 child jobs.
- **LSF project**: the charge group of the worker jobs, the name only.

Per child job, `dask-config.yaml` asks for 8 slots, 8 processes, 120 GB and one hour. Change it here for the Fileglancer flow; the command-line flow keeps using `lsf-config/dask-config.yaml`. The driver job's own charge group is set in Fileglancer's **Cluster** tab under extra `bsub` arguments (`-P <project>`); don't add `-n` there, the CPUs field sets it.

## Trying the wrapper by hand

```bash
pixi run python fileglancer/run_meshify.py \
  --input examples/data/example.zarr/seg \
  --output /tmp/meshify-fg-test \
  --num-workers 1 --dry-run
```

`--dry-run` prints the runs and their configs without meshing. Without `FG_WORK_DIR`, the configs land in `./meshify-config/`.

## Adding more runnables

Each subcommand (`to-neuroglancer`, `skeletonize`, `analyze`) can follow the same pattern: a runnable in `runnables.yaml` and a `run_<subcommand>.py` wrapper that writes its config directory.
