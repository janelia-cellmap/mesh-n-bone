# Fileglancer integration

mesh-n-bone is a [Fileglancer](https://github.com/JaneliaSciComp/fileglancer) app: `runnables.yaml` at the repo root describes a form, Fileglancer renders it, and submitting it runs `mesh-n-bone meshify` as a cluster job.

## Files

- [`../runnables.yaml`](../runnables.yaml): the Fileglancer manifest. One runnable, `meshify`, with the input and output fields and sections for the output layout, levels of detail, resolution, simplification and smoothing, region and segments, segment properties, compute, and advanced options.
- [`run_meshify.py`](run_meshify.py): the wrapper the job runs. Fileglancer passes the form as flags; the wrapper decides where the meshes go and which LSF project the workers bill to, writes `run-config.yaml` and `dask-config.yaml` under `$FG_WORK_DIR/meshify-config/`, and invokes `mesh-n-bone meshify`.
- [`dask-config.yaml`](dask-config.yaml): the dask-jobqueue settings of the worker jobs the driver launches, used instead of `lsf-config/dask-config.yaml`.

## Registering the app

In Fileglancer, open **Apps**, then **My Apps**, then **Add from URL**, and give the repository URL:

```
https://github.com/janelia-cellmap/mesh-n-bone
```

A branch or tag can be chosen; a release tag (`v0.9.0`) pins the version. Fileglancer clones the repo at that revision and, on the first launch, `pixi run` builds the environment from `pixi.lock` (a few minutes, once per revision). The job needs `pixi` on the cluster node and access to conda-forge and PyPI for that first build.

## The form

Fileglancer pre-fills every field with its default and does not mark it as one, so each description states the default and what a blank field means. The defaults are mesh-n-bone's own (`Meshify`'s signature; a test pins them) and give hemibrain-density meshes. A first run is the **Label volume** and, optionally, an **Output directory**; nothing else needs touching. Only **Compute** (the worker count) starts open; every tuning section is collapsed, and **Advanced** holds the memory and quality trade-offs. A debugging option (keeping the per-level PLY files) is behind the form's **Show hidden** toggle.

The level-of-detail fields, since they are the ones people ask about:

- **Levels of detail** is a ceiling that counts the full-resolution level, so 4 means s0 to s3 at most. The input's pyramid levels are used first and any beyond them are built. A segment ships with fewer levels when a coarser mesh stops getting smaller or falls under 50 faces.
- **LOD strategy** `downsample` re-meshes a coarser volume for each level, the input's own pyramid levels when it has them; `decimate` meshes once and decimates.
- **Face count ratio between levels** (6) is how many times fewer faces each level has than the one before. **Target face reduction** (0.933), under Simplification, is the fraction removed from the full-resolution mesh. They are different numbers for different things.
- **Downsample method** is how label voxels are merged whenever a coarser volume has to be made: for pyramid levels the input lacks, and for the **Downsample factor** under Resolution, which shrinks the volume before meshing at all.

## Where the meshes go

The **Label volume** field takes a label group of an OME-Zarr, an N5 dataset or a precomputed volume. Leave **Output directory** blank and the meshes go next to the input, in a sibling named after it plus the **Mesh folder suffix**:

```
crop-001.zarr/
  labels/
    mitochondria/        <- Label volume
    mitochondria_mesh/   <- written here
    nucleus/
    zarr.json
  raw/
  zarr.json
```

No convention is read for this: `cells.n5/seg` is meshed to `cells.n5/seg_mesh` the same way. Point at the group rather than a level inside it, or `mito/s0` gets `mito/s0_mesh` inside the group. A remote URL has no "next to", so it needs an explicit output directory. One submission is one label; Fileglancer's relaunch pre-fills the form, so the next label is one field changed. An explicit output directory is used as given, and Fileglancer creates it if it does not exist.

Each mesh folder is a Neuroglancer precomputed multiresolution mesh source with an `info` file and, by default, shards. To view it, add it to the label's segmentation layer as a second source, `precomputed://<url of the _mesh folder>`, where `<url>` is the Fileglancer data link of the folder.

## Multiscale inputs

A label group with its own pyramid (`s0`, `s1`, ...) is meshed from its finest level, and its coarser levels are used as the levels of detail, up to the **Levels of detail** count, whatever factors they were built with, including anisotropic ones. Levels that are missing are built once under the output directory and deleted afterwards unless **Delete the levels built for this run** (under Advanced) is off. Volumes stored x, y, z (as TensorSwitch writes them) are read z, y, x; a ROI is always given z, y, x in world units.

## Compute

Fileglancer's job is the driver (the **Resources** defaults: 2 cpus, 30 GB, 8 hours). The driver reads [`dask-config.yaml`](dask-config.yaml) from this directory and launches the worker jobs with dask-jobqueue. The one compute field is **Dask workers**, worker processes in total; with 8 processes per child job, 48 workers are 6 child jobs.

The worker jobs are billed where the driver is. The wrapper reads the driver's project from `LSB_PROJECT_NAME` when LSF sets it (what cellmap-flow reads) and otherwise asks `bjobs` for its own job id, then passes that to dask-jobqueue as the workers' `-P`. So the project is set once, in Fileglancer's **Cluster** tab under Extra Arguments as `-P <project>`, or not at all when LSF's default for your account is the right one, which is the usual case at Janelia. The job log prints what LSF told the job and which project the workers got; when the driver has no project and your account must pass `-P`, the log says so, and the fix is the Extra Arguments field. Don't add `-n` there; the CPUs field sets it.

Per child job, `dask-config.yaml` asks for 8 slots, 8 processes and 120 GB. Change it here for the Fileglancer flow; the command-line flow keeps using `lsf-config/dask-config.yaml`.

The worker jobs get the driver's walltime: the wrapper reads the `-W` from the job script Fileglancer writes and gives it to the workers, and the log prints it. A one-hour limit would put the workers in the short queue, but dask-jobqueue never replaces a worker job that ends, so a block that takes longer than the workers' limit used to leave the driver waiting, idle, until its own limit. Workers that lose the driver exit within a minute, so a killed or finished driver takes its workers with it. If every worker does exit while work remains, the run now fails within ten minutes and says how the worker jobs ended.

## Trying the wrapper by hand

```bash
pixi run python fileglancer/run_meshify.py \
  --input examples/data/example.zarr/seg \
  --output /tmp/meshify-fg-test \
  --num-workers 1 --dry-run
```

`--dry-run` prints the run and its configs without meshing. Without `FG_WORK_DIR`, the configs land in `./meshify-config/`. Outside an LSF job there is no `LSB_PROJECT_NAME`, so `--lsf-project <name>` gives the workers a project; it also overrides the inherited one inside a job.

## Adding more runnables

Each subcommand (`to-neuroglancer`, `skeletonize`, `analyze`) can follow the same pattern: a runnable in `runnables.yaml` and a `run_<subcommand>.py` wrapper that writes its config directory.
