# Batch object detection

DETR finds objects in two COCO validation images and writes their labels, scores and bounding boxes.

The [machine.dev guide](https://machine.dev/docs/use-cases/gpu-object-detection/?utm_source=github&utm_medium=oss&utm_campaign=2026-11-workflow-examples&utm_content=batch-object-detection) walks through the workflow, what a real run found and how to adapt it to your project.

## How to run

1. Complete the [repository setup](../README.md#run-an-example).
2. Open **Actions → Batch object detection → Run workflow** and choose `main`.
3. Download the `detection` output artifact from the completed run (the full artifact name includes the attempt or matrix values).

To run on a configured NVIDIA GPU machine:

```sh
uv run --locked --no-dev --extra vision python batch-object-detection/run.py --output outputs/detection
```

Run this command from the repository root. The shared `pyproject.toml` and `uv.lock` install the packages for this example.

## What the run checks

The run fails if it finds no objects or produces non-finite boxes. The images are downloaded at runtime; network or upstream model failures will fail the job.

Results include the GPU and PyTorch version. Artifacts expire after seven days, so download anything you want to keep. See [verification status](../docs/verification.md) for the runs completed against this version.
