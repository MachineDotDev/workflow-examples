# Parallel hyperparameter tuning

Train ResNet50 on CIFAR-10 with four combinations of learning rate and batch size.

## How to run

1. Complete the [repository setup](../README.md#run-an-example).
2. Open **Actions → Parallel hyperparameter tuning → Run workflow** and choose `main`.
3. Download the `tuning` output artifact from the completed run (the full artifact name includes the attempt or matrix values).

To run on a configured NVIDIA GPU machine:

```sh
uv run --locked --no-dev --extra vision python parallel-hyperparameter-tuning/run.py --output outputs/tuning
```

Run this command from the repository root. The shared `pyproject.toml` and `uv.lock` install the packages for this example.

## What the run checks

The matrix uses learning rates 0.001 and 0.0001 with batch sizes 8 and 16. Each job trains for two steps and checks that model parameters changed. All four jobs start together. Machine starts each one when your team has a free runner slot; jobs above your team's limit wait their turn. Two steps measure execution, not model quality.

Results include the GPU and PyTorch version. Artifacts expire after seven days, so download anything you want to keep. See [verification status](../docs/verification.md) for the runs completed against this version.
