# Supervised fine-tuning

Train a LoRA adapter on the public SmolLM2 135M model using four short example texts.

## How to run

1. Complete the [repository setup](../README.md#run-an-example).
2. Open **Actions → Supervised fine-tuning → Run workflow** and choose `main`.
3. Download the `sft` output artifact from the completed run (the full artifact name includes the attempt or matrix values).

To run on a configured NVIDIA GPU machine:

```sh
uv run --locked --no-dev --extra training python supervised-fine-tuning/run.py --output outputs/sft
```

Run this command from the repository root. The shared `pyproject.toml` and `uv.lock` install the packages for this example.

## What the run checks

The default run trains for two steps, saves a checkpoint, then starts a new Python process, resumes the checkpoint and trains step three. The resumed run fails unless step three changes the adapter weights saved in the checkpoint. The learning rate is constant so the resumed step still trains. This checks checkpoint loading on the same runner. It does not test recovery from an interrupted instance. Replace the toy texts and add held-out evaluation before using this for a real training task.

Results include the GPU and PyTorch version. Artifacts expire after seven days, so download anything you want to keep. See [verification status](../docs/verification.md) for the runs completed against this version.
