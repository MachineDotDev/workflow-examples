# GRPO fine-tuning

Train a LoRA adapter with GRPO using short arithmetic prompts and a public SmolLM2 model.

The [machine.dev guide](https://machine.dev/docs/use-cases/grpo-fine-tuning/?utm_source=github&utm_medium=oss&utm_campaign=2026-11-workflow-examples&utm_content=grpo-fine-tuning) walks through the workflow, what a real run found and how to adapt it to your project.

## How to run

1. Complete the [repository setup](../README.md#run-an-example).
2. Open **Actions → GRPO fine-tuning → Run workflow** and choose `main`.
3. Download the `grpo` output artifact from the completed run (the full artifact name includes the attempt or matrix values).

To run on a configured NVIDIA GPU machine:

```sh
uv run --locked --no-dev --extra training python grpo-fine-tuning/run.py --output outputs/grpo
```

Run this command from the repository root. The shared `pyproject.toml` and `uv.lock` install the packages for this example.

## What the run checks

The reward gives most credit for a correct number, with a small length reward to distinguish attempts. This is a toy reward, not a recommendation for production training. Each prompt produces two completions. The default run trains for two steps, then resumes the saved checkpoint in a new process and trains step three with a constant learning rate. Each run fails unless its steps change the adapter weights (for the resumed run, compared with the checkpoint); it does not prove the model got better or simulate a spot interruption.

Results include the GPU and PyTorch version. Artifacts expire after seven days, so download anything you want to keep. See [verification status](../docs/verification.md) for the runs completed against this version.
