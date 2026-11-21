# LLM evaluation gate

Fail a GitHub Actions run when a change to your LLM application answers
questions significantly worse than the baseline.

The application code here is a small multiple-choice answerer
([`answerer.py`](answerer.py)): it builds a prompt and picks the answer the
model scores highest. The workflow runs the baseline and the candidate version
of that code with the same pinned model on the same 500 ARC Easy questions,
then compares them question by question. The job fails when the candidate
loses significantly more questions than it gains.

## How to run

1. Complete the [repository setup](../README.md#run-an-example).
2. Open **Actions → LLM evaluation gate → Run workflow** on `main`.
3. Keep `baseline_ref` as `main` and `defect` as `none`. Both versions are the
   same code, so the gate passes with no disagreements.
4. Run it again with `defect` set to `truncated-question`. This applies
   [a deliberate bug](defects/truncated-question.patch) that cuts each question
   to 48 characters. The unchanged gate should fail the job.
5. Download the `evaluation-1` artifact (the suffix is the run attempt) from either run. `results.json` holds
   both accuracies, the disagreement counts and the decision.
   `predictions.jsonl` holds each question's answer from both versions.

To run on a Linux machine with an NVIDIA GPU, from the repository root:

```sh
git worktree add ../baseline main
uv run --locked --no-dev --extra evaluation python llm-evaluation/run.py \
  --baseline ../baseline/llm-evaluation --output outputs/evaluation
```

## How the decision works

Both versions answer the same questions, so the comparison is paired. Questions
both versions get right, or both get wrong, say nothing about the change. The
gate counts the questions the candidate **lost** (baseline right, candidate
wrong) and **gained**, and runs a one-sided exact McNemar test on them: if the
change made no difference, each disagreement would be equally likely to go
either way. The job fails when p < 0.05.

There's no accuracy threshold to tune. The test asks whether the candidate is
worse than *this* baseline on *these* questions, and the significance level is
the conventional one, not a number picked to make a run pass.

## What it proves and what it doesn't

- **The evaluation runs**: the model, dataset and both versions of your code
  load and answer on a GPU runner.
- **A regression decision**: a large loss against the baseline fails the job.
  Small losses are not detected. The margin needed depends on how many
  questions the two versions disagree on: with 20 disagreements the candidate
  must lose at least 15 of them to fail; with 100, at least 59.
- **Not a model-quality measurement**: ARC Easy accuracy for a 0.5B model says
  little about your application. Replace the questions with ones that represent
  what your users ask.

See [verification status](../docs/verification.md) for the GPU runs recorded
against this version.

## Adapt it

- Replace `answerer.py` with your own application code and keep `choose`'s
  signature, or change `run.py` to call your code. Helper modules next to
  `answerer.py` are imported separately for the baseline and the candidate.
- Replace the dataset with your own fixed set of questions and expected
  answers. Keep the same questions for the baseline and candidate.
- Pin the model revision. A moving revision changes the baseline under you.
- To gate pull requests, run on `pull_request` with the PR's base branch as the
  baseline. Machine runs PR code on your GPU runner: decide who can trigger it
  and don't use `pull_request_target` to execute untrusted code. The baseline
  runs on the GPU too, so only point `baseline_ref` at code you trust.

## Troubleshooting

- **`This example requires an NVIDIA GPU`**: the job didn't land on a GPU runner.
  Check the Machine installation and the `runs-on` label.
- **`No answerer.py in baseline/llm-evaluation`**: the baseline ref predates
  this example or has a different layout.
- **The job hits its 30-minute limit**: the first run downloads the model and
  dataset. Lower `--limit` only for a smoke run; a smaller sample detects only
  larger regressions.
