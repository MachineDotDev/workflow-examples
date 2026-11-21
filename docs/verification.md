# Verification status

This repository remains private. All five examples ran successfully on real
Machine GPU runners on 2026-11-15; SFT and GRPO checkpoint resume and the LLM
evaluation gate's defect rejection were verified on GPU.

## GPU runs (2026-11-15)

**Where they ran.** The development Machine GitHub App can't see this private
repository yet, so the runs used an exact copy of commit `906e3d1` (the
evaluation-gate change, before rebasing onto #2's docs-only commit) pushed to
the `verify/workflow-examples` branch of the private `MachineDotDev/examples`
repository, which the development App can reach. The copy is commit
[`1afd5ee`](https://github.com/MachineDotDev/examples/commit/1afd5ee). Only
three things differ from this repository: each GPU job's branch guard names
that branch instead of `main`, each workflow has its own concurrency group so
the six runs could queue together, and the CPU-only CI workflows were removed.
Code, dependencies, runner labels, timeouts and inputs are unchanged.

**Hardware and versions.** Every job ran on a Machine `gpu=t4` runner
(`g4dn.xlarge`, Tesla T4, default tenancy) in the development environment, with
Python 3.12, uv 0.13.0, torch 2.14.1+cu126, torchvision 0.29.1, transformers
5.19.0, datasets 5.1.0, peft 0.21.2, trl 1.15.0 and accelerate 1.15.0 from
`uv.lock`.

| Example | Run | Inputs | Result | Job time | Billed |
| --- | --- | --- | --- | --- | --- |
| LLM evaluation gate | [43684118729](https://github.com/MachineDotDev/examples/actions/runs/43684118729) | `baseline_ref=verify/workflow-examples`, `defect=none` | Passed. Baseline and candidate 59.4% on 500 ARC Easy questions; 0 lost, 0 gained, p = 1.0 | 3 min 2 s | $0.06 |
| LLM evaluation gate (deliberate defect) | [43684132171](https://github.com/MachineDotDev/examples/actions/runs/43684132171) | `baseline_ref=verify/workflow-examples`, `defect=truncated-question` | **Failed as intended.** Candidate 49.0% vs baseline 59.4%; lost 78, gained 26, p = 1.6e-7 | 2 min 53 s | $0.06 |
| Batch object detection | [43684121108](https://github.com/MachineDotDev/examples/actions/runs/43684121108) | none | 17 detections on 2 images (both cats, both remotes and the couch in image 0) | 1 min 3 s | $0.04 |
| Supervised fine-tuning | [43693099757](https://github.com/MachineDotDev/examples/actions/runs/43693099757) | `resume=true` | 2 steps, checkpoint saved; a new process resumed `checkpoint-2` and trained step 3 at the constant learning rate 2e-4 (loss 6.39, grad norm 1.65); the adapter changed relative to the checkpoint; adapter and checkpoints uploaded | 1 min 30 s | $0.04 |
| GRPO fine-tuning | [43693101979](https://github.com/MachineDotDev/examples/actions/runs/43693101979) | `resume=true` | 2 steps, then a new process resumed `checkpoint-2` and trained step 3 at learning rate 1e-5; the adapter changed relative to the checkpoint. Steps 1–2: reward mean 0.50, std 0.71 (one right, one wrong answer). Step 3: reward mean 0.0007, both answers wrong, separated only by the length term; grad norm 0.044 | 1 min 32 s | $0.04 |
| Parallel hyperparameter tuning | [43684128616](https://github.com/MachineDotDev/examples/actions/runs/43684128616) | none | 4 matrix jobs, one at a time, each 2 steps with changed parameters | 4 × 1–1.5 min | 4 × $0.04 |

**SFT and GRPO were re-run after a fix.** The first runs
([43684123461](https://github.com/MachineDotDev/examples/actions/runs/43684123461),
[43684125739](https://github.com/MachineDotDev/examples/actions/runs/43684125739))
passed, but review found the resumed step made no update: the default linear
schedule, sized to the first two steps, had already reached a learning rate of
0, and the weight-change check compared with the fresh LoRA initialisation
rather than the checkpoint. Commit `c30109a` switched to a constant schedule and
compares a resumed run with the checkpoint's adapter; the rows above are the
re-runs of that code (verification copy `a9a106d`).

Job time is GitHub's job start to finish and includes the model and dataset
downloads. Billed is the Machine charge recorded for each job in the
development environment. Queueing before the runner started took between about one and four minutes,
partly because six jobs arrived at once against the team's concurrency limit.

**What these runs prove.** Each workflow installs from the lock, finds CUDA,
runs its workload and uploads its artifacts on a Machine GPU runner. SFT and
GRPO load a saved checkpoint in a fresh process on the same runner. The
evaluation gate gives identical code zero disagreements on the same GPU, and
fails a real application defect with the unchanged test.

**What they don't.** Two or three training steps measure execution, not model
quality. The tuning losses (2.2–4.8 after two steps) don't rank the
configurations. Resume was tested on the same runner, not after a lost
instance or a spot interruption. The gate detects large regressions on this
sample; small ones need more questions (see the example README).

**Local calibration (not GPU evidence).** Before the GPU runs, the gate was run
on Apple MPS in float32: baseline 59.6%, defect 49.2%, lost 78, gained 26.

## Repository checks

| Check | Evidence |
| --- | --- |
| Dependency resolution | `uv.lock` resolved with uv 0.13.0; the `evaluation` extra no longer installs lm-eval |
| Local checks | All pre-commit hooks passed, including actionlint, Gitleaks and permission checks |
| Linux installation and ML APIs | [CI run passed](https://github.com/MachineDotDev/workflow-examples/actions/runs/38244966645): locked CUDA packages installed; current trainer configurations and imports passed |
| Evaluation gate unit tests | 9 CPU tests: McNemar decision, answer selection with a stub model, prompt construction, defect patch applies |
| Release automation | [Release workflow passed](https://github.com/MachineDotDev/workflow-examples/actions/runs/38245049022) and opened [version PR #1](https://github.com/MachineDotDev/workflow-examples/pull/1); no version tag or release published |
| Release lock consistency | Tested the pinned Release Please 17.6.0 TOML updater against both manifests; simulated version bump passed `uv lock --check` |
| Dependency graph | [uv graph update passed](https://github.com/MachineDotDev/workflow-examples/actions/runs/38244975773); monthly Dependabot version jobs configured |
| Secret scanning and push protection | Enabled, verified through GitHub API |
| Dependabot security updates | Enabled, verified through GitHub API |
| Environment reviewer approval | Unavailable on current private-repository plan; manual dispatch gate used |
| Branch and tag protection | Active rulesets `main` (24925412) and `tags` (24925413), adapted from nat-zero |
| Explore | Draft [Machine PR #699](https://github.com/MachineDotDev/machine/pull/699); component tests, production build and pre-commit passed |

## Review and rollout

Fresh-context review covered the complete initial repository and the Explore follow-up. The release lockfile update finding was fixed and tested with the pinned updater. Local checks and GitHub CPU CI passed on commit `b72bc5d`.

Release Please's bot-created PR checks entered `action_required`. After reviewing the version-only diff, the checks were approved through GitHub's run-approval API. The semantic title check passed. This confirms the bot approval flow, not a GPU environment approval gate.

Keep the release PR unmerged until publication is approved. No live Explore links or legacy migration notices should point here until publication is explicitly approved.
