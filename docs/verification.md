# Verification status

All five examples ran successfully from `main` on Machine GPU runners on
2026-11-23, after the ML library update. SFT and GRPO checkpoint resume was
verified on GPU.

## GPU runs (2026-11-23)

**Hardware and versions.** Every job ran on a Machine `gpu=t4` runner
(`g4dn.xlarge`, Tesla T4) with Python 3.12, torch 2.14.1+cu126, torchvision
0.29.1, transformers 5.23.0, datasets 5.2.0, peft 0.22.0, trl 1.18.2,
accelerate 1.16.0, timm 1.0.31 and Pillow 13.0.0 from `uv.lock`, at commit
[`6f2ea2d`](https://github.com/MachineDotDev/workflow-examples/commit/6f2ea2d).
Each workflow ran with its default inputs.

| Example | Run | Result | Job time | Billed |
| --- | --- | --- | --- | --- |
| Batch object detection | [45261603223](https://github.com/MachineDotDev/workflow-examples/actions/runs/45261603223) | 17 detections on 2 images | 1 min 22 s | $0.04 |
| Supervised fine-tuning | [45262001065](https://github.com/MachineDotDev/workflow-examples/actions/runs/45262001065) | 2 steps, checkpoint saved; a new process resumed `checkpoint-2` and trained step 3 at learning rate 2e-4 (loss 6.39, grad norm 1.65); the adapter changed relative to the checkpoint | 1 min 46 s | $0.04 |
| GRPO fine-tuning | [45262459913](https://github.com/MachineDotDev/workflow-examples/actions/runs/45262459913) | 2 steps (reward mean 0.50, std 0.71), then a new process resumed `checkpoint-2` and trained step 3 at learning rate 1e-5 (grad norm 0.044); the adapter changed relative to the checkpoint | 2 min 2 s | $0.04 |
| Parallel hyperparameter tuning | [45262985499](https://github.com/MachineDotDev/workflow-examples/actions/runs/45262985499) | 4 matrix jobs at once, each 2 steps; losses after step 2 between 2.15 and 4.77 | 12 min 53 s – 12 min 57 s each | 4 × $0.25 |
| LLM evaluation gate | [45265339879](https://github.com/MachineDotDev/workflow-examples/actions/runs/45265339879) | Passed. Baseline (`main`) and candidate both 59.4% on 500 ARC Easy questions; 0 lost, 0 gained | 3 min 12 s | $0.08 |

Job time is GitHub's job start to finish and includes the model and dataset
downloads. Most of each tuning job was the 170 MB CIFAR-10 download from its
host at the University of Toronto, which ran at about 350 kB/s that day.

**What these runs prove.** Each workflow installs from the lock, finds CUDA,
runs its workload and uploads its artifacts on a Machine GPU runner. SFT and
GRPO load a saved checkpoint in a fresh process on the same runner. The
evaluation gate gives identical code zero disagreements on the same GPU.

**What they don't.** Two or three training steps measure execution, not model
quality. The tuning losses don't rank the configurations. Resume was tested on
the same runner, not after a lost instance or a spot interruption.

**The gate rejects a real defect.** Run with `defect=truncated-question`, which
cuts each question to 48 characters before the answerer sees it, the candidate
scored 49.0% against the baseline's 59.4%: 78 questions lost, 26 gained,
p = 1.6e-7, so the job failed as intended. That run used the same evaluation
code on a T4 runner on 2026-11-15, before this repository existed.

## Repository checks

| Check | Evidence |
| --- | --- |
| Linux installation and ML APIs | [CI passed](https://github.com/MachineDotDev/workflow-examples/actions/runs/44851680125): locked CUDA packages installed; trainer configurations, imports and the evaluation gate's unit tests passed |
| Workflow rules | `scripts/validate_examples.py` checks every job with a Machine runner label is dispatch-only, runs from `main` in the `gpu` environment with read-only permissions and a timeout, and that every Action is pinned to a full SHA |
| Secret scanning and push protection | Enabled |
| Dependabot | Security updates enabled; monthly version updates for uv, Actions and pre-commit |
