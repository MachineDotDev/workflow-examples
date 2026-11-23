# Verification status

All five examples ran successfully from `main` on Machine GPU runners on
2026-11-23. SFT and GRPO checkpoint resume was verified on GPU.

## GPU runs (2026-11-23)

**Hardware and versions.** Every job ran on a Machine `gpu=t4` runner
(`g4dn.xlarge`, Tesla T4) with Python 3.12, torch 2.14.1+cu126, torchvision
0.29.1, transformers 5.19.0, datasets 5.1.0, peft 0.21.2, trl 1.15.0,
accelerate 1.15.0 and Pillow 13.0.0 from `uv.lock`. Four examples ran at commit
[`3ad1189`](https://github.com/MachineDotDev/workflow-examples/commit/3ad1189);
tuning ran at
[`0ee1c00`](https://github.com/MachineDotDev/workflow-examples/commit/0ee1c00),
which only removes `max-parallel` from its workflow. Each workflow ran with its
default inputs.

| Example | Run | Result | Job time |
| --- | --- | --- | --- |
| Batch object detection | [45104694881](https://github.com/MachineDotDev/workflow-examples/actions/runs/45104694881) | 17 detections on 2 images | 1 min 28 s |
| Supervised fine-tuning | [45104926134](https://github.com/MachineDotDev/workflow-examples/actions/runs/45104926134) | 2 steps, checkpoint saved; a new process resumed `checkpoint-2` and trained step 3 at learning rate 2e-4 (loss 6.39, grad norm 1.65); the adapter changed relative to the checkpoint | 1 min 55 s |
| GRPO fine-tuning | [45105198305](https://github.com/MachineDotDev/workflow-examples/actions/runs/45105198305) | 2 steps, then a new process resumed `checkpoint-2` and trained step 3 at learning rate 1e-5 (grad norm 0.044); the adapter changed relative to the checkpoint | 1 min 53 s |
| Parallel hyperparameter tuning | [45135746263](https://github.com/MachineDotDev/workflow-examples/actions/runs/45135746263) | 4 matrix jobs at once, each 2 steps; losses after step 2 between 2.15 and 4.77 | 11 min 12 s – 11 min 15 s each |
| LLM evaluation gate | [45135170178](https://github.com/MachineDotDev/workflow-examples/actions/runs/45135170178) | Passed. Baseline (`main`) and candidate both 59.4% on 500 ARC Easy questions; 0 lost, 0 gained | 3 min 10 s |

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
| Workflow rules | `scripts/validate_examples.py` checks every GPU job runs from `main` in the `gpu` environment with a timeout and pinned Actions |
| Secret scanning and push protection | Enabled |
| Dependabot | Security updates enabled; monthly version updates for uv, Actions and pre-commit |
