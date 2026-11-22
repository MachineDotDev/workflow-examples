# Verification status

All five examples ran successfully from `main` on Machine GPU runners on
2026-11-22. SFT and GRPO checkpoint resume was verified on GPU.

## GPU runs (2026-11-22)

**Hardware and versions.** Every job ran on a Machine `gpu=t4` runner
(`g4dn.xlarge`, Tesla T4) with Python 3.12, torch 2.14.1+cu126, torchvision
0.29.1, transformers 5.19.0, datasets 5.1.0, peft 0.21.2, trl 1.15.0 and
accelerate 1.15.0 from `uv.lock`, at commit
[`48c5081`](https://github.com/MachineDotDev/workflow-examples/commit/48c5081).
Each workflow ran with its default inputs.

| Example | Run | Result | Job time |
| --- | --- | --- | --- |
| Batch object detection | [45080649018](https://github.com/MachineDotDev/workflow-examples/actions/runs/45080649018) | 17 detections on 2 images | 4 min 3 s |
| Supervised fine-tuning | [45081139590](https://github.com/MachineDotDev/workflow-examples/actions/runs/45081139590) | 2 steps, checkpoint saved; a new process resumed `checkpoint-2` and trained step 3 at learning rate 2e-4 (loss 6.39, grad norm 1.65); the adapter changed relative to the checkpoint | 2 min 28 s |
| GRPO fine-tuning | [45081511145](https://github.com/MachineDotDev/workflow-examples/actions/runs/45081511145) | 2 steps (reward mean 0.50, std 0.71), then a new process resumed `checkpoint-2` and trained step 3 at learning rate 1e-5 (grad norm 0.044); the adapter changed relative to the checkpoint | 1 min 45 s |
| Parallel hyperparameter tuning | [45081769592](https://github.com/MachineDotDev/workflow-examples/actions/runs/45081769592) | 4 matrix jobs, one at a time, each 2 steps; losses after step 2 between 2.15 and 4.77 | 1 min 13 s – 2 min 6 s each |
| LLM evaluation gate | [45082457062](https://github.com/MachineDotDev/workflow-examples/actions/runs/45082457062) | Passed. Baseline (`main`) and candidate both 59.4% on 500 ARC Easy questions; 0 lost, 0 gained | 3 min 20 s |

Job time is GitHub's job start to finish and includes the model and dataset
downloads.

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
