# Workflow controls

The repository follows the relevant controls from [nat-zero](https://github.com/MachineDotDev/nat-zero). Terraform, Packer, AWS deployment and Go checks don't apply here.

## GPU approval

Only `workflow_dispatch` starts GPU jobs. Each job checks that it is running from `main` and checks out the event's exact commit. GitHub requires write access to dispatch a workflow. Automatic pull request checks run on GitHub-hosted CPU runners with no repository secrets.

Every GPU job runs in the `gpu` environment, which requires Leonard's approval before the job starts, with no administrator bypass. Pull requests from external contributors need approval before any workflow runs. All GPU workflows share a concurrency group. GitHub keeps one pending run per group, so dispatch and finish one example before starting the next.

## Maintenance

Default workflow tokens have read-only access. Allowed external Action publishers are `astral-sh/*` and `pre-commit/*`, alongside GitHub-owned Actions. Source files pin Actions to full SHAs.

Dependabot checks uv, Actions and pre-commit monthly. It skips `torch` and `torchvision`, which install from the PyTorch CUDA index; update those by hand. Secret scanning, push protection and Dependabot security updates are enabled. Report vulnerabilities through the contact in [SECURITY.md](../SECURITY.md).

## Branch protection

Branch and tag rulesets are active. The main branch requires one approval, code-owner review, stale approval dismissal, last-push approval, resolved conversations and strict `precommit`, `dependencies` and `semantic-pr-title` checks. Merges are squash-only. As in nat-zero, administrators can bypass through a pull request; version tag updates and deletion are blocked with an administrator bypass. These controls prevent ordinary direct pushes, not an administrator deliberately overriding them.
