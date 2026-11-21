# Workflow controls

The repository follows the relevant controls from [nat-zero](https://github.com/MachineDotDev/nat-zero). Terraform, Packer, AWS deployment and Go checks don't apply here.

## GPU approval

Only `workflow_dispatch` starts GPU jobs. Each job checks that it is running from `main` and checks out the event's exact commit. GitHub requires write access to dispatch a workflow. Automatic pull request checks run on GitHub-hosted CPU runners with no repository secrets.

Every GPU job runs in the `gpu` environment, which requires Leonard's approval before the job starts, with no administrator bypass. Pull requests from external contributors need approval before any workflow runs. All GPU workflows share a concurrency group and matrix jobs run sequentially. GitHub keeps one pending run per group, so dispatch and finish one example before starting the next.

## Maintenance

Default workflow tokens have read-only access. Only the release job receives contents and pull-request write access. Allowed external Action publishers are `astral-sh/*`, `googleapis/*` and `pre-commit/*`, alongside GitHub-owned Actions. Source files pin Actions to full SHAs.

Dependabot checks uv, Actions and pre-commit monthly. It skips `torch` and `torchvision`, which install from the PyTorch CUDA index; update those by hand. Secret scanning, push protection and Dependabot security updates are enabled. Report vulnerabilities through the contact in [SECURITY.md](../SECURITY.md).

## Releases

Versions apply to the whole workflow collection. Run **Release** manually from `main` after verification. Release Please opens or updates a version pull request. Review its changelog and version changes before merging. Run **Release** again after merge to create the immutable version tag and GitHub release. GitHub provides source archives for the tag.

Release Please uses `GITHUB_TOKEN`. GitHub may hold bot-created PR checks for approval. Review the exact version diff and approve the checks in Actions. If no runs appear because GitHub suppressed the token-created event, close and reopen the PR as a maintainer. Don't bypass missing checks. Release workflows don't run GPU workloads.

Before publishing a version, record successful GPU run URLs for its workload code in [verification status](verification.md). A release tag alone isn't evidence that training worked.

Branch and tag rulesets are active. The main branch requires one approval, code-owner review, stale approval dismissal, last-push approval, resolved conversations and strict `precommit`, `dependencies` and `semantic-pr-title` checks. Merges are squash-only. As in nat-zero, administrators can bypass through a pull request; version tag updates and deletion are blocked with an administrator bypass. These controls prevent ordinary direct pushes, not an administrator deliberately overriding them.
