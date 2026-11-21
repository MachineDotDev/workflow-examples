# Contributing

Changes go through a pull request with a conventional title, such as `fix: restore training checkpoints`.

Run `uv run --no-sync pre-commit run --all-files` before pushing. CI checks Python syntax, workflow permissions, pinned Actions and the dependency lock. A separate Linux job installs the ML packages and imports the training APIs.

Workload and dependency changes also need a real GPU run. A maintainer reviews the change, merges it, then manually dispatches the relevant workflow from `main`. Don't call a dependency update verified because imports passed. Record the commit, run URL and result in `docs/verification.md`.

Keep smoke defaults small. Bigger training runs belong behind explicit inputs with limits. Don't add automatic model publication or credentials to examples.

## Dependency updates

Dependabot groups monthly updates for uv, Actions and pre-commit. Review each group, resolve any API changes and regenerate the lock with `uv lock --upgrade`. Keep torch and torchvision on the explicit CUDA wheel index. Test affected workloads before a release.
