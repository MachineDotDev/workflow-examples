# Security

Report a suspected vulnerability privately through [support@machine.dev](mailto:support@machine.dev). Include the affected file, commit and reproduction steps. Don't put credentials or private customer data in a public issue.

GitHub tokens default to read-only access. GPU workflows run only after a maintainer dispatches them from `main`; pull request checks use GitHub-hosted CPU runners. Checkout doesn't persist credentials. Actions are pinned to full commit SHAs.

See [workflow controls](docs/workflows.md). Never run unreviewed contributor code on a GPU runner with access to your secrets.
