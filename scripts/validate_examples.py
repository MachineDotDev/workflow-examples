"""Check the catalog, immutable Actions and GPU execution boundaries."""

import ast
import json
import re
from pathlib import Path

import yaml

root = Path(__file__).resolve().parents[1]
examples = json.loads((root / "examples.json").read_text())
assert len(examples) == 5
for example in examples:
    assert (root / example["path"] / "README.md").is_file()
    assert (root / example["path"] / "run.py").is_file()
    assert (root / example["workflow"]).is_file()
for path in root.rglob("*.py"):
    if ".venv" not in path.parts:
        ast.parse(path.read_text(), filename=str(path))
for path in (root / ".github/workflows").glob("*.yml"):
    data = yaml.safe_load(path.read_text())
    triggers = data.get("on", data.get(True))
    assert "pull_request_target" not in triggers, path
    for job in data["jobs"].values():
        runner = str(job.get("runs-on", ""))
        if "machine" in runner:
            assert runner.startswith("machine/"), path
            assert set(triggers) == {"workflow_dispatch"}, path
            assert (
                job["if"]
                == "github.event_name == 'workflow_dispatch' && github.ref == 'refs/heads/main'"
            ), path
            assert "permissions" not in job, path
            assert job["timeout-minutes"] <= 30, path
            assert job.get("environment") == "gpu", path
            assert "${{ github.run_id }}-${{ github.run_attempt }}" in runner, path
            assert "/gpu=t4" in runner, path
            assert data["permissions"] == {"contents": "read"}, path
        for step in job.get("steps", []):
            if "uses" in step:
                assert re.fullmatch(r"[\w.-]+/[\w./-]+@[0-9a-f]{40}", step["uses"]), path
                if step["uses"].startswith("actions/checkout@"):
                    assert step["with"]["persist-credentials"] is False, path
print("Five examples and all workflow permission checks passed")
