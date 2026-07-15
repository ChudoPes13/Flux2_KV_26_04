"""End-to-end validation of the activated remote runtime."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_runtime_smoke(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output = tmp_path / "runtime_smoke.json"
    environment = os.environ.copy()
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/01_runtime_smoke.py",
            "--strict",
            "--output",
            str(output),
        ],
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "pass"
