from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_cli_help() -> None:
    project_root = Path(__file__).resolve().parents[2]
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(project_root / "src")

    proc = subprocess.run(
        [sys.executable, "-m", "visual_servo_tracker.cli", "--help"],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    help_text = (proc.stdout or "") + (proc.stderr or "")
    assert "offline-run" in help_text
