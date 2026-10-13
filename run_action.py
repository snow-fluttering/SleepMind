"""Select the package-local Python and forward one workflow action without changing its output."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ACTIONS = {"transfer", "stage", "verify"}
HELP = """Usage:
  python run_action.py [--python PYTHON_EXE] transfer|stage|verify [action arguments]
  python run_action.py [--python PYTHON_EXE] --check

By default, actions use the platform-specific .venv Python. --python selects another
Python executable for relocated or separately managed test environments.
--check [--assets [PATH]] runs source health and optional private-asset checks.
"""


def error(message: str, code: int = 2) -> int:
    print(f"run_action.py: {message}", file=sys.stderr)
    return code


def main() -> int:
    arguments = sys.argv[1:]
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(HELP, end="")
        return 0 if arguments else 2

    runtime_arg: str | None = None
    if arguments[0] == "--python":
        if len(arguments) < 2:
            return error("--python requires an executable path.")
        runtime_arg = arguments[1]
        arguments = arguments[2:]
    if not arguments:
        return error("supply one action or --check.\n" + HELP)

    if runtime_arg is None:
        runtime = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    else:
        runtime = Path(runtime_arg).expanduser()
        if not runtime.is_absolute():
            runtime = ROOT / runtime
    if not runtime.is_file():
        return error(
            f"Python executable not found: {runtime}. Install the local environment "
            "with `python setup_env.py`, or pass --python with an existing executable."
        )

    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"

    if arguments[0] == "--check":
        if len(arguments) > 1 and arguments[1] != "--assets":
            return error("--check accepts only --assets with an optional private asset path.")
        target = ROOT / "health_check.py"
        forwarded = arguments[1:]
    else:
        action = arguments[0]
        if action not in ACTIONS:
            return error(f"expected exactly one of {', '.join(sorted(ACTIONS))}; got {action!r}.\n" + HELP)
        target = ROOT / "workflow" / "entry.py"
        forwarded = arguments

    if not target.is_file():
        return error(f"package file is missing: {target}")

    command = [str(runtime), str(target), *forwarded]
    try:
        completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    except OSError as exc:
        return error(f"could not start the selected Python executable: {exc}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
