"""Create an isolated Python 3.12 CPU environment for the source repository."""

from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENV_DIR = ROOT / ".venv"
ENV_PYTHON = ENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
PYPI_INDEX = "https://pypi.org/simple"


def fail(message: str, code: int = 1) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return code


def run(command: list[str], label: str) -> int:
    print(f"\n== {label} ==", flush=True)
    try:
        result = subprocess.run(command, cwd=ROOT, check=False)
    except OSError as exc:
        return fail(f"Could not start {label}: {exc}")
    if result.returncode != 0:
        return fail(f"{label} failed with exit code {result.returncode}.", result.returncode)
    return 0


def main() -> int:
    if sys.version_info[:2] != (3, 12):
        return fail(
            f"Run this script with Python 3.12; current interpreter is {sys.version.split()[0]}. "
            "Use a verified Python 3.12 interpreter: python setup_env.py"
        )

    requirements = ROOT / "requirements.txt"
    if not requirements.is_file():
        return fail(f"Pinned dependency file is missing: {requirements}")

    if ENV_DIR.exists():
        if not ENV_PYTHON.is_file():
            return fail(
                f"An incomplete environment already exists at {ENV_DIR}. "
                "Use a fresh extracted package folder; this setup will not delete it."
            )
        probe = subprocess.run(
            [str(ENV_PYTHON), "-c", "import sys; print('.'.join(map(str, sys.version_info[:2])))"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode != 0 or probe.stdout.strip() != "3.12":
            detail = probe.stderr.strip() or probe.stdout.strip() or f"exit code {probe.returncode}"
            return fail(f"Existing {ENV_DIR} is not a working Python 3.12 environment ({detail}).")
    else:
        print(f"Creating package-local virtual environment: {ENV_DIR}", flush=True)
        try:
            venv.EnvBuilder(with_pip=True).create(ENV_DIR)
        except Exception as exc:
            return fail(f"Could not create {ENV_DIR}: {exc}")
        if not ENV_PYTHON.is_file():
            return fail(f"Virtual environment creation did not produce {ENV_PYTHON}.")

    common = [str(ENV_PYTHON), "-m", "pip", "install", "--disable-pip-version-check", "--no-input"]
    code = run(
        [*common, "--index-url", TORCH_CPU_INDEX, "--extra-index-url", PYPI_INDEX, "torch==2.12.1+cpu"],
        "Install CPU PyTorch 2.12.1",
    )
    if code:
        return code
    code = run(
        [*common, "--index-url", PYPI_INDEX, "-r", str(requirements)],
        "Install pinned workflow dependencies",
    )
    if code:
        return code

    return run([str(ENV_PYTHON), str(ROOT / "health_check.py")], "Verify the package and environment")


if __name__ == "__main__":
    raise SystemExit(main())

