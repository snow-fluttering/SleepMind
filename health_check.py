"""Check source and CPU dependencies, optionally requiring private trained assets."""
from __future__ import annotations
import argparse
import importlib
import importlib.metadata
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED = {
    "numpy": ("numpy", "2.5.0"),
    "scipy": ("scipy", "1.18.0"),
    "sklearn": ("scikit-learn", "1.9.0"),
    "matplotlib": ("matplotlib", "3.11.0"),
}
CORE_MODULES = (
    "config", "data_preprocessing", "dataset", "decoding", "evaluation", "main",
    "model", "predict", "readers", "reporting", "splits", "sysinfo", "train", "utils",
)
REQUIRED_FILES = [f"src/{name}.py" for name in CORE_MODULES] + [
    "run_action.py", "setup_env.py", "requirements.txt",
    "workflow/common.py", "workflow/entry.py", "workflow/stage.py",
    "workflow/transfer.py", "workflow/verify.py",
    "bci_replay/shared.py", "bci_replay/send_eeg.py", "bci_replay/receive_eeg.py",
]

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", nargs="?", const="", help="Also require exact private trained assets; omit PATH to use SLEEPMIND_ASSET_ROOT")
    args = parser.parse_args()
    problems = []
    print(f"Python: {sys.executable}")
    print(f"Python version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print(f"Repository root: {ROOT}")
    if sys.version_info[:2] != (3, 12):
        problems.append("Python 3.12 is required for the pinned CPU environment.")
    for relative in REQUIRED_FILES:
        if not (ROOT / relative).is_file():
            problems.append(f"Missing source file: {relative}")
    if not any(problem.startswith("Missing source file:") for problem in problems):
        print(f"Required source files: present ({len(REQUIRED_FILES)} checks)")
    for module_name, (distribution, expected_version) in EXPECTED.items():
        try:
            importlib.import_module(module_name)
            actual = importlib.metadata.version(distribution)
            print(f"{distribution}: {actual}")
            if actual != expected_version:
                problems.append(f"Expected {distribution}=={expected_version}, found {actual}.")
        except Exception as exc:
            problems.append(f"Could not import {module_name}: {exc}")
    try:
        torch = importlib.import_module("torch")
        actual = importlib.metadata.version("torch")
        cuda_build = getattr(torch.version, "cuda", None)
        print(f"torch: {actual}")
        print(f"torch CUDA build: {cuda_build or 'none (CPU)'}")
        if actual.split("+", 1)[0] != "2.12.1" or cuda_build is not None:
            problems.append("The pinned environment requires CPU torch 2.12.1.")
    except Exception as exc:
        problems.append(f"Could not import torch: {exc}")
    if args.assets is not None:
        try:
            sys.path.insert(0, str(ROOT / "workflow"))
            from common import model_integrity
            integrity = model_integrity(args.assets or None)
            print(f"Private trained assets: PASS ({len(integrity['asset_file_sha256'])} fixed files)")
            print(f"Checkpoint SHA-256: {integrity['checkpoint_sha256']}")
        except Exception as exc:
            problems.append(f"Private trained assets are required for this check: {exc}")
    else:
        print("Private trained assets: not requested. Prediction requires matched assets supplied separately via SLEEPMIND_ASSET_ROOT.")
    if problems:
        print("\nHealth check: FAILED", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1
    print("Health check: PASS")
    print("This check ran no transport, prediction, training, or accuracy evaluation.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
