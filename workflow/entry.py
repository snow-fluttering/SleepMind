"""Run one explicit transfer, stage, or verify action in the local research workflow."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import stage
import transfer
import verify
from common import VERSION_ROOT, emit_result, error_details


def read_record(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def invoke(module: Any, args: list[str]) -> int:
    previous = sys.argv
    sys.argv = [str(Path(module.__file__).resolve()), *args]
    try:
        return module.main()
    finally:
        sys.argv = previous


def stop(action: str, code: str, message: str) -> int:
    emit_result(action, None, {}, message, {"code": code, "root_cause": message, "secondary": []}, None)
    return 2


def unique_run_directory(run_root: Path) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    while True:
        candidate = run_root / f"run-{stamp}-{uuid.uuid4().hex[:8]}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)

    transfer_parser = actions.add_parser("transfer", help="Validate the raw source contract and replay one input.")
    transfer_parser.add_argument("--input", required=True)
    transfer_parser.add_argument("--run-root", default=str(VERSION_ROOT / "workflow_runs"))
    transfer_parser.add_argument("--drop-chunk", type=int)

    stage_parser = actions.add_parser("stage", help="Run SleepMind for one successful transfer.")
    stage_parser.add_argument("--transfer-result", required=True)
    stage_parser.add_argument("--output-dir")

    verify_parser = actions.add_parser("verify", help="Verify one successful prediction and write its research summary.")
    verify_parser.add_argument("--stage-result", required=True)

    args = parser.parse_args()
    if args.action == "transfer":
        try:
            run_directory = unique_run_directory(Path(args.run_root).expanduser().resolve())
        except OSError as exc:
            return stop("transfer", "RUN_DIRECTORY_ERROR", f"Cannot create a unique run directory under {args.run_root}: {exc}")
        component_args = ["--input", str(Path(args.input).expanduser().resolve()), "--output-dir", str(run_directory / "transfer"), "--run-directory", str(run_directory)]
        if args.drop_chunk is not None:
            component_args.extend(("--drop-chunk", str(args.drop_chunk)))
        return invoke(transfer, component_args)

    if args.action == "stage":
        transfer_result_path = Path(args.transfer_result).expanduser().resolve()
        try:
            transfer_record = read_record(transfer_result_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            return stop("stage", "TRANSFER_RESULT_UNREADABLE", f"Cannot read the transfer result {transfer_result_path}: {exc}")
        if transfer_record.get("status") != "success":
            return stop("stage", "TRANSFER_NOT_SUCCESSFUL", "The supplied transfer record is not successful; stop before SleepMind analysis.")
        transfer_dir = transfer_result_path.parent
        if args.output_dir:
            output_dir = Path(args.output_dir).expanduser().resolve()
        else:
            run_directory = transfer_record.get("run_directory")
            if isinstance(run_directory, str) and run_directory:
                output_dir = Path(run_directory).expanduser().resolve() / "prediction"
            else:
                output_dir = transfer_dir.parent / f"{transfer_dir.name}_prediction"
        return invoke(stage, ["--transfer-dir", str(transfer_dir), "--output-dir", str(output_dir)])

    stage_result_path = Path(args.stage_result).expanduser().resolve()
    try:
        stage_record = read_record(stage_result_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return stop("verify", "STAGE_RESULT_UNREADABLE", f"Cannot read the stage result {stage_result_path}: {exc}")
    if stage_record.get("status") != "success" or stage_record.get("predictor_returncode") != 0:
        return stop("verify", "STAGE_NOT_SUCCESSFUL", "The supplied stage record is not successful; stop before verification.")
    return invoke(verify, ["--prediction-dir", str(stage_result_path.parent)])


if __name__ == "__main__":
    raise SystemExit(main())
