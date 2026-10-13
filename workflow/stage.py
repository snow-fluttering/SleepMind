"""Validate one complete transfer and run the frozen raw-input SleepMind protocol."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import VERSION_ROOT, child_environment, emit_result, error_details, asset_root, executing_source_hashes, model_integrity, next_action, require_received_integrity, utc_now, write_json


PREDICTION_FILES = (
    "bci_recording_epochs.npz",
    "bci_recording_hypnogram.tsv",
    "bci_recording_report.json",
    "bci_recording_report.html",
    "bci_recording_hypnogram.png",
)


def remove_prediction_files(output_dir: Path) -> None:
    for name in PREDICTION_FILES:
        (output_dir / name).unlink(missing_ok=True)


def emit_stage(result: dict[str, Any], result_path: Path | None, following: dict[str, Any] | None = None) -> None:
    output_dir = Path(result["output_dir"])
    artifacts = {}
    if result_path is not None:
        for name, filename in (
            ("prediction_report", "bci_recording_report.json"),
            ("epochs", "bci_recording_epochs.npz"),
            ("hypnogram", "bci_recording_hypnogram.tsv"),
            ("html_report", "bci_recording_report.html"),
            ("hypnogram_plot", "bci_recording_hypnogram.png"),
            ("stdout", "stdout.txt"),
            ("stderr", "stderr.txt"),
        ):
            path = output_dir / filename
            if path.is_file():
                artifacts[name] = path
    if result.get("status") == "success":
        metrics = result.get("prediction_summary", {})
        summary = (
            f"SleepMind returned 0; {result.get('sample_count')} raw samples, "
            f"{metrics.get('recording_seconds')} seconds total, {metrics.get('scorable_seconds')} seconds scored, "
            f"{metrics.get('unscorable_seconds')} seconds Unknown."
        )
    else:
        summary = result.get("message") or "Prediction stopped."
    emit_result("stage", result_path, artifacts, summary, error_details(result), following)


def fail_record(result: dict[str, Any], output_dir: Path, code: str, message: str, returncode: int = 2) -> int:
    result.update(status="error", error_code=code, root_cause=message, secondary_errors=[], message=message, completed_utc=utc_now())
    result_path = output_dir / "stage_result.json"
    write_json(result_path, result)
    emit_stage(result, result_path)
    return returncode if returncode != 0 else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer-dir", required=True, help="Successful output directory from transfer.py")
    parser.add_argument("--output-dir", required=True, help="A new prediction output directory")
    args = parser.parse_args()

    transfer_dir = Path(args.transfer_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    result: dict[str, Any] = {
        "status": "error",
        "transfer_dir": str(transfer_dir),
        "output_dir": str(output_dir),
        "predictor_invoked": False,
        "predictor_returncode": None,
        "error_code": None,
        "message": None,
    }
    result_path = output_dir / "stage_result.json"
    if output_dir.exists():
        result.update(error_code="OUTPUT_DIR_EXISTS", root_cause="Output directory already exists; refusing to overwrite.", message="Output directory already exists; refusing to overwrite.")
        emit_stage(result, None)
        return 2
    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=False, exist_ok=False)
    except Exception as exc:
        result.update(error_code="OUTPUT_DIR_ERROR", root_cause=str(exc), message=str(exc))
        emit_stage(result, None)
        return 2

    report_path = transfer_dir / "receiver" / "report.json"
    sample_path = transfer_dir / "receiver" / "received_eeg.txt"
    transfer_result_path = transfer_dir / "transfer_result.json"
    result["receiver_report"] = str(report_path)
    result["received_eeg"] = str(sample_path)
    try:
        transfer_record = json.loads(transfer_result_path.read_text(encoding="utf-8"))
        receiver = json.loads(report_path.read_text(encoding="utf-8"))
        if transfer_record.get("status") != "success":
            raise ValueError("Transfer record is not successful.")
        if transfer_record.get("sender_returncode") != 0 or transfer_record.get("receiver_returncode") != 0:
            raise ValueError("Sender and receiver return codes must both be zero.")
        if transfer_record.get("sender_status") != "sent" or transfer_record.get("receiver_status") != "success":
            raise ValueError("Sender and receiver status fields must both indicate success.")
        if receiver.get("status") != "success":
            raise ValueError("Receiver report is not successful.")
        if not sample_path.is_file():
            raise ValueError("Complete received_eeg.txt is missing.")
        if receiver.get("fs") != 100 or receiver.get("unit") != "V":
            raise ValueError("Receiver metadata does not match declared 100 Hz/V contract.")
        expected = receiver.get("expected_total_samples")
        received = receiver.get("received_samples")
        chunks = receiver.get("received_chunks")
        if type(expected) is not int or expected <= 0 or expected != received:
            raise ValueError("Receiver expected and received sample totals do not match.")
        if type(chunks) is not int or chunks <= 0 or receiver.get("expected_total_chunks") != chunks:
            raise ValueError("Receiver expected and received chunk totals do not match.")
        if transfer_record.get("sample_count") != received or transfer_record.get("tokens_identical") is not True:
            raise ValueError("Workflow transfer totals or token equality did not pass.")
        if transfer_record.get("sender_samples") != received or transfer_record.get("receiver_expected_samples") != received or transfer_record.get("receiver_received_samples") != received:
            raise ValueError("Transfer sender, receiver expected, and receiver received sample counts do not match.")
        source_manifest = transfer_record.get("source_manifest")
        if not isinstance(source_manifest, dict):
            raise ValueError("Transfer does not include a source manifest declaring the raw input contract.")
        if (
            source_manifest.get("processing_state") != "raw_unfiltered"
            or source_manifest.get("sampling_rate_contract_hz") != 100
            or source_manifest.get("amplitude_unit_contract") != "V"
            or source_manifest.get("amplitude_unit_auto_detected") is not False
            or source_manifest.get("labels_read") is not False
            or source_manifest.get("sample_count") != received
        ):
            raise ValueError("Source manifest does not declare this complete raw 100 Hz/V sample input.")

        replay_dir = VERSION_ROOT / "bci_replay"
        if str(replay_dir) not in sys.path:
            sys.path.insert(0, str(replay_dir))
        from shared import scan_source

        actual_chunks, actual_samples = scan_source(sample_path, 100)
        if actual_samples != received or actual_chunks != chunks:
            raise ValueError("Received EEG file token count differs from receiver totals.")
        received_hash = require_received_integrity(transfer_record, sample_path)
        result.update(
            sample_count=received,
            sample_rate_hz=100,
            unit="V",
            processing_state="raw_unfiltered",
            source_manifest_path=transfer_record.get("source_manifest_path"),
            source_manifest=transfer_record.get("source_manifest"),
            received_eeg_sha256=received_hash,
        )
    except Exception as exc:
        return fail_record(result, output_dir, "TRANSFER_PREFLIGHT_FAILED", str(exc))

    try:
        result["model_integrity"] = model_integrity()
        result["executing_source_sha256"] = executing_source_hashes()
    except Exception as exc:
        return fail_record(result, output_dir, "MODEL_INTEGRITY_FAILED", str(exc))

    sleepmind_dir = VERSION_ROOT
    private_assets = asset_root()
    predict_script = sleepmind_dir / "src" / "predict.py"
    protocol_path = private_assets / "reports" / "ens_CE" / "frozen_protocol.json"
    command = [
        sys.executable, str(predict_script),
        "--source_file", str(sample_path),
        "--unit", "V",
        "--input_mode", "raw",
        "--run_name", str(private_assets / "runs" / "ens_CE"),
        "--protocol", str(protocol_path),
        "--threads", "4",
        "--output_dir", str(output_dir),
    ]
    result.update(
        command=command,
        working_directory=str(sleepmind_dir),
        predictor_invoked=False,
        workflow_python_executable=sys.executable,
        workflow_python_version=platform.python_version(),
        started_utc=utc_now(),
    )
    try:
        completed = subprocess.run(
            command,
            cwd=str(sleepmind_dir),
            env=child_environment(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1800,
            check=False,
        )
        result["predictor_invoked"] = True
    except subprocess.TimeoutExpired as exc:
        result["predictor_invoked"] = True
        (output_dir / "stdout.txt").write_text(exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""), encoding="utf-8")
        (output_dir / "stderr.txt").write_text(exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or ""), encoding="utf-8")
        remove_prediction_files(output_dir)
        result.update(predictor_returncode=124, completed_utc=utc_now())
        return fail_record(result, output_dir, "PREDICTOR_TIMEOUT", "SleepMind prediction exceeded the 1800 second timeout.", 124)
    except Exception as exc:
        return fail_record(result, output_dir, "PREDICTOR_LAUNCH_FAILED", str(exc))

    (output_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    result.update(predictor_returncode=completed.returncode, completed_utc=utc_now())
    if completed.returncode != 0:
        remove_prediction_files(output_dir)
        stderr_lines = [line.strip() for line in completed.stderr.splitlines() if line.strip()]
        cause = f"SleepMind returned {completed.returncode}."
        if stderr_lines:
            cause += f" Last stderr line: {stderr_lines[-1]}"
        return fail_record(result, output_dir, "PREDICTOR_FAILED", cause, completed.returncode)

    try:
        require_received_integrity(result, sample_path)
        if model_integrity() != result["model_integrity"]:
            raise ValueError("Reviewed model assets changed during prediction.")
        if executing_source_hashes() != result["executing_source_sha256"]:
            raise ValueError("Executing repository source changed during prediction.")
    except Exception as exc:
        remove_prediction_files(output_dir)
        return fail_record(result, output_dir, "PREDICTION_INTEGRITY_FAILED", str(exc))

    report = output_dir / "bci_recording_report.json"
    if not all((output_dir / name).is_file() for name in PREDICTION_FILES):
        remove_prediction_files(output_dir)
        return fail_record(result, output_dir, "PREDICTOR_OUTPUT_MISSING", "SleepMind returned 0 but did not create all expected prediction files.")
    try:
        prediction_report = json.loads(report.read_text(encoding="utf-8"))
        metadata = prediction_report.get("metadata", {})
        if (metadata.get("checkpoint_sha256") != result["model_integrity"]["checkpoint_sha256"]
                or metadata.get("protocol_sha256") != result["model_integrity"]["protocol_sha256"]):
            raise ValueError("Predictor report does not identify the reviewed checkpoint and protocol.")
        result["prediction_summary"] = {
            "recording_seconds": prediction_report.get("recording_seconds"),
            "scorable_seconds": prediction_report.get("scorable_seconds"),
            "unscorable_seconds": prediction_report.get("unscorable_seconds"),
            "stage_minutes": prediction_report.get("stage_minutes"),
        }
        result["predictor_runtime"] = {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "torch_version": prediction_report.get("metadata", {}).get("torch_version"),
            "device": prediction_report.get("metadata", {}).get("device"),
        }
    except (OSError, json.JSONDecodeError, AttributeError, ValueError) as exc:
        remove_prediction_files(output_dir)
        return fail_record(result, output_dir, "PREDICTOR_REPORT_INVALID", str(exc))
    result.update(status="success", error_code=None, root_cause=None, secondary_errors=[], message="Frozen raw-input prediction completed.")
    write_json(result_path, result)
    following = next_action("verify", ["--stage-result", str(result_path.resolve())])
    emit_stage(result, result_path, following)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
