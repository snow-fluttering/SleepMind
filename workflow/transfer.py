"""Replay one EEG text input through the copied local BCI sender and receiver."""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import subprocess
import sys
import threading
from itertools import zip_longest
from pathlib import Path
from typing import Any

from common import VERSION_ROOT, child_environment, emit_result, error_details, next_action, sha256_file, utc_now, write_json


SENTINEL = object()


class SourceManifestError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def validate_source_manifest(input_path: Path) -> tuple[dict[str, Any], int, int, str]:
    manifest_path = input_path.parent / "source_manifest.json"
    if not manifest_path.is_file():
        raise SourceManifestError(
            "SOURCE_MANIFEST_MISSING",
            f"Add {manifest_path} beside the input. It must explicitly declare sample_count, sampling_rate_contract_hz=100, amplitude_unit_contract='V', amplitude_unit_auto_detected=false, processing_state='raw_unfiltered', labels_read=false, extracted_bytes, and sha256.",
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceManifestError("SOURCE_MANIFEST_INVALID", f"Cannot read {manifest_path} as UTF-8 JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise SourceManifestError("SOURCE_MANIFEST_INVALID", f"{manifest_path} must contain a JSON object.")

    expected = {
        "sampling_rate_contract_hz": 100,
        "amplitude_unit_contract": "V",
        "amplitude_unit_auto_detected": False,
        "processing_state": "raw_unfiltered",
        "labels_read": False,
    }
    missing = [key for key in ("sample_count", *expected, "extracted_bytes", "sha256") if key not in manifest]
    if missing:
        raise SourceManifestError(
            "SOURCE_MANIFEST_INCOMPLETE",
            f"{manifest_path} is missing required declarations: {', '.join(missing)}. Add explicit values before transfer; metadata is not inferred.",
        )
    mismatched = [key for key, value in expected.items()
                  if (manifest.get(key) is not value if isinstance(value, bool) else manifest.get(key) != value)]
    if mismatched:
        details = ", ".join(f"{key}={manifest.get(key)!r} (required {expected[key]!r})" for key in mismatched)
        raise SourceManifestError("SOURCE_CONTRACT_UNSUPPORTED", f"{manifest_path} does not match the supported raw 100 Hz/V contract: {details}.")
    if type(manifest.get("sample_count")) is not int or manifest["sample_count"] <= 0:
        raise SourceManifestError("SOURCE_MANIFEST_INVALID", f"{manifest_path}: sample_count must be a positive integer.")
    if type(manifest.get("extracted_bytes")) is not int or manifest["extracted_bytes"] <= 0:
        raise SourceManifestError("SOURCE_MANIFEST_INVALID", f"{manifest_path}: extracted_bytes must be a positive integer.")
    expected_sha = manifest.get("sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64 or any(character not in "0123456789abcdefABCDEF" for character in expected_sha):
        raise SourceManifestError("SOURCE_MANIFEST_INVALID", f"{manifest_path}: sha256 must contain 64 hexadecimal characters for this exact input file.")

    size = input_path.stat().st_size
    if size != manifest["extracted_bytes"]:
        raise SourceManifestError("SOURCE_MANIFEST_FILE_MISMATCH", f"Input is {size} bytes but {manifest_path} declares {manifest['extracted_bytes']} extracted bytes. Provide the manifest for this exact file.")
    digest = hashlib.sha256()
    with input_path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    actual_sha = digest.hexdigest()
    if actual_sha.lower() != expected_sha.lower():
        raise SourceManifestError("SOURCE_MANIFEST_FILE_MISMATCH", f"Input SHA-256 does not match {manifest_path}; provide the manifest for this exact file.")

    replay_dir = VERSION_ROOT / "bci_replay"
    if str(replay_dir) not in sys.path:
        sys.path.insert(0, str(replay_dir))
    from shared import scan_source

    chunks, samples = scan_source(input_path, 100)
    if samples != manifest["sample_count"]:
        raise SourceManifestError("SOURCE_SAMPLE_COUNT_MISMATCH", f"Input contains {samples} numeric samples but {manifest_path} declares {manifest['sample_count']}.")
    return manifest, chunks, samples, actual_sha


def emit_transfer(record: dict[str, Any], result_path: Path | None, following: dict[str, Any] | None = None) -> None:
    artifacts: dict[str, Path] = {}
    if result_path is not None:
        for name, key in (("receiver_report", "receiver_report"), ("received_eeg", "received_eeg"), ("sender_result", "sender_result")):
            value = record.get(key)
            if isinstance(value, str) and Path(value).is_file():
                artifacts[name] = Path(value)
    if record.get("status") == "success":
        summary = f"Transferred {record.get('sample_count')} samples; sender/receiver exit codes are 0 and all numeric tokens match."
    elif record.get("error_code") == "CHUNK_ORDER":
        summary = "Receiver detected CHUNK_ORDER; transfer stopped before staging."
    else:
        summary = record.get("message") or "Transfer stopped."
    emit_result("transfer", result_path, artifacts, summary, error_details(record), following)


def read_pipe(stream: Any, lines: list[str], ready: queue.Queue[str | object] | None = None) -> None:
    try:
        for line in stream:
            lines.append(line)
            if ready is not None:
                ready.put(line)
    finally:
        if ready is not None:
            ready.put(SENTINEL)


def json_lines(lines: list[str]) -> list[dict[str, Any]]:
    values = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            values.append(value)
    return values


def token_chunks(path: Path, chunk_size: int):
    replay_dir = VERSION_ROOT / "bci_replay"
    if str(replay_dir) not in sys.path:
        sys.path.insert(0, str(replay_dir))
    from shared import iter_chunks

    yield from iter_chunks(path, chunk_size)


def compare_tokens(source: Path, received: Path, chunk_size: int) -> tuple[bool, int]:
    compared = 0
    sentinel = object()
    for left, right in zip_longest(token_chunks(source, chunk_size), token_chunks(received, chunk_size), fillvalue=sentinel):
        if left is sentinel or right is sentinel or left != right:
            return False, compared
        compared += len(left)
    return True, compared


def terminate_receiver(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="One whitespace- or comma-separated EEG text file")
    parser.add_argument("--output-dir", required=True, help="A new transfer output directory")
    parser.add_argument("--drop-chunk", type=int, help="Failure demonstration: drop this zero-based sender chunk")
    parser.add_argument("--run-directory", help=argparse.SUPPRESS)
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    record: dict[str, Any] = {
        "status": "error",
        "input": str(input_path),
        "output_dir": str(output_dir),
        "run_directory": str(Path(args.run_directory).expanduser().resolve()) if args.run_directory else None,
        "sampling_rate_hz": 100,
        "unit": "V",
        "metadata_source": "declared by this workflow; not automatically detected",
        "processing_state": "raw_unfiltered",
        "sender_returncode": None,
        "receiver_returncode": None,
        "sender_status": None,
        "receiver_status": None,
        "sender_samples": None,
        "receiver_expected_samples": None,
        "receiver_received_samples": None,
        "compared_tokens": 0,
        "tokens_identical": False,
        "error_code": None,
        "message": None,
    }
    source_manifest_path = input_path.parent / "source_manifest.json"
    record["source_manifest_path"] = str(source_manifest_path)
    result_path = output_dir / "transfer_result.json"

    if output_dir.exists():
        record.update(error_code="OUTPUT_DIR_EXISTS", root_cause="Output directory already exists; refusing to overwrite.", message="Output directory already exists; refusing to overwrite.")
        emit_transfer(record, None)
        return 2
    output_created = False
    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        if not input_path.is_file():
            raise FileNotFoundError(f"Input file does not exist: {input_path}")
        output_dir.mkdir(parents=False, exist_ok=False)
        output_created = True
    except Exception as exc:
        record.update(error_code="TRANSFER_SETUP_ERROR", root_cause=str(exc), message=str(exc))
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=False)
                output_created = True
            except OSError:
                pass
        if output_created:
            write_json(output_dir / "transfer_result.json", record)
        emit_transfer(record, result_path if output_created else None)
        return 2

    try:
        source_manifest, source_chunks, source_samples, source_sha256 = validate_source_manifest(input_path)
        record.update(
            source_manifest=source_manifest,
            source_manifest_validation={
                "status": "pass",
                "path": str(source_manifest_path),
                "sha256_verified": source_sha256,
                "numeric_sample_count": source_samples,
                "chunk_count_at_100_samples": source_chunks,
            },
        )
    except SourceManifestError as exc:
        record.update(error_code=exc.code, root_cause=str(exc), message=str(exc), completed_utc=utc_now())
        write_json(result_path, record)
        emit_transfer(record, result_path)
        return 2
    except Exception as exc:
        record.update(error_code="SOURCE_PREFLIGHT_FAILED", root_cause=str(exc), message=str(exc), completed_utc=utc_now())
        write_json(result_path, record)
        emit_transfer(record, result_path)
        return 2

    # The receiver owns its output directory; use a child directory so transfer logs
    # and the final workflow record can be saved even if receiver startup fails.
    receiver_dir = output_dir / "receiver"
    receiver_script = VERSION_ROOT / "bci_replay" / "receive_eeg.py"
    sender_script = VERSION_ROOT / "bci_replay" / "send_eeg.py"
    environment = child_environment()
    receiver_command = [
        sys.executable, str(receiver_script), "--output-dir", str(receiver_dir),
        "--port", "0", "--timeout", "300",
    ]
    sender_command: list[str] = []
    receiver_process: subprocess.Popen[str] | None = None
    receiver_stdout: list[str] = []
    receiver_stderr: list[str] = []
    ready_queue: queue.Queue[str | object] = queue.Queue()
    receiver_threads: list[threading.Thread] = []
    sender_stdout = ""
    sender_stderr = ""
    sender_payload: dict[str, Any] | None = None
    failure_code: str | None = None
    failure_message: str | None = None

    try:
        receiver_process = subprocess.Popen(
            receiver_command,
            cwd=str(VERSION_ROOT / "bci_replay"),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert receiver_process.stdout is not None and receiver_process.stderr is not None
        stdout_thread = threading.Thread(target=read_pipe, args=(receiver_process.stdout, receiver_stdout, ready_queue), daemon=True)
        stderr_thread = threading.Thread(target=read_pipe, args=(receiver_process.stderr, receiver_stderr), daemon=True)
        receiver_threads.extend((stdout_thread, stderr_thread))
        stdout_thread.start()
        stderr_thread.start()

        try:
            ready_line = ready_queue.get(timeout=15)
        except queue.Empty:
            raise RuntimeError("Receiver did not emit a ready line within 15 seconds.")
        if ready_line is SENTINEL:
            raise RuntimeError("Receiver exited before emitting a ready line.")
        try:
            ready_payload = json.loads(str(ready_line))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Receiver ready line was not valid JSON.") from exc
        if not isinstance(ready_payload, dict) or ready_payload.get("status") != "ready":
            raise RuntimeError(f"Receiver did not become ready: {ready_payload}")
        port = ready_payload.get("port")
        if type(port) is not int or not 1 <= port <= 65535:
            raise RuntimeError("Receiver ready JSON did not contain a valid assigned port.")

        sender_command = [
            sys.executable, str(sender_script), str(input_path), "--port", str(port),
            "--chunk-size", "100", "--fs", "100", "--unit", "V", "--speed", "0", "--timeout", "30",
        ]
        if args.drop_chunk is not None:
            sender_command.extend(("--drop-chunk", str(args.drop_chunk)))
        try:
            sender_result = subprocess.run(
                sender_command,
                cwd=str(VERSION_ROOT / "bci_replay"),
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=300,
                check=False,
            )
            sender_stdout, sender_stderr = sender_result.stdout, sender_result.stderr
            record["sender_returncode"] = sender_result.returncode
            parsed_sender = json_lines(sender_stdout.splitlines())
            if parsed_sender:
                sender_payload = parsed_sender[-1]
                record["sender_status"] = sender_payload.get("status")
                record["sender_samples"] = sender_payload.get("total_samples")
                record["sender_error_code"] = sender_payload.get("error_code")
                record["sender_message"] = sender_payload.get("message")
            if sender_result.returncode != 0:
                failure_code = "SENDER_FAILED"
                failure_message = f"Sender returned {sender_result.returncode}."
        except subprocess.TimeoutExpired as exc:
            sender_stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            sender_stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            record["sender_returncode"] = 124
            failure_code = "SENDER_TIMEOUT"
            failure_message = "Sender exceeded the 300 second timeout."
    except Exception as exc:
        failure_code = "TRANSFER_PROCESS_ERROR"
        failure_message = str(exc)
    finally:
        if receiver_process is not None:
            if failure_code is not None:
                if record.get("sender_returncode") not in (None, 0):
                    try:
                        receiver_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        terminate_receiver(receiver_process)
                else:
                    terminate_receiver(receiver_process)
            else:
                try:
                    receiver_process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    terminate_receiver(receiver_process)
                    failure_code = failure_code or "RECEIVER_TIMEOUT"
                    failure_message = failure_message or "Receiver did not exit within 30 seconds after sender completion."
            record["receiver_returncode"] = receiver_process.poll()
            for thread in receiver_threads:
                thread.join(timeout=5)
            if any(thread.is_alive() for thread in receiver_threads):
                terminate_receiver(receiver_process)
                for thread in receiver_threads:
                    thread.join(timeout=2)

    (output_dir / "receiver_stdout.txt").write_text("".join(receiver_stdout), encoding="utf-8")
    (output_dir / "receiver_stderr.txt").write_text("".join(receiver_stderr), encoding="utf-8")
    (output_dir / "sender_stdout.txt").write_text(sender_stdout, encoding="utf-8")
    (output_dir / "sender_stderr.txt").write_text(sender_stderr, encoding="utf-8")
    if sender_payload is not None:
        write_json(output_dir / "sender_result.json", sender_payload)

    receiver_report_path = receiver_dir / "report.json"
    receiver_payload: dict[str, Any] | None = None
    if receiver_report_path.is_file():
        try:
            parsed_report = json.loads(receiver_report_path.read_text(encoding="utf-8"))
            if isinstance(parsed_report, dict):
                receiver_payload = parsed_report
        except (OSError, json.JSONDecodeError):
            pass
    if receiver_payload is not None:
        record["receiver_status"] = receiver_payload.get("status")
        record["receiver_expected_samples"] = receiver_payload.get("expected_total_samples")
        record["receiver_received_samples"] = receiver_payload.get("received_samples")
    if receiver_payload is not None and receiver_payload.get("error_code") == "CHUNK_ORDER":
        receiver_message = receiver_payload.get("message") or "Receiver rejected a chunk sequence."
        record["root_cause"] = f"Receiver CHUNK_ORDER: {receiver_message}"
        secondary_errors = []
        if sender_payload is not None and sender_payload.get("status") == "error":
            secondary_errors.append({
                "source": "sender",
                "code": sender_payload.get("error_code"),
                "message": sender_payload.get("message"),
            })
        elif record.get("sender_returncode") not in (None, 0):
            secondary_errors.append({"source": "sender", "code": "SENDER_FAILED", "message": f"Sender returned {record['sender_returncode']}."})
        record["secondary_errors"] = secondary_errors
        failure_code = "CHUNK_ORDER"
        failure_message = record["root_cause"]
    if record["receiver_returncode"] != 0 or receiver_payload is None or receiver_payload.get("status") != "success":
        failure_code = failure_code or "RECEIVER_FAILED"
        failure_message = failure_message or "Receiver did not report a successful complete transfer."
    if sender_payload is None or sender_payload.get("status") != "sent":
        failure_code = failure_code or "SENDER_RESULT_INVALID"
        failure_message = failure_message or "Sender output did not contain a sent result JSON object."

    if failure_code is None and receiver_payload is not None and sender_payload is not None:
        try:
            expected_chunks = receiver_payload["expected_total_chunks"]
            received_chunks = receiver_payload["received_chunks"]
            sender_chunks = sender_payload["total_chunks"]
            if not (
                receiver_payload.get("fs") == 100
                and receiver_payload.get("unit") == "V"
                and isinstance(receiver_payload.get("metadata_source"), str)
                and bool(receiver_payload.get("metadata_source"))
                and receiver_payload.get("expected_total_samples") == receiver_payload.get("received_samples")
                and sender_payload.get("total_samples") == receiver_payload.get("received_samples")
                and sender_payload.get("sent_chunks") == sender_chunks == expected_chunks == received_chunks
                and sender_payload.get("dropped_chunks") == 0
            ):
                raise ValueError("Sender and receiver metadata or totals do not match the 100 Hz/V contract.")
            identical, compared = compare_tokens(input_path, receiver_dir / "received_eeg.txt", 100)
            record["compared_tokens"] = compared
            record["tokens_identical"] = identical
            if not identical or compared != receiver_payload["received_samples"]:
                raise ValueError("Source and received numeric tokens differ.")
            if sha256_file(input_path) != source_sha256:
                raise ValueError("The source file changed after source-manifest validation.")
            record["received_eeg_sha256"] = sha256_file(receiver_dir / "received_eeg.txt")
        except Exception as exc:
            failure_code = "TRANSFER_VALIDATION_FAILED"
            failure_message = str(exc)

    record["receiver_report"] = str(receiver_report_path) if receiver_report_path.is_file() else None
    record["sender_result"] = str(output_dir / "sender_result.json") if sender_payload is not None else None
    record["received_eeg"] = str(receiver_dir / "received_eeg.txt") if (receiver_dir / "received_eeg.txt").is_file() else None
    record["sender_command"] = sender_command
    record["receiver_command"] = receiver_command
    record["completed_utc"] = utc_now()
    if failure_code is None:
        record.update(status="success", error_code=None, root_cause=None, secondary_errors=[], message="Transfer complete; sender/receiver totals and numeric tokens match.")
        record["sample_count"] = record["receiver_received_samples"]
    else:
        record.update(status="error", error_code=failure_code, root_cause=record.get("root_cause", failure_message), message=failure_message)
        if (receiver_dir / "received_eeg.txt").exists() and (receiver_payload is None or receiver_payload.get("status") != "success"):
            record["error_code"] = "INCOMPLETE_RECEIVER_FILE"
            record["root_cause"] = "An incomplete receiver file unexpectedly exists."
            record["message"] = "An incomplete receiver file unexpectedly exists; do not stage this transfer."
    write_json(output_dir / "transfer_result.json", record)
    following = next_action("stage", ["--transfer-result", str(result_path.resolve())]) if record["status"] == "success" else None
    emit_transfer(record, result_path, following)
    return 0 if record["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
