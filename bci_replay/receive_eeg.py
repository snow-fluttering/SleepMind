"""Receive one local JSONL EEG replay and save it only after full validation."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
from pathlib import Path

from shared import ReplayError, read_frame, require_finite_positive, require_int, require_text, validate_number_token


def configure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def print_json(value: dict[str, object]) -> None:
    configure_streams()
    print(json.dumps(value, ensure_ascii=False, separators=(",", ":")), flush=True)


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print_json({"status": "error", "error_code": "INVALID_ARGUMENT", "message": "Invalid arguments; check arguments and the output directory."})
        raise SystemExit(2)


def parse_message(line: bytes) -> dict[str, object]:
    if not line:
        raise ReplayError("MISSING_END", "The connection closed before the end frame.")
    try:
        value = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayError("INVALID_JSON", "The frame is not valid UTF-8 JSONL.") from exc
    if not isinstance(value, dict):
        raise ReplayError("INVALID_PROTOCOL", "Every frame must be a JSON object.")
    return value


def write_report(directory: Path, report: dict[str, object]) -> None:
    temporary = directory / "report.json.tmp"
    target = directory / "report.json"
    with open(temporary, "x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, target)


def main() -> int:
    configure_streams()
    parser = JsonArgumentParser(description="Receive local TCP EEG replay and validate JSONL.")
    parser.add_argument("--output-dir", required=True, help="New output directory; refuse if it already exists.")
    parser.add_argument("--port", type=int, default=8765, help="Listening port; 0 selects a system-assigned port (default: 8765).")
    parser.add_argument("--timeout", type=float, default=30.0, help="Connection/frame timeout in seconds (default: 30).")
    args = parser.parse_args()

    if not 0 <= args.port <= 65535 or not math.isfinite(args.timeout) or args.timeout <= 0:
        print_json({"status": "error", "error_code": "INVALID_ARGUMENT", "message": "port must be between 0 and 65535; timeout must be finite and positive."})
        return 2

    output_dir = Path(args.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print_json({"status": "error", "error_code": "OUTPUT_DIR_EXISTS", "message": f"Output directory already exists; refusing to overwrite: {output_dir}"})
        return 2
    except OSError as exc:
        print_json({"status": "error", "error_code": "OUTPUT_DIR_ERROR", "message": f"Cannot create new output directory: {exc}"})
        return 2

    report: dict[str, object] = {
        "status": "error",
        "error_code": None,
        "message": "Reception incomplete.",
        "session": None,
        "fs": None,
        "unit": None,
        "metadata_source": "Declared by the sender; not automatically detected.",
        "received_chunks": 0,
        "received_samples": 0,
        "expected_total_chunks": None,
        "expected_total_samples": None,
    }
    temporary_data = output_dir / ".received_eeg.txt.part"
    final_data = output_dir / "received_eeg.txt"
    finalized_data = False
    listener: socket.socket | None = None

    try:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", args.port))
        listener.listen(1)
        listener.settimeout(args.timeout)
        bound_port = listener.getsockname()[1]
        print_json({
            "status": "ready",
            "host": "127.0.0.1",
            "port": bound_port,
            "message": "Receiver ready; waiting for a local sender.",
        })

        connection, _address = listener.accept()
        connection.settimeout(args.timeout)
        with connection:
            with connection.makefile("rb") as reader, open(temporary_data, "x", encoding="ascii", newline="\n") as data_file:
                first_line = read_frame(reader)
                if not first_line:
                    raise ReplayError("MISSING_META", "No metadata frame was received before disconnection.")
                meta = parse_message(first_line)
                if meta.get("type") != "meta":
                    raise ReplayError("PROTOCOL_ORDER", "The first frame must be a meta frame.")
                session = require_text(meta.get("session"), "session")
                fs = require_finite_positive(meta.get("fs"), "fs")
                unit = require_text(meta.get("unit"), "unit")
                report["session"] = session
                report["fs"] = fs
                report["unit"] = unit
                report["message"] = f"Received declared metadata: fs={fs:g} Hz, unit={unit}; values are declared by the sender, not automatically detected."

                received_chunks = 0
                received_samples = 0
                while True:
                    line = read_frame(reader)
                    if not line:
                        raise ReplayError("MISSING_END", "The connection closed before the end frame; temporary data were discarded.")
                    message = parse_message(line)
                    kind = message.get("type")
                    if kind not in ("chunk", "end"):
                        raise ReplayError("INVALID_PROTOCOL", "Only chunk or end frames may follow meta.")
                    if message.get("session") != session:
                        raise ReplayError("SESSION_MISMATCH", "Frame session differs from the meta session.")

                    if kind == "chunk":
                        seq = require_int(message.get("seq"), "seq")
                        start_sample = require_int(message.get("start_sample"), "start_sample")
                        if seq != received_chunks:
                            raise ReplayError("CHUNK_ORDER", f"Nonconsecutive chunk sequence: expected {received_chunks}, received {seq}.")
                        if start_sample != received_samples:
                            raise ReplayError("CHUNK_START", f"Nonconsecutive sample start: expected {received_samples}, received {start_sample}.")
                        samples = message.get("samples")
                        if not isinstance(samples, list) or not samples:
                            raise ReplayError("INVALID_PROTOCOL", "chunk.samples must be a nonempty array.")
                        preserved: list[str] = []
                        for token in samples:
                            try:
                                preserved.append(validate_number_token(token))
                            except ValueError as exc:
                                raise ReplayError("INVALID_SAMPLE", f"Chunk {seq} contains an invalid sample: {exc}") from exc
                        data_file.writelines(token + "\n" for token in preserved)
                        received_chunks += 1
                        received_samples += len(preserved)
                        report["received_chunks"] = received_chunks
                        report["received_samples"] = received_samples
                        if received_chunks % 1000 == 0:
                            configure_streams()
                            print(f"Received {received_chunks} chunks and {received_samples} samples.", file=sys.stderr, flush=True)
                        continue

                    total_chunks = require_int(message.get("total_chunks"), "total_chunks")
                    total_samples = require_int(message.get("total_samples"), "total_samples")
                    report["expected_total_chunks"] = total_chunks
                    report["expected_total_samples"] = total_samples
                    if total_chunks == 0 or total_samples == 0:
                        raise ReplayError("EMPTY_INPUT", "The end frame declares an empty replay; refusing to create an EEG file.")
                    if total_chunks != received_chunks or total_samples != received_samples:
                        raise ReplayError(
                            "TOTAL_MISMATCH",
                            f"End-frame totals mismatch: declared {total_chunks} chunks/{total_samples} samples; received {received_chunks} chunks/{received_samples} samples.",
                        )
                    trailing = reader.read(1)
                    if trailing:
                        raise ReplayError("TRAILING_DATA", "Extra data follow the end frame.")
                    break

                data_file.flush()

        if final_data.exists():
            raise ReplayError("OUTPUT_CONFLICT", "Final output already exists; refusing to overwrite.")
        temporary_data.rename(final_data)
        finalized_data = True
        report.update({
            "status": "success",
            "error_code": None,
            "message": f"Reception complete: {report['received_chunks']} chunks and {report['received_samples']} samples; fs={report['fs']} Hz, unit={report['unit']} (sender declarations; not automatically detected).",
        })
        write_report(output_dir, report)
        print_json(report)
        return 0
    except ReplayError as exc:
        report["error_code"] = exc.code
        report["message"] = exc.message
    except socket.timeout:
        report["error_code"] = "TIMEOUT"
        report["message"] = f"Waiting for a connection or frame exceeded {args.timeout:g} seconds; temporary data were discarded."
    except OSError as exc:
        report["error_code"] = "RECEIVER_IO_ERROR"
        report["message"] = f"Reception failed: {exc}; temporary data were discarded."
    except Exception as exc:  # Keep a structured failure report for unexpected local errors.
        report["error_code"] = "INTERNAL_ERROR"
        report["message"] = f"Internal receiver error: {exc}; temporary data were discarded."
    finally:
        if listener is not None:
            listener.close()

    report["status"] = "error"
    try:
        temporary_data.unlink(missing_ok=True)
    except OSError:
        pass
    if finalized_data:
        try:
            final_data.unlink(missing_ok=True)
        except OSError:
            pass
    try:
        write_report(output_dir, report)
    except OSError as exc:
        report["report_write_error"] = str(exc)
    print_json(report)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
