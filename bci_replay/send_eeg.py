"""Stream an EEG text file to a local receive_eeg.py process."""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
import uuid
from pathlib import Path

from shared import MAX_FRAME_BYTES, ReplayError, encode_frame, iter_chunks, scan_source


def configure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def emit_result(result: dict[str, object]) -> None:
    configure_streams()
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")), flush=True)


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        emit_result({"status": "error", "error_code": "INVALID_ARGUMENT", "message": "Invalid arguments; check arguments and the input path."})
        raise SystemExit(2)


def main() -> int:
    configure_streams()
    parser = JsonArgumentParser(description="Replay EEG text samples to a local TCP receiver.")
    parser.add_argument("input", help="Whitespace- or comma-separated numeric EEG text file.")
    parser.add_argument("--port", type=int, default=8765, help="Receiver port (default: 8765).")
    parser.add_argument("--chunk-size", type=int, default=100, help="Samples per frame (default: 100).")
    parser.add_argument("--fs", type=float, default=100.0, help="Declared sample rate in Hz (default: 100).")
    parser.add_argument("--unit", default="V", help="Declared amplitude unit (default: V).")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier; 0 sends as fast as possible (default: 1).")
    parser.add_argument("--drop-chunk", type=int, help="Fault injection: drop a specified zero-based chunk index.")
    parser.add_argument("--timeout", type=float, default=10.0, help="Connection/send timeout in seconds (default: 10).")
    args = parser.parse_args()

    try:
        if not 1 <= args.port <= 65535:
            raise ReplayError("INVALID_ARGUMENT", "port must be between 1 and 65535.")
        if not 1 <= args.chunk_size <= 100000:
            raise ReplayError("INVALID_ARGUMENT", "chunk-size must be between 1 and 100000.")
        if not math.isfinite(args.fs) or args.fs <= 0:
            raise ReplayError("INVALID_ARGUMENT", "fs must be finite and positive.")
        if not math.isfinite(args.speed) or args.speed < 0:
            raise ReplayError("INVALID_ARGUMENT", "speed must be finite and nonnegative.")
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ReplayError("INVALID_ARGUMENT", "timeout must be finite and positive.")
        if not isinstance(args.unit, str) or not args.unit.strip():
            raise ReplayError("INVALID_ARGUMENT", "unit must not be empty.")

        input_path = Path(args.input)
        if not input_path.is_file():
            raise ReplayError("INPUT_NOT_FOUND", f"Input file does not exist: {input_path}")

        total_chunks, total_samples = scan_source(input_path, args.chunk_size)
        if total_samples == 0:
            raise ReplayError("EMPTY_INPUT", "The input has no EEG samples; refusing empty replay.")
        if args.drop_chunk is not None and not 0 <= args.drop_chunk < total_chunks:
            raise ReplayError("DROP_CHUNK_RANGE", f"drop-chunk must be between 0 and {total_chunks - 1}.")

        session = uuid.uuid4().hex
        sent_chunks = 0
        dropped_chunks = 0
        sent_samples = 0
        next_progress = max(1, math.ceil(total_chunks / 10))
        progress_step = next_progress

        with socket.create_connection(("127.0.0.1", args.port), timeout=args.timeout) as connection:
            connection.settimeout(args.timeout)

            def send_frame(frame: dict[str, object]) -> None:
                data = encode_frame(frame)
                if len(data) > MAX_FRAME_BYTES:
                    raise ReplayError("FRAME_TOO_LARGE", "Generated frame exceeds 4 MiB; reduce chunk-size.")
                connection.sendall(data)

            send_frame({"type": "meta", "session": session, "fs": args.fs, "unit": args.unit.strip()})
            for seq, samples in enumerate(iter_chunks(input_path, args.chunk_size)):
                start_sample = seq * args.chunk_size
                sent_samples += len(samples)
                if seq == args.drop_chunk:
                    dropped_chunks += 1
                else:
                    send_frame({
                        "type": "chunk",
                        "session": session,
                        "seq": seq,
                        "start_sample": start_sample,
                        "samples": samples,
                    })
                    sent_chunks += 1

                if total_chunks and (seq + 1) >= next_progress:
                    percent = int((seq + 1) * 100 / total_chunks)
                    configure_streams()
                    print(f"Send progress {percent}% ({seq + 1}/{total_chunks} chunks).", file=sys.stderr, flush=True)
                    next_progress += progress_step
                if args.speed > 0 and seq + 1 < total_chunks:
                    time.sleep(len(samples) / args.fs / args.speed)

            if sent_samples != total_samples:
                raise ReplayError("SOURCE_CHANGED", "Source sample count changed during transmission; no end frame was sent.")
            send_frame({
                "type": "end",
                "session": session,
                "total_chunks": total_chunks,
                "total_samples": total_samples,
            })

        emit_result({
            "status": "sent",
            "error_code": None,
            "message": "Sender completed replay; the receiver must confirm completeness independently.",
            "session": session,
            "fs": args.fs,
            "unit": args.unit.strip(),
            "metadata_source": "Declared by sender arguments; not automatically detected.",
            "total_chunks": total_chunks,
            "sent_chunks": sent_chunks,
            "dropped_chunks": dropped_chunks,
            "total_samples": total_samples,
        })
        return 0
    except ReplayError as exc:
        emit_result({"status": "error", "error_code": exc.code, "message": exc.message})
        return 2
    except (OSError, UnicodeError, socket.timeout) as exc:
        emit_result({"status": "error", "error_code": "SENDER_IO_ERROR", "message": f"Transmission failed: {exc}"})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
