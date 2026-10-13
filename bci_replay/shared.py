"""Small shared helpers for the local JSONL EEG replay tools."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import BinaryIO, Iterator


TOKEN_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
TOKEN_SPLIT_RE = re.compile(r"[\s,]+")
MAX_FRAME_BYTES = 4 * 1024 * 1024


class ReplayError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def validate_number_token(token: object) -> str:
    """Return a decimal/scientific token unchanged, or raise ValueError."""
    if not isinstance(token, str) or not TOKEN_RE.fullmatch(token):
        raise ValueError("Sample is not valid decimal real-number text.")
    try:
        value = float(token)
    except ValueError as exc:
        raise ValueError("Sample is not a valid real number.") from exc
    if not math.isfinite(value):
        raise ValueError("Sample must be a finite real number.")
    return token


def iter_source_tokens(path: str | Path) -> Iterator[str]:
    """Read whitespace- or comma-separated sample tokens without rewriting them."""
    with open(path, "r", encoding="utf-8", newline=None) as source:
        for line_number, line in enumerate(source, 1):
            for token in TOKEN_SPLIT_RE.split(line.strip()):
                if not token:
                    continue
                try:
                    yield validate_number_token(token)
                except ValueError as exc:
                    raise ReplayError(
                        "INVALID_SAMPLE_TOKEN",
                        f"Line {line_number} contains invalid sample {token!r}: {exc}",
                    ) from exc


def scan_source(path: str | Path, chunk_size: int) -> tuple[int, int]:
    sample_count = sum(1 for _ in iter_source_tokens(path))
    chunk_count = (sample_count + chunk_size - 1) // chunk_size
    return chunk_count, sample_count


def iter_chunks(path: str | Path, chunk_size: int) -> Iterator[list[str]]:
    chunk: list[str] = []
    for token in iter_source_tokens(path):
        chunk.append(token)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def encode_frame(value: dict[str, object]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def read_frame(reader: BinaryIO) -> bytes:
    """Read one bounded JSONL frame from a binary buffered socket reader."""
    line = reader.readline(MAX_FRAME_BYTES + 1)
    if len(line) > MAX_FRAME_BYTES:
        raise ReplayError("FRAME_TOO_LARGE", "Protocol frame exceeds the 4 MiB limit.")
    return line


def require_int(value: object, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ReplayError("INVALID_PROTOCOL", f"Field {field} must be an integer greater than or equal to {minimum}.")
    return value


def require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReplayError("INVALID_PROTOCOL", f"Field {field} must be nonempty text.")
    return value


def require_finite_positive(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayError("INVALID_PROTOCOL", f"Field {field} must be finite and positive.")
    try:
        parsed = float(value)
    except OverflowError as exc:
        raise ReplayError("INVALID_PROTOCOL", f"Field {field} must be finite and positive.") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ReplayError("INVALID_PROTOCOL", f"Field {field} must be finite and positive.")
    return parsed
