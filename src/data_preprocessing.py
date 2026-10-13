"""Strict EEG ingestion, record reconstruction, epoch alignment, and provenance caches.

Filtered input uses ``input_mode='preprocessed'``. Raw text must
explicitly select ``input_mode='raw'``; filtering happens once per continuous
record, after joining its original file parts. Invalid labels and absent signal
parts split temporal sequences instead of connecting unrelated observations.
"""
from collections import defaultdict
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import warnings

import numpy as np
from scipy import signal


PREPROCESSING_VERSION = 5
FILENAME_EEG_RE = re.compile(
    r"^(.+)_EEG[^/]*?_Part_(\d+)_of_(\d+)(?:_filtered_eeg_data)?\.txt$", re.I)
FILENAME_LABEL_RE = re.compile(
    r"^(.+)_Hypnogram_Data_Part_(\d+)_of_(\d+)\.txt$", re.I)


def participant_id(record_id):
    """Group multiple nights for the same source participant before splitting."""
    record_id = str(record_id).split("::", 1)[0]
    match = re.match(r"^(SC\d{3}|ST\d{3})\d[A-Za-z]\d$", record_id)
    return match.group(1) if match else record_id


def butterworth_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply a zero-phase Butterworth bandpass using stable second-order sections."""
    if not 0 < lowcut < highcut < fs / 2:
        raise ValueError("Bandpass requires 0 < lowcut < highcut < fs / 2")
    sos = signal.butter(order, [lowcut, highcut], fs=fs, btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, np.asarray(data, dtype=np.float64))


def apply_notch_filter(data, fs=100, notch_freq=50.0, notch_q=30.0):
    if not 0 < notch_freq < fs / 2:
        raise ValueError("Notch frequency must be strictly below the Nyquist frequency")
    if notch_q <= 0:
        raise ValueError("Notch quality factor must be positive")
    b, a = signal.iirnotch(notch_freq, notch_q, fs=fs)
    return signal.filtfilt(b, a, data)


def apply_filter(data, fs=100, lowcut=0.5, highcut=30.0, order=4,
                 enable_notch=False, notch_freq=None, notch_q=30.0,
                 detrend_type=None):
    filtered = butterworth_bandpass_filter(data, lowcut, highcut, fs, order)
    if detrend_type is not None:
        if detrend_type not in ("constant", "linear"):
            raise ValueError("detrend_type must be constant, linear, or None")
        filtered = signal.detrend(filtered, type=detrend_type)
    if enable_notch:
        if notch_freq is None:
            raise ValueError("An enabled notch requires notch_freq")
        filtered = apply_notch_filter(filtered, fs, notch_freq, notch_q)
    return filtered.astype(np.float32)


def normalize_epoch(epoch, mode=None, eps=1e-8):
    if mode is None:
        return epoch
    if mode != "zscore":
        raise ValueError("epoch_normalize must be zscore or None")
    epoch = np.asarray(epoch, dtype=np.float32)
    return (epoch - epoch.mean()) / max(float(epoch.std()), eps)


def clip_epoch(epoch, clip_value=None):
    if clip_value is None:
        return epoch
    if not np.isfinite(clip_value) or clip_value <= 0:
        raise ValueError("clip_value must be finite and positive")
    return np.clip(epoch, -float(clip_value), float(clip_value))


def load_eeg_txt(filepath):
    """Read a finite numeric TXT vector without silently dropping bad tokens.

    Whitespace-separated numeric values and comment lines starting with ``#``
    are supported. A corruption raises an error rather than shifting timestamps.
    """
    try:
        values = np.loadtxt(filepath, dtype=np.float64, encoding="utf-8-sig", ndmin=1)
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Invalid numeric EEG text in {filepath}: {exc}") from exc
    if values.ndim != 1:
        raise ValueError(f"EEG text must contain one numeric column: {filepath}")
    if not len(values):
        raise ValueError(f"EEG file is empty: {filepath}")
    if not np.isfinite(values).all():
        raise ValueError(f"EEG contains NaN or infinity: {filepath}")
    return values


def parse_label_file(filepath):
    """Parse onset/end/duration/description rows, preserving blank unknown labels."""
    segments = []
    with open(filepath, encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("onset"):
                continue
            parts = re.split(r"\s+", line, maxsplit=3)
            if len(parts) < 3:
                raise ValueError(f"Malformed label row {filepath}:{line_number}")
            try:
                onset, end, duration = map(float, parts[:3])
            except ValueError as exc:
                raise ValueError(f"Invalid label timestamp {filepath}:{line_number}") from exc
            if not all(math.isfinite(x) for x in (onset, end, duration)):
                raise ValueError(f"Non-finite label timestamp {filepath}:{line_number}")
            if onset < 0 or end < onset or duration < 0 or abs(end - onset - duration) > 1e-5:
                raise ValueError(f"Inconsistent label interval {filepath}:{line_number}")
            segments.append((onset, end, duration, parts[3].strip() if len(parts) == 4 else ""))
    return segments


def map_description_to_label(desc):
    d = desc.strip().lower()
    aliases = {"w": 0, "wake": 0, "sleep stage w": 0,
               "1": 1, "n1": 1, "sleep stage 1": 1,
               "2": 2, "n2": 2, "sleep stage 2": 2,
               "3": 3, "4": 3, "n3": 3, "sleep stage 3": 3, "sleep stage 4": 3,
               "r": 4, "rem": 4, "sleep stage r": 4}
    return aliases.get(d, -1)


def scan_dataset(data_dir):
    """Recursively pair raw/filtered EEG and labels; reject duplicate record parts."""
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    subjects, labels = defaultdict(dict), defaultdict(dict)
    eeg_files = []
    for path in sorted(root.rglob("*.txt")):
        label_match = FILENAME_LABEL_RE.fullmatch(path.name)
        eeg_match = FILENAME_EEG_RE.fullmatch(path.name)
        if label_match:
            sid, part = label_match.group(1), int(label_match.group(2))
            if part in labels[sid]:
                raise ValueError(f"Duplicate label for {sid} part {part}")
            labels[sid][part] = str(path)
        elif eeg_match:
            eeg_files.append((path, eeg_match))
    for path, match in eeg_files:
        sid, part = match.group(1), int(match.group(2))
        if part in subjects[sid]:
            raise ValueError(f"Duplicate EEG for {sid} part {part}; select one input mode directory")
        subjects[sid][part] = (str(path), labels[sid].get(part))
    return dict(subjects)


def _resolve_notch_options(enable_notch=False, notch=None, notch_freq=50.0):
    return (True, float(notch)) if notch is not None else (bool(enable_notch), float(notch_freq))


def localize_label_segments(segments, start_sec, duration_sec, timebase="global"):
    """Convert a part's labels to local seconds and clip its boundary markers."""
    if timebase not in ("global", "local"):
        raise ValueError("label_timebase must be global or local")
    offset = start_sec if timebase == "global" else 0.0
    result = []
    for onset, end, _, desc in segments:
        onset, end = onset - offset, end - offset
        if end < -1e-5 or onset > duration_sec + 1e-5:
            raise ValueError("Label interval lies outside its EEG part; check label_timebase and sample rate")
        onset, end = max(onset, 0.0), min(end, duration_sec)
        if end > onset:
            result.append((onset, end, end - onset, desc))
    return result


def load_record(subject_parts, fs=100, input_mode="preprocessed", lowcut=0.5,
                highcut=30.0, order=4, enable_notch=False, notch=None,
                notch_freq=50.0, notch_q=30.0, detrend_type=None,
                label_timebase="global", part_start_samples=None):
    """Join original parts and filter each continuous raw block once.

    ``eeg`` indexes record-global samples, with explicit NaN holes for absent
    parts. ``blocks`` identify continuous signal. Absolute starts after missing
    parts require ``part_start_samples``; sparse labels cannot prove a part start.
    """
    if fs <= 0 or input_mode not in ("raw", "preprocessed"):
        raise ValueError("Use a positive sample rate and input_mode raw or preprocessed")
    enable_notch, notch_freq = _resolve_notch_options(enable_notch, notch, notch_freq)
    declared_starts = {int(key): int(value) for key, value in (part_start_samples or {}).items()}
    if any(float(value) != declared_starts[int(key)] or declared_starts[int(key)] < 0
           for key, value in (part_start_samples or {}).items()):
        raise ValueError("part_start_samples values must be nonnegative integer sample indices")
    arrays, metadata, labels, blocks = [], [], [], []
    timeline_end, previous_part = 0.0, None
    for part in sorted(subject_parts):
        eeg_path, label_path = subject_parts[part]
        values = load_eeg_txt(eeg_path)
        duration = len(values) / fs
        part_labels = parse_label_file(label_path) if label_path else []
        part_contiguous = previous_part is not None and part == previous_part + 1
        missing_before = (previous_part is None and part != 1) or (previous_part is not None and not part_contiguous)
        if missing_before and part not in declared_starts:
            raise ValueError("Missing EEG parts require explicit part_start_samples; annotation onsets cannot prove signal offsets")
        start_sec = declared_starts[part] / fs if part in declared_starts else timeline_end
        contiguous = part_contiguous and abs(start_sec - timeline_end) < 1e-7
        start_sample = int(round(start_sec * fs))
        if start_sample < int(round(timeline_end * fs)):
            raise ValueError("Global part timestamps overlap earlier samples")
        if abs(start_sec * fs - start_sample) > 1e-5:
            raise ValueError("Global part timestamp does not align with a sample")
        local_labels = localize_label_segments(part_labels, start_sec, duration, label_timebase)
        labels.extend((a + start_sec, b + start_sec, d, desc) for a, b, d, desc in local_labels)
        if not contiguous:
            blocks.append({"start_sample": start_sample, "end_sample": start_sample,
                           "start_sec": start_sec, "end_sec": start_sec})
        blocks[-1]["end_sample"] = start_sample + len(values)
        blocks[-1]["end_sec"] = start_sec + duration
        metadata.append({"part": int(part), "sample_count": int(len(values)),
                         "start_sample": start_sample, "end_sample": start_sample + len(values),
                         "start_sec": start_sec, "end_sec": start_sec + duration,
                         "eeg_path": str(eeg_path), "label_path": str(label_path) if label_path else None})
        arrays.append(values)
        timeline_end, previous_part = start_sec + duration, part
    eeg = np.full(int(round(timeline_end * fs)), np.nan, dtype=np.float64)
    for values, part in zip(arrays, metadata):
        eeg[part["start_sample"]:part["end_sample"]] = values
    if input_mode == "raw":
        for block in blocks:
            start, end = block["start_sample"], block["end_sample"]
            eeg[start:end] = apply_filter(eeg[start:end], fs, lowcut, highcut, order,
                                           enable_notch, notch_freq, notch_q, detrend_type)
    elif enable_notch or detrend_type is not None:
        warnings.warn("Filter options are ignored for preprocessed input", RuntimeWarning)
    return {"eeg": eeg.astype(np.float32), "label_segments": labels,
            "part_metadata": metadata, "blocks": blocks, "input_mode": input_mode}


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_cache_key(subject_parts, **processing):
    """Content-addressed key: source file hashes, processing parameters, code version."""
    bound = _LOAD_RECORD_SIGNATURE.bind(subject_parts, **processing)
    bound.apply_defaults()  # equivalent calls (explicit vs default arguments) share one key
    processing = {name: value for name, value in bound.arguments.items() if name != "subject_parts"}
    files = []
    for part in sorted(subject_parts):
        for path in subject_parts[part]:
            if path is not None:
                files.append({"part": int(part), "name": Path(path).name, "sha256": _file_sha256(path)})
    payload = {"version": PREPROCESSING_VERSION, "processing": processing, "files": files}
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("ascii")).hexdigest()


def load_record_cached(record_id, subject_parts, cache_dir, **kwargs):
    """``load_record`` with a validated on-disk cache of the joined, filtered signal.

    The cache key covers the content of every source file, all processing
    parameters and the preprocessing version, so edited data, a different filter
    or changed alignment code can never be served stale. The signal is
    memory-mapped read-only.
    """
    key = record_cache_key(subject_parts, **kwargs)
    folder = Path(cache_dir)
    signal_path, meta_path = folder / f"{record_id}_{key[:16]}.npy", folder / f"{record_id}_{key[:16]}.json"
    if signal_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="ascii"))
        if meta.get("cache_key") == key:
            meta.pop("cache_key")
            meta["eeg"] = np.load(signal_path, mmap_mode="r")
            return meta
    record = load_record(subject_parts, **kwargs)
    folder.mkdir(parents=True, exist_ok=True)
    temporary = signal_path.with_name(signal_path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, record["eeg"])
    os.replace(temporary, signal_path)
    meta = {key_: value for key_, value in record.items() if key_ != "eeg"}
    meta_path.write_text(json.dumps({**meta, "cache_key": key}), encoding="ascii")
    record["eeg"] = np.load(signal_path, mmap_mode="r")
    return record


_LOAD_RECORD_SIGNATURE = inspect.signature(load_record)


def _merged_labels(label_segments):
    merged = []
    for onset, end, _, desc in sorted(label_segments, key=lambda x: (x[0], x[1])):
        if end <= onset:
            continue
        label = map_description_to_label(desc)
        if merged and onset < merged[-1][1] - 1e-5:
            raise ValueError("Overlapping positive-duration labels")
        if merged and abs(onset - merged[-1][1]) < 1e-5 and label == merged[-1][2]:
            merged[-1] = (merged[-1][0], end, label)
        else:
            merged.append((onset, end, label))
    return merged


def _record_epochs(record, fs, epoch_sec, epoch_normalize, clip_value):
    epoch_len = int(round(fs * epoch_sec))
    if epoch_len <= 0 or abs(epoch_len - fs * epoch_sec) > 1e-6:
        raise ValueError("fs * epoch_sec must be a positive integer")
    frames, labels, starts, block_ids = [], [], [], []
    intervals = _merged_labels(record["label_segments"])
    for block_id, block in enumerate(record["blocks"]):
        for onset, end, label in intervals:
            if label < 0:
                continue
            onset, end = max(onset, block["start_sec"]), min(end, block["end_sec"])
            first = int(math.ceil((onset - 1e-7) / epoch_sec))
            last = int(math.floor((end + 1e-7) / epoch_sec))
            for epoch_index in range(first, last):
                onset_sec = epoch_index * epoch_sec
                index = block["start_sample"] + int(round((onset_sec - block["start_sec"]) * fs))
                if index < block["start_sample"] or index + epoch_len > block["end_sample"]:
                    continue
                frame = record["eeg"][index:index + epoch_len]
                frames.append(clip_epoch(normalize_epoch(frame, epoch_normalize), clip_value))
                labels.append(label)
                starts.append(epoch_index)
                block_ids.append(block_id)
    X = np.stack(frames).astype(np.float32) if frames else np.empty((0, epoch_len), np.float32)
    return X, np.asarray(labels, np.int64), np.asarray(starts, np.int64), np.asarray(block_ids, np.int64)


def segment_by_epoch(eeg, label_segments, fs=100, epoch_sec=30,
                     epoch_normalize=None, clip_value=None):
    record = {"eeg": np.asarray(eeg), "label_segments": label_segments,
              "blocks": [{"start_sample": 0, "end_sample": len(eeg), "start_sec": 0.0,
                           "end_sec": len(eeg) / fs}]}
    X, y, _, _ = _record_epochs(record, fs, epoch_sec, epoch_normalize, clip_value)
    return X, y


def preprocess_subject(subject_parts, fs=100, epoch_sec=30, lowcut=0.5,
                       highcut=30.0, order=4, enable_notch=False, notch=None,
                       notch_freq=50.0, notch_q=30.0, detrend_type=None,
                       epoch_normalize=None, clip_value=None,
                       input_mode="preprocessed", label_timebase="global", part_start_samples=None):
    record = load_record(subject_parts, fs, input_mode, lowcut, highcut, order,
                         enable_notch, notch, notch_freq, notch_q, detrend_type, label_timebase, part_start_samples)
    X, y, _, _ = _record_epochs(record, fs, epoch_sec, epoch_normalize, clip_value)
    return X, y


def _cache_metadata(subjects, processing):
    files = []
    for sid, parts in sorted(subjects.items()):
        for part, pair in sorted(parts.items()):
            for path in pair:
                if path is not None:
                    info = Path(path).stat()
                    digest = hashlib.sha256()
                    with open(path, "rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                    files.append({"path": str(Path(path).resolve()), "size": info.st_size,
                                  "mtime_ns": info.st_mtime_ns, "sha256": digest.hexdigest(),
                                  "record": sid, "part": part})
    payload = {"version": PREPROCESSING_VERSION, "processing": processing, "files": files}
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return serialized, hashlib.sha256(serialized.encode("ascii")).hexdigest()


def build_training_set(train_dir, fs=100, epoch_sec=30, lowcut=0.5, highcut=30.0,
                       order=4, enable_notch=False, notch=None, notch_freq=50.0,
                       notch_q=30.0, detrend_type=None, epoch_normalize=None,
                       clip_value=None, cache_path=None, verbose=True,
                       input_mode="preprocessed", label_timebase="global", record_part_start_samples=None,
                       records=None, record_cache_dir=None):
    """Return epochs, labels, and continuity IDs with metadata-validated caching.

    ``records`` restricts loading to the named recordings (for example one
    partition of a split manifest). ``record_cache_dir`` stores each joined,
    filtered record once so repeated experiments skip text parsing.
    """
    subjects = scan_dataset(train_dir)
    if records is not None:
        missing = sorted(set(records) - set(subjects))
        if missing:
            raise ValueError(f"Requested records not found in {train_dir}: {missing}")
        subjects = {sid: parts for sid, parts in subjects.items() if sid in set(records)}
    processing = {key: value for key, value in locals().copy().items()
                  if key not in ("subjects", "train_dir", "cache_path", "verbose", "record_cache_dir")}
    processing["records"] = sorted(subjects)
    metadata, cache_key = _cache_metadata(subjects, processing)
    if cache_path and Path(cache_path).exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            if "cache_key" in cached and str(cached["cache_key"].item()) == cache_key:
                if verbose:
                    print(f"[preprocess] Validated cache: {cache_path}")
                return cached["X"], cached["y"], cached["subject_ids"]
    all_X, all_y, all_sid = [], [], []
    for sid, parts in sorted(subjects.items()):
        arguments = dict(fs=fs, input_mode=input_mode, lowcut=lowcut, highcut=highcut, order=order,
                         enable_notch=enable_notch, notch=notch, notch_freq=notch_freq, notch_q=notch_q,
                         detrend_type=detrend_type, label_timebase=label_timebase,
                         part_start_samples=(record_part_start_samples or {}).get(sid))
        record = (load_record_cached(sid, parts, record_cache_dir, **arguments) if record_cache_dir
                  else load_record(parts, **arguments))
        X, y, epoch_indices, block_ids = _record_epochs(record, fs, epoch_sec, epoch_normalize, clip_value)
        if not len(y):
            continue
        segment_breaks = np.r_[True, (np.diff(epoch_indices) != 1) | (np.diff(block_ids) != 0)]
        segment_numbers = np.cumsum(segment_breaks) - 1
        all_X.append(X)
        all_y.append(y)
        all_sid.append(np.asarray([f"{sid}::segment_{number}" for number in segment_numbers]))
        if verbose:
            print(f"[preprocess] {sid}: {len(y)} epochs; class counts {np.bincount(y, minlength=5).tolist()}")
    epoch_len = int(round(fs * epoch_sec))
    X = np.concatenate(all_X) if all_X else np.empty((0, epoch_len), np.float32)
    y = np.concatenate(all_y) if all_y else np.empty(0, np.int64)
    subject_ids = np.concatenate(all_sid) if all_sid else np.array([], dtype=str)
    if cache_path:
        target = Path(cache_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, X=X, y=y, subject_ids=subject_ids,
                                cache_key=np.asarray(cache_key), metadata=np.asarray(metadata))
        os.replace(temporary, target)
    return X, y, subject_ids
