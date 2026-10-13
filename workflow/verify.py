"""Verify prediction timeline, sample coverage, labels, and frozen inference settings."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from common import VERSION_ROOT, asset_root, emit_result, executing_source_hashes, model_integrity, require_received_integrity, utc_now, write_json


STAGE_NAMES = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage R": "REM",
}
STAGE_IDS = {label: index for index, label in enumerate(STAGE_NAMES)}
SHORT_STAGE_IDS = {name: index for index, name in enumerate(STAGE_NAMES.values())}
UNKNOWN_ID = -1


class VerificationError(Exception):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VerificationError(f"Expected a JSON object in {path.name}.")
    return value


def recompute_decoded(logits: np.ndarray, scorable: np.ndarray, transition: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode each observed scorable run separately, leaving invalid epochs Unknown."""
    raw = np.full(len(logits), -1, dtype=np.int64)
    decoded = raw.copy()
    require(transition.shape == (5, 5) and np.isfinite(transition).all() and np.all(transition > 0), "Reviewed learned transition matrix is invalid.")
    transition = transition / transition.sum(axis=1, keepdims=True)
    edges = np.flatnonzero(np.diff(np.r_[False, scorable, False]))
    for begin, end in zip(edges[::2], edges[1::2]):
        emissions = logits[begin:end].astype(np.float64)
        raw[begin:end] = emissions.argmax(axis=1)
        score = emissions[0] + math.log(0.2)
        back = np.zeros((len(emissions), 5), dtype=np.int64)
        for index in range(1, len(emissions)):
            candidates = score[:, None] + 0.25 * np.log(transition)
            back[index] = candidates.argmax(axis=0)
            score = emissions[index] + candidates.max(axis=0)
        path = np.empty(len(emissions), dtype=np.int64)
        path[-1] = score.argmax()
        for index in range(len(emissions) - 1, 0, -1):
            path[index - 1] = back[index, path[index]]
        decoded[begin:end] = path
    return raw, decoded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", required=True, help="Prediction output directory from stage.py")
    args = parser.parse_args()
    prediction_dir = Path(args.prediction_dir).expanduser().resolve()
    result_path = prediction_dir / "verification.json"
    verification: dict[str, Any] = {
        "status": "error",
        "prediction_dir": str(prediction_dir),
        "error_code": None,
        "message": None,
        "checks": {},
        "accuracy_evaluated": False,
    }

    try:
        stage = read_json(prediction_dir / "stage_result.json")
        require(stage.get("status") == "success" and stage.get("predictor_returncode") == 0, "Stage record is not successful.")
        received_hash = require_received_integrity(stage, Path(stage["received_eeg"]))
        integrity = model_integrity()
        require(stage.get("model_integrity") == integrity, "Stage model assets differ from the frozen trained assets.")
        source_hashes = executing_source_hashes()
        require(stage.get("executing_source_sha256") == source_hashes, "Executing repository source differs from the recorded prediction source.")
        private_assets = asset_root()
        source_manifest = stage.get("source_manifest")
        require(isinstance(source_manifest, dict), "Stage record does not include a source manifest.")
        require(source_manifest.get("processing_state") == "raw_unfiltered" and source_manifest.get("sample_count") == stage.get("sample_count"), "Source manifest does not declare the exact received raw sample total.")
        receiver = read_json(Path(stage["receiver_report"]))
        report = read_json(prediction_dir / "bci_recording_report.json")
        metadata = report.get("metadata", {})
        require(isinstance(metadata, dict), "Prediction report metadata is missing.")
        receiver_count = receiver.get("received_samples")
        fs = receiver.get("fs")
        require(receiver.get("status") == "success", "Receiver report is not successful.")
        require(type(receiver_count) is int and receiver_count > 0 and fs == 100 and receiver.get("unit") == "V", "Receiver totals or declared 100 Hz/V metadata are invalid.")
        require(stage.get("sample_count") == receiver_count and receiver.get("expected_total_samples") == receiver_count, "Stage and receiver sample totals differ.")
        transfer_path = Path(stage["transfer_dir"]) / "transfer_result.json"
        transfer = read_json(transfer_path)
        require(transfer.get("status") == "success" and transfer.get("tokens_identical") is True and transfer.get("received_eeg_sha256") == received_hash and transfer.get("sample_count") == receiver_count, "Stage contents are not bound to a successful reviewed transfer.")
        raw_seconds = receiver_count / fs
        require(metadata.get("sampling_rate_hz") == 100 and metadata.get("input_mode") == "raw", "Prediction did not use raw 100 Hz input.")
        require(metadata.get("total_seconds") == raw_seconds, "Prediction duration differs from the receiver sample total.")
        require(Path(metadata.get("source_file", "")).resolve() == Path(stage["received_eeg"]).resolve(), "Prediction source is not the received EEG file.")

        epochs_path = prediction_dir / "bci_recording_epochs.npz"
        hypnogram_path = prediction_dir / "bci_recording_hypnogram.tsv"
        with np.load(epochs_path, allow_pickle=False) as archive:
            required_arrays = {"onset_sec", "decoded", "argmax", "confidence", "logits", "scorable", "ground_truth"}
            require(required_arrays.issubset(set(archive.files)), "Epoch NPZ is missing required arrays.")
            arrays = {key: archive[key] for key in required_arrays}

        epoch_count = receiver_count // (100 * 30)
        expected_onsets = np.arange(epoch_count, dtype=np.float64) * 30.0
        onsets = np.asarray(arrays["onset_sec"])
        decoded = np.asarray(arrays["decoded"])
        ground_truth = np.asarray(arrays["ground_truth"])
        scorable = np.asarray(arrays["scorable"])
        logits = np.asarray(arrays["logits"])
        argmax = np.asarray(arrays["argmax"])
        confidence = np.asarray(arrays["confidence"])
        require(onsets.shape == expected_onsets.shape and np.array_equal(onsets, expected_onsets), "Epoch time axis is not continuous from 0 at 30 second intervals.")
        require(decoded.shape == (epoch_count,) and ground_truth.shape == (epoch_count,) and scorable.shape == (epoch_count,), "Epoch arrays do not match the complete 30 second epoch count.")
        require(argmax.shape == (epoch_count,) and confidence.shape == (epoch_count,), "Argmax or confidence shape differs from the complete epoch count.")
        require(scorable.dtype == np.dtype(bool), "Scorable flags must be Boolean values.")
        require(np.issubdtype(onsets.dtype, np.number) and not np.issubdtype(onsets.dtype, np.complexfloating), "Epoch times must be real numeric values.")
        require(np.issubdtype(logits.dtype, np.floating) and np.issubdtype(confidence.dtype, np.floating), "Logits and confidence must be floating-point arrays.")
        require(np.issubdtype(argmax.dtype, np.integer) and np.issubdtype(ground_truth.dtype, np.integer), "Argmax and ground truth must be integer arrays.")
        require(logits.shape == (epoch_count, 5), "Logits do not contain five classes for every full epoch.")
        require(np.isfinite(onsets).all() and np.isfinite(logits[scorable]).all(), "Scored epoch values contain non-finite values.")
        require(np.issubdtype(decoded.dtype, np.integer), "Decoded epoch labels are not stored as integers.")
        require(np.all((decoded[scorable] >= 0) & (decoded[scorable] <= 4)) and np.all(decoded[~scorable] == -1), "Decoded labels do not match scorable epoch flags.")
        require(np.isfinite(confidence[scorable]).all() and np.all((confidence[scorable] >= 0) & (confidence[scorable] <= 1)), "Scored confidence must be finite probabilities between zero and one.")
        require(np.isnan(logits[~scorable]).all() and np.isnan(confidence[~scorable]).all(), "Unscorable epochs must have unavailable logits and confidence.")
        require(np.all(ground_truth == -1), "The standalone recording unexpectedly contains annotations.")
        require(metadata.get("known_annotation_classes") == [], "Prediction metadata reports known annotations.")

        protocol_path = private_assets / "reports" / "ens_CE" / "frozen_protocol.json"
        require(Path(metadata.get("protocol", "")).resolve() == protocol_path.resolve(), "Prediction did not use the reviewed frozen protocol path.")
        require(metadata.get("checkpoint_sha256") == integrity["checkpoint_sha256"] and metadata.get("protocol_sha256") == integrity["protocol_sha256"], "Prediction checkpoint or protocol identifier differs from reviewed assets.")
        protocol = read_json(protocol_path)
        require(metadata.get("decoder") == "viterbi", "Prediction decoder is not Viterbi.")
        require(metadata.get("transition_strength") == 0.25, "Prediction transition strength is not 0.25.")
        require(protocol.get("selected_variant") == "viterbi_learned_s0.25" and protocol.get("selected_prior") == "learned" and protocol.get("selected_strength") == 0.25, "Frozen protocol does not select learned Viterbi strength 0.25.")
        require(metadata.get("transition_is_heuristic") is False, "Prediction did not use the learned transition prior.")
        require(metadata.get("protocol_inference_contract_bound") in (True, False), "Protocol binding status is absent.")
        require(metadata.get("protocol_inference_contract_bound") is (protocol.get("inference_contract") is not None), "Protocol binding status differs from the unchanged protocol artifact.")
        config = read_json(private_assets / "runs" / "ens_CE" / "config.json")
        require(metadata.get("processing_parameters") == config["processing_parameters"], "Prediction preprocessing parameters differ from the reviewed configuration.")
        require(metadata.get("sequence_length") == config["SEQ_LEN"] and metadata.get("sequence_stride") == config["SEQ_LEN"] // 2 and metadata.get("tta") == 1, "Prediction sequence or augmentation settings differ from this workflow.")
        require(metadata.get("input_unit") == "V" and metadata.get("checkpoint_input_mode") == "raw" and metadata.get("normalization_epsilon") == 0.0, "Prediction input or normalization contract changed.")
        transition_path = private_assets / "runs" / "ens_CE" / "transition.json"
        require(Path(metadata.get("transition_source", "")).resolve() == transition_path.resolve(), "Prediction did not use the reviewed learned transition file.")
        transition = np.asarray(json.loads(transition_path.read_text(encoding="utf-8")), dtype=np.float64)
        require(np.array_equal(np.asarray(metadata.get("transition_prior")), transition), "Prediction transition values differ from the reviewed learned prior.")
        expected_argmax, expected_decoded = recompute_decoded(logits, scorable, transition)
        require(np.array_equal(argmax, expected_argmax), "NPZ argmax differs from the saved logits or scorable flags.")
        require(np.array_equal(decoded, expected_decoded), "NPZ decoded labels differ from learned Viterbi recomputation.")
        temperature = protocol.get("temperature")
        require(type(temperature) in (int, float) and math.isfinite(temperature) and temperature > 0, "Frozen confidence temperature is invalid.")
        expected_confidence = np.empty(int(scorable.sum()), dtype=np.float64)
        if scorable.any():
            values = logits[scorable].astype(np.float64) / temperature
            values -= values.max(axis=1, keepdims=True)
            probability = np.exp(values)
            expected_confidence = probability.max(axis=1) / probability.sum(axis=1)
        require(np.allclose(confidence[scorable], expected_confidence, rtol=2e-6, atol=1e-7), "NPZ confidence differs from temperature-scaled saved logits.")
        confidence_metadata = metadata.get("confidence")
        require(isinstance(confidence_metadata, dict) and report.get("confidence") == confidence_metadata, "Report confidence metadata is missing or inconsistent.")
        threshold = protocol["abstention"]["confidence_threshold"]
        require(confidence_metadata.get("temperature") == temperature and confidence_metadata.get("low_confidence_threshold") == threshold, "Report confidence settings differ from the frozen protocol.")
        require(confidence_metadata.get("scorable_epochs") == int(scorable.sum()) and confidence_metadata.get("low_confidence_epochs") == int((confidence[scorable] < threshold).sum()), "Report confidence counts differ from the saved epochs.")
        mean_confidence = confidence_metadata.get("mean_top_probability")
        require((mean_confidence is None if not scorable.any() else isinstance(mean_confidence, (int, float)) and math.isfinite(mean_confidence) and abs(mean_confidence - float(confidence[scorable].mean())) <= 2e-6), "Report mean confidence differs from saved epochs.")

        rows = []
        with hypnogram_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            require(reader.fieldnames == ["onset_sec", "end_sec", "duration_sec", "description"], "Hypnogram TSV columns do not match the expected schema.")
            for row in reader:
                start, end, duration = float(row["onset_sec"]), float(row["end_sec"]), float(row["duration_sec"])
                require(all(math.isfinite(value) for value in (start, end, duration)) and end > start, "Hypnogram contains an invalid time interval.")
                require(abs((end - start) - duration) <= 1e-5, "Hypnogram duration does not match interval endpoints.")
                description = row["description"]
                require(description in STAGE_NAMES or description == "Unknown", f"Unexpected hypnogram label: {description}")
                label_id = STAGE_IDS.get(description, UNKNOWN_ID)
                rows.append({"start": start, "end": end, "duration": duration, "description": description, "label_id": label_id})
        require(bool(rows), "Hypnogram has no timeline rows.")
        require(abs(rows[0]["start"]) <= 1e-6 and abs(rows[-1]["end"] - raw_seconds) <= 1e-5, "Hypnogram does not span the complete raw recording.")
        for left, right in zip(rows, rows[1:]):
            require(abs(left["end"] - right["start"]) <= 1e-5, "Hypnogram timeline contains a gap or overlap.")

        for epoch_index, label in enumerate(decoded):
            start, end = epoch_index * 30.0, (epoch_index + 1) * 30.0
            overlapping = [row for row in rows if row["end"] > start + 1e-5 and row["start"] < end - 1e-5]
            require(bool(overlapping), f"Hypnogram has no label for full epoch {epoch_index}.")
            observed = {row["label_id"] for row in overlapping}
            require(observed == {int(label)}, f"Hypnogram label differs from NPZ decoded label at epoch {epoch_index}.")

        stage_seconds: dict[str, float] = defaultdict(float)
        unknown_seconds = 0.0
        for row in rows:
            label = row["description"]
            if label in STAGE_NAMES:
                stage_seconds[STAGE_NAMES[label]] += row["duration"]
            elif label == "Unknown":
                unknown_seconds += row["duration"]
            else:
                raise VerificationError(f"Unexpected hypnogram label: {label}")
        scored_seconds = sum(stage_seconds.values())
        require(abs(scored_seconds + unknown_seconds - raw_seconds) <= 1e-4, "Scored and unknown seconds do not sum to the raw duration.")
        require(abs(scored_seconds - int(np.count_nonzero(decoded >= 0)) * 30.0) <= 1e-4, "Hypnogram stage durations do not match scored NPZ epochs.")
        decoded_stage_seconds = {name: int(np.count_nonzero(decoded == label_id)) * 30.0 for name, label_id in SHORT_STAGE_IDS.items()}
        require(all(abs(stage_seconds[name] - decoded_stage_seconds[name]) <= 1e-4 for name in SHORT_STAGE_IDS), "Per-class hypnogram durations differ from NPZ decoded epoch counts.")
        require(abs(float(report.get("recording_seconds", math.nan)) - raw_seconds) <= 1e-5, "Summary report recording duration differs from the raw sample total.")
        require(abs(float(report.get("scorable_seconds", math.nan)) - scored_seconds) <= 1e-4, "Summary report scored duration differs from the hypnogram.")
        require(abs(float(report.get("unscorable_seconds", math.nan)) - unknown_seconds) <= 1e-4, "Summary report Unknown duration differs from the hypnogram.")
        require(abs(float(report.get("scorable_fraction", math.nan)) - scored_seconds / raw_seconds) <= 1e-7, "Summary report coverage differs from the scored timeline.")
        stage_minutes = report.get("stage_minutes")
        require(isinstance(stage_minutes, dict), "Summary report stage totals are missing.")
        reported_stage_seconds = {name: float(stage_minutes.get(name, math.nan)) * 60.0 for name in SHORT_STAGE_IDS}
        require(all(abs(reported_stage_seconds[name] - stage_seconds[name]) <= 1e-4 for name in SHORT_STAGE_IDS), "Summary report per-class totals differ from the hypnogram.")
        remainder = receiver_count % (100 * 30)
        tail_check = "not_applicable" if remainder == 0 else "pass"
        if remainder:
            tail_start = epoch_count * 30.0
            tail_rows = [row for row in rows if row["end"] > tail_start + 1e-5 and row["start"] < raw_seconds - 1e-5]
            require(bool(tail_rows) and all(row["label_id"] == UNKNOWN_ID for row in tail_rows), "The incomplete tail shorter than 30 seconds is not marked Unknown.")

        protocol_bound = bool(metadata.get("protocol_inference_contract_bound"))
        limitations = [
            "This is a local workflow demonstration on one EEG file member, not an independent evaluation.",
            "The input is raw unfiltered text; SleepMind applies its offline raw filtering once before inference.",
            "Inference is offline and may use future context; this does not validate real-time BCI performance.",
            "No labels were read or supplied, so accuracy was not evaluated.",
            "Only the implemented non-finite and flat-epoch checks apply; broader artifact detection was not performed.",
            "Sleep-stage durations are research outputs and are not clinical sleep efficiency.",
        ]
        if not protocol_bound:
            limitations.append("The legacy frozen protocol selected the decoder and checkpoint but did not bind the full input-processing contract.")
        verification.update({
            "status": "pass",
            "error_code": None,
            "root_cause": None,
            "secondary_errors": [],
            "message": "All transfer, epoch, timeline, label, and inference-setting checks passed.",
            "created_utc": utc_now(),
            "sample_count": receiver_count,
            "sampling_rate_hz": fs,
            "recording_seconds": raw_seconds,
            "full_epoch_count": epoch_count,
            "epoch_samples": 3000,
            "epoch_onset_first_last_sec": [float(onsets[0]), float(onsets[-1])] if epoch_count else [],
            "tail_samples": remainder,
            "tail_seconds": remainder / 100.0,
            "tail_check": tail_check,
            "stage_seconds": {name: float(stage_seconds.get(name, 0.0)) for name in ("W", "N1", "N2", "N3", "REM")},
            "decoded_stage_seconds": decoded_stage_seconds,
            "scored_seconds": scored_seconds,
            "unknown_seconds": unknown_seconds,
            "ground_truth_values": sorted(set(int(value) for value in np.unique(ground_truth))),
            "labels_read": False,
            "accuracy_evaluated": False,
            "input_mode": metadata["input_mode"],
            "preprocessing": metadata.get("preprocessing"),
            "decoder": metadata["decoder"],
            "transition_strength": metadata["transition_strength"],
            "transition_source": metadata.get("transition_source"),
            "protocol_inference_contract_bound": protocol_bound,
            "received_eeg_sha256": received_hash,
            "reviewed_model_integrity": integrity,
            "executing_source_sha256": source_hashes,
            "checks": {
                "receiver_complete": True,
                "source_is_received_raw_eeg": True,
                "npz_safe_load_and_epoch_alignment": True,
                "npz_epoch_labels_match_tsv": True,
                "per_class_totals_match_npz_and_tsv": True,
                "timeline_continuous_from_zero_to_raw_end": True,
                "stage_plus_unknown_matches_duration": True,
                "incomplete_tail_is_unknown": True if remainder else "not_applicable",
                "ground_truth_is_unlabeled_minus_one": True,
                "frozen_learned_viterbi_025": True,
                "received_file_contents_unchanged": True,
                "reviewed_model_and_protocol_bytes": True,
                "argmax_and_viterbi_recomputed": True,
                "confidence_recomputed": True,
            },
            "limitations": limitations,
        })
    except Exception as exc:
        verification.update(error_code="VERIFICATION_FAILED", root_cause=str(exc), secondary_errors=[], message=str(exc), created_utc=utc_now())

    write_json(result_path, verification)
    if verification["status"] != "pass":
        (prediction_dir / "research_summary.md").unlink(missing_ok=True)
    if verification["status"] == "pass":
        stage_seconds = verification["stage_seconds"]
        total_seconds = verification["recording_seconds"]
        unknown_seconds = verification["unknown_seconds"]
        lines = [
            "# Local BCI-to-SleepMind Research Demonstration",
            "",
            "This report describes one locally replayed EEG file member. It is a workflow demonstration, not an independent model evaluation or clinical report.",
            "",
            f"- Input: {verification['sample_count']} raw samples at the declared {verification['sampling_rate_hz']} Hz and V contract.",
            f"- Duration: {total_seconds:.2f} seconds; {verification['full_epoch_count']} complete 30-second epochs.",
            f"- Inference: raw input mode with one offline filtering pass; learned Viterbi decoder strength {verification['transition_strength']:.2f}.",
            f"- Scored stage time: {verification['scored_seconds']:.2f} seconds; Unknown time: {unknown_seconds:.2f} seconds.",
            "",
            "| Stage | Duration (minutes) |",
            "| --- | ---: |",
        ]
        for name in ("W", "N1", "N2", "N3", "REM"):
            lines.append(f"| {name} | {stage_seconds[name] / 60.0:.2f} |")
        lines.extend([
            f"| Unknown | {unknown_seconds / 60.0:.2f} |",
            "",
            "No labels were read, so accuracy was not evaluated. Stage duration is not clinical sleep efficiency. This is offline inference with future context and does not establish real-time BCI performance.",
            "Only the implemented non-finite and flat-epoch checks apply; broader artifact detection was not performed.",
            "",
        ])
        if not verification["protocol_inference_contract_bound"]:
            lines.append("The legacy frozen protocol selected the learned decoder and checkpoint but did not bind the full preprocessing and inference contract.")
            lines.append("")
        (prediction_dir / "research_summary.md").write_text("\n".join(lines), encoding="ascii")
    artifacts = {"verification": result_path}
    research_summary_path = prediction_dir / "research_summary.md"
    if verification["status"] == "pass" and research_summary_path.is_file():
        artifacts["research_summary"] = research_summary_path
    if verification["status"] == "pass":
        stage_seconds = verification["stage_seconds"]
        metrics = ", ".join(f"{name} {stage_seconds[name]:.0f}s" for name in ("W", "N1", "N2", "N3", "REM"))
        summary = (
            f"Verified {verification['sample_count']} samples / {verification['recording_seconds']:.2f}s, "
            f"{verification['full_epoch_count']} full epochs; {verification['scored_seconds']:.2f}s scored, "
            f"{verification['unknown_seconds']:.2f}s Unknown, tail={verification['tail_check']}; {metrics}."
        )
        error = None
    else:
        summary = verification.get("message") or "Verification failed."
        error = {
            "code": verification.get("error_code"),
            "root_cause": verification.get("root_cause") or summary,
            "secondary": [],
        }
    emit_result("verify", result_path, artifacts, summary, error, None)
    return 0 if verification["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
