"""Synthetic workflow tests requiring no released data, checkpoints or accounts."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workflow"))
import common
import entry
import stage
import transfer
import verify


def invoke(monkeypatch, module, arguments):
    monkeypatch.setattr(sys, "argv", [str(module.__file__), *map(str, arguments)])
    return module.main()


def synthetic_input(folder, count=301):
    folder.mkdir()
    path = folder / "input.txt"
    path.write_text("\n".join(f"{index * 1e-7:.8f}" for index in range(count)) + "\n", encoding="ascii")
    manifest = {
        "sample_count": count, "extracted_bytes": path.stat().st_size,
        "sha256": common.sha256_file(path), "sampling_rate_contract_hz": 100,
        "amplitude_unit_contract": "V", "amplitude_unit_auto_detected": False,
        "processing_state": "raw_unfiltered", "labels_read": False,
    }
    common.write_json(folder / "source_manifest.json", manifest)
    return path


def transfer_result(tmp_path, monkeypatch):
    source = synthetic_input(tmp_path / "input")
    output = tmp_path / "transfer"
    assert invoke(monkeypatch, transfer, ["--input", source, "--output-dir", output]) == 0
    return output, json.loads((output / "transfer_result.json").read_text())


def test_loopback_transfer_preserves_generated_tokens(tmp_path, monkeypatch):
    output, result = transfer_result(tmp_path, monkeypatch)
    assert result["sample_count"] == 301
    assert result["compared_tokens"] == 301
    assert result["tokens_identical"] is True
    assert result["received_eeg_sha256"] == common.sha256_file(output / "receiver/received_eeg.txt")


def test_drop_stops_before_prediction(tmp_path, monkeypatch, capsys):
    source = synthetic_input(tmp_path / "input")
    output = tmp_path / "transfer"
    assert invoke(monkeypatch, transfer, ["--input", source, "--output-dir", output, "--drop-chunk", 1]) == 2
    result = json.loads((output / "transfer_result.json").read_text())
    assert result["error_code"] == "CHUNK_ORDER"
    assert not (output / "receiver/received_eeg.txt").exists()
    envelope = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert envelope["next_action"] is None


def test_missing_manifest_starts_no_transport(tmp_path, monkeypatch):
    source = synthetic_input(tmp_path / "input")
    (source.parent / "source_manifest.json").unlink()
    monkeypatch.setattr(transfer.subprocess, "Popen", lambda *a, **k: pytest.fail("Transport started"))
    output = tmp_path / "transfer"
    assert invoke(monkeypatch, transfer, ["--input", source, "--output-dir", output]) == 2
    result = json.loads((output / "transfer_result.json").read_text())
    assert result["error_code"] == "SOURCE_MANIFEST_MISSING"
    assert result["sender_returncode"] is None and result["receiver_returncode"] is None


def test_received_mutation_rejected_before_predictor(tmp_path, monkeypatch):
    output, result = transfer_result(tmp_path, monkeypatch)
    path = output / "receiver/received_eeg.txt"
    tokens = path.read_text().split()
    tokens[0] = "1.0"
    path.write_text("\n".join(tokens) + "\n", encoding="ascii")
    monkeypatch.setattr(stage.subprocess, "run", lambda *a, **k: pytest.fail("Predictor started"))
    prediction = tmp_path / "prediction"
    assert invoke(monkeypatch, stage, ["--transfer-dir", output, "--output-dir", prediction]) == 2
    result = json.loads((prediction / "stage_result.json").read_text())
    assert result["error_code"] == "TRANSFER_PREFLIGHT_FAILED"
    assert result["predictor_invoked"] is False


def test_no_assets_fails_clearly_without_random_inference(tmp_path, monkeypatch):
    output, _ = transfer_result(tmp_path, monkeypatch)
    monkeypatch.delenv("SLEEPMIND_ASSET_ROOT", raising=False)
    monkeypatch.setattr(stage.subprocess, "run", lambda *a, **k: pytest.fail("Predictor started"))
    prediction = tmp_path / "prediction"
    assert invoke(monkeypatch, stage, ["--transfer-dir", output, "--output-dir", prediction]) == 2
    result = json.loads((prediction / "stage_result.json").read_text())
    assert result["error_code"] == "MODEL_INTEGRITY_FAILED"
    assert "SLEEPMIND_ASSET_ROOT" in result["message"]
    assert result["predictor_invoked"] is False
    assert not (prediction / "bci_recording_epochs.npz").exists()


def test_fixed_private_asset_hash_rejects_changed_normalization(tmp_path, monkeypatch):
    root = tmp_path / "assets"
    file = root / "runs/ens_CE/norm_stats.npz"
    file.parent.mkdir(parents=True)
    file.write_bytes(b"synthetic normalization fixture")
    monkeypatch.setattr(common, "EXPECTED_ASSET_HASHES", {"runs/ens_CE/norm_stats.npz": "0" * 64})
    with pytest.raises(ValueError, match="frozen hash"):
        common.model_integrity(root)


def test_source_provenance_is_separate_from_trained_assets():
    sources = common.executing_source_hashes()
    assert "src/predict.py" in sources and "workflow/stage.py" in sources
    assert not any(name.startswith(("runs/", "reports/")) for name in sources)
    assert len(common.EXPECTED_ASSET_HASHES) == 20


def zero_epoch_prediction(tmp_path, monkeypatch):
    """Construct an empty-epoch verifier fixture; no predictor is launched."""
    source = synthetic_input(tmp_path / "input", count=2999)
    prediction = tmp_path / "prediction"
    receiver_dir = tmp_path / "transfer/receiver"
    prediction.mkdir()
    receiver_dir.mkdir(parents=True)
    received = receiver_dir / "received_eeg.txt"
    received.write_bytes(source.read_bytes())
    received_hash = common.sha256_file(received)
    manifest = json.loads((source.parent / "source_manifest.json").read_text())
    assets = tmp_path / "assets"
    protocol_path = assets / "reports/ens_CE/frozen_protocol.json"
    config_path = assets / "runs/ens_CE/config.json"
    protocol_path.parent.mkdir(parents=True)
    config_path.parent.mkdir(parents=True)
    protocol = {
        "selected_variant": "viterbi_learned_s0.25", "selected_prior": "learned",
        "selected_strength": 0.25, "temperature": 1.0,
        "abstention": {"confidence_threshold": 0.7},
    }
    processing = {"fs": 100, "epoch_sec": 30, "input_mode": "raw"}
    common.write_json(protocol_path, protocol)
    common.write_json(config_path, {"SEQ_LEN": 20, "processing_parameters": processing})
    transition = np.full((5, 5), 0.2)
    transition_path = assets / "runs/ens_CE/transition.json"
    transition_path.write_text(json.dumps(transition.tolist()), encoding="ascii")
    integrity = {"checkpoint_sha256": "synthetic-checkpoint", "protocol_sha256": common.sha256_file(protocol_path)}
    monkeypatch.setattr(verify, "model_integrity", lambda: integrity)
    monkeypatch.setenv("SLEEPMIND_ASSET_ROOT", str(assets))
    common.write_json(receiver_dir / "report.json", {
        "status": "success", "received_samples": 2999, "expected_total_samples": 2999,
        "fs": 100, "unit": "V",
    })
    common.write_json(receiver_dir.parent / "transfer_result.json", {
        "status": "success", "tokens_identical": True, "sample_count": 2999,
        "received_eeg_sha256": received_hash,
    })
    common.write_json(prediction / "stage_result.json", {
        "status": "success", "predictor_returncode": 0, "sample_count": 2999,
        "received_eeg": str(received), "received_eeg_sha256": received_hash,
        "receiver_report": str(receiver_dir / "report.json"), "transfer_dir": str(receiver_dir.parent),
        "source_manifest": manifest, "model_integrity": integrity,
        "executing_source_sha256": common.executing_source_hashes(),
    })
    confidence = {
        "temperature": 1.0, "low_confidence_threshold": 0.7, "scorable_epochs": 0,
        "low_confidence_epochs": 0, "mean_top_probability": None,
    }
    metadata = {
        "sampling_rate_hz": 100, "input_mode": "raw", "total_seconds": 29.99,
        "source_file": str(received), "known_annotation_classes": [],
        "protocol": str(protocol_path), "checkpoint_sha256": integrity["checkpoint_sha256"],
        "protocol_sha256": integrity["protocol_sha256"], "decoder": "viterbi", "transition_strength": 0.25,
        "transition_is_heuristic": False, "protocol_inference_contract_bound": False,
        "processing_parameters": processing, "sequence_length": 20, "sequence_stride": 10, "tta": 1,
        "input_unit": "V", "checkpoint_input_mode": "raw", "normalization_epsilon": 0.0,
        "transition_source": str(transition_path), "transition_prior": transition.tolist(), "confidence": confidence,
    }
    common.write_json(prediction / "bci_recording_report.json", {
        "metadata": metadata, "recording_seconds": 29.99, "scorable_seconds": 0.0,
        "unscorable_seconds": 29.99, "scorable_fraction": 0.0, "confidence": confidence,
        "stage_minutes": {name: 0.0 for name in ("W", "N1", "N2", "N3", "REM")},
    })
    np.savez_compressed(prediction / "bci_recording_epochs.npz", onset_sec=np.empty(0),
        decoded=np.empty(0, np.int64), argmax=np.empty(0, np.int64), ground_truth=np.empty(0, np.int64),
        confidence=np.empty(0, np.float32), logits=np.empty((0, 5), np.float32), scorable=np.empty(0, bool))
    (prediction / "bci_recording_hypnogram.tsv").write_text(
        "onset_sec\tend_sec\tduration_sec\tdescription\n0.000000\t29.990000\t29.990000\tUnknown\n", encoding="ascii")
    return prediction


def test_zero_complete_epochs_verify_as_unknown(tmp_path, monkeypatch):
    prediction = zero_epoch_prediction(tmp_path, monkeypatch)
    assert invoke(monkeypatch, verify, ["--prediction-dir", prediction]) == 0
    result = json.loads((prediction / "verification.json").read_text())
    assert result["full_epoch_count"] == 0 and result["unknown_seconds"] == 29.99


def test_verify_rejects_different_executing_source(tmp_path, monkeypatch):
    prediction = zero_epoch_prediction(tmp_path, monkeypatch)
    record_path = prediction / "stage_result.json"
    record = json.loads(record_path.read_text())
    record["executing_source_sha256"]["src/predict.py"] = "0" * 64
    common.write_json(record_path, record)
    assert invoke(monkeypatch, verify, ["--prediction-dir", prediction]) == 2
    assert "source differs" in json.loads((prediction / "verification.json").read_text())["message"]
