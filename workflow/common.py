"""Shared helpers for the local BCI-to-SleepMind demonstration workflow."""

from __future__ import annotations

import json
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VERSION_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = VERSION_ROOT
EXPECTED_MEMBERS = [f"cv_CE_f{index}" for index in range(5)]
EXPECTED_CHECKPOINT_SHA256 = "7ede711f95049f9eb703eb3917f03fdf87c7e817280d762fdf2e1b39c13fe68f"
EXPECTED_PROTOCOL_SHA256 = "c94ccb7cef124c5391ed00d4739501d82743a8bb7056d466c817cb59059c5c85"
EXPECTED_ASSET_HASHES = {
    "reports/ens_CE/frozen_protocol.json": "c94ccb7cef124c5391ed00d4739501d82743a8bb7056d466c817cb59059c5c85",
    "runs/cv_CE_f0/best_model.pth": "67cdcb6e865a802abeac2e7210e74559c02b6531460a122ffeff98ee976dd006",
    "runs/cv_CE_f0/config.json": "2a2eb20f076b980a183d505c2e714add64b51531e1964ec2bd9b996f6496cb47",
    "runs/cv_CE_f0/norm_stats.npz": "9dfe71e1324dfdc10ba2b389aa56287b74d0737c40f0f3e0d1662c3195483225",
    "runs/cv_CE_f1/best_model.pth": "bebd44bc56d20894aaee494a352578d1c6e013bd07e80934eb1714bd2ac2d4a8",
    "runs/cv_CE_f1/config.json": "8e78ad1d25c68c3d807f4966a4657e2a2306c497a6bfceabaff9067c4848436b",
    "runs/cv_CE_f1/norm_stats.npz": "81b41cb0c599682010803fc742f5ca40bf2925a49cb01b1cf9a7cbbbf38f9352",
    "runs/cv_CE_f2/best_model.pth": "a847a41680460f4c9d0f9b59de2eef03a55c921fa8a76ff04450ee9f4a14a88f",
    "runs/cv_CE_f2/config.json": "5ba4c448a4b28fc842d01504e806a23d196f3d42d9145f3bce91203b8fafe40d",
    "runs/cv_CE_f2/norm_stats.npz": "f0a1c0d15d6b163bee017d8ed41384f6a8000e5160ad6fc64c5b2aa28a86eec4",
    "runs/cv_CE_f3/best_model.pth": "3515f1aca652e9e9adecf92e68b08c6c4548087f36490f025437163e03de1f5a",
    "runs/cv_CE_f3/config.json": "5c86c8f4c2da4413081f734de8f34e6da2a4d2a00b8ace1401229bb81617ce29",
    "runs/cv_CE_f3/norm_stats.npz": "196385e3fe3b3d2141df21e711916f723d1973eb40e08bc23a672d560ac8c139",
    "runs/cv_CE_f4/best_model.pth": "028dcb86ae0d4cdcdebfa828e90398a0a07a15b4bd0af3cff85170abea1687b5",
    "runs/cv_CE_f4/config.json": "722fe13d4d5bb42647ad15abc0e10999fe2a7205e7c16774a3803c13a96d09fd",
    "runs/cv_CE_f4/norm_stats.npz": "a04a813cbb10e00097b121bdab3c7de957167951d6e0477af625b5410512440e",
    "runs/ens_CE/config.json": "82702e58733676293dd1dffc71d3ca301fad67a00545395d0fe6681d79006a85",
    "runs/ens_CE/ensemble.json": "409a47fe206e813e1f59e315d1d289de254233a2fc7abb7b5a17b8aa9eb3b9ff",
    "runs/ens_CE/norm_stats.npz": "9dfe71e1324dfdc10ba2b389aa56287b74d0737c40f0f3e0d1662c3195483225",
    "runs/ens_CE/transition.json": "b864b033f1e8ccae03228342575f842bd0bfa9169768b857248aad0af7460118"
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def asset_root(value: str | Path | None = None) -> Path:
    selected = value if value is not None else os.environ.get("SLEEPMIND_ASSET_ROOT")
    if not selected:
        raise ValueError("Set SLEEPMIND_ASSET_ROOT to a private directory containing the matched runs/ and reports/ assets. No checkpoints are included in this source repository.")
    root = Path(selected).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Private trained-asset directory does not exist: {root}")
    return root


def require_received_integrity(record: dict[str, Any], path: Path) -> str:
    """Require the exact finalized bytes recorded by the reviewed transfer."""
    expected = record.get("received_eeg_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Reviewed transfer received_eeg_sha256 is missing or invalid.")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError("Received EEG contents changed after the successful transfer.")
    return actual


def executing_source_hashes() -> dict[str, str]:
    """Record executing repository code independently of private trained assets."""
    paths = []
    for folder in ("src", "workflow", "bci_replay"):
        paths.extend((CODE_ROOT / folder).glob("*.py"))
    paths.extend(CODE_ROOT / name for name in ("run_action.py", "setup_env.py", "health_check.py"))
    return {path.relative_to(CODE_ROOT).as_posix(): sha256_file(path) for path in sorted(paths)}


def model_integrity(value: str | Path | None = None) -> dict[str, Any]:
    """Check exact trained assets; source provenance is recorded separately."""
    root = asset_root(value)
    actual = {}
    for relative, expected in EXPECTED_ASSET_HASHES.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"Required private trained asset is missing: {relative}. Supply the matched model assets separately.")
        actual[relative] = sha256_file(path)
        if actual[relative] != expected:
            raise ValueError(f"Private trained asset differs from its frozen hash: {relative}")
    ensemble = json.loads((root / "runs/ens_CE/ensemble.json").read_text(encoding="utf-8"))
    if ensemble.get("members") != EXPECTED_MEMBERS or ensemble.get("aggregation") != "mean_logits":
        raise ValueError("The frozen five-member mean-logit ensemble changed.")
    member_hashes = [actual[f"runs/{member}/best_model.pth"] for member in EXPECTED_MEMBERS]
    checkpoint_hash = hashlib.sha256("\n".join(member_hashes).encode("ascii")).hexdigest()
    if (member_hashes != ensemble.get("member_checkpoint_sha256")
            or checkpoint_hash != ensemble.get("bundle_checkpoint_sha256")
            or checkpoint_hash != EXPECTED_CHECKPOINT_SHA256):
        raise ValueError("The frozen ensemble checkpoint hashes changed.")
    protocol = json.loads((root / "reports/ens_CE/frozen_protocol.json").read_text(encoding="utf-8"))
    protocol_hash = actual["reports/ens_CE/frozen_protocol.json"]
    if (protocol_hash != EXPECTED_PROTOCOL_SHA256 or protocol.get("checkpoint_sha256") != checkpoint_hash
            or protocol.get("selected_variant") != "viterbi_learned_s0.25"
            or protocol.get("selected_prior") != "learned" or protocol.get("selected_strength") != 0.25):
        raise ValueError("The frozen learned Viterbi protocol changed.")
    return {
        "asset_root": str(root),
        "asset_sha256": hashlib.sha256(json.dumps(actual, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest(),
        "asset_file_sha256": actual,
        "checkpoint_sha256": checkpoint_hash,
        "protocol_sha256": protocol_hash,
        "transition_sha256": actual["runs/ens_CE/transition.json"],
        "normalization_sha256": actual["runs/ens_CE/norm_stats.npz"],
        "scope": "Fixed trained-file integrity and consistency; not a signature or authentication.",
    }


def child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def emit_json(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=True, separators=(",", ":")), flush=True)


def next_action(action: str, args: list[str]) -> dict[str, Any]:
    entry = VERSION_ROOT / "workflow" / "entry.py"
    return {
        "action": action,
        "args": args,
        "command": [sys.executable, str(entry), action, *args],
    }


def emit_result(
    action: str,
    result_path: Path | None,
    artifacts: dict[str, Path],
    summary: str,
    error: dict[str, Any] | None = None,
    following: dict[str, Any] | None = None,
) -> None:
    envelope = {
        "schema_version": "1.0",
        "action": action,
        "status": "error" if error else "success",
        "result_path": str(result_path.resolve()) if result_path is not None else None,
        "artifacts": {name: str(path.resolve()) for name, path in artifacts.items()},
        "summary": summary,
        "error": error,
        "next_action": following,
    }
    emit_json(envelope)


def error_details(record: dict[str, Any]) -> dict[str, Any] | None:
    if record.get("status") == "success" or record.get("status") == "pass":
        return None
    return {
        "code": record.get("error_code"),
        "root_cause": record.get("root_cause") or record.get("message") or "Unspecified workflow error.",
        "secondary": record.get("secondary_errors", []),
    }
