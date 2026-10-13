"""Reject stale, switched and contaminated inference evidence before claiming model quality."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import cv_summary
import evaluate_run
from data_preprocessing import participant_id
from decoding import default_transition
from splits import build_manifest, participant_table


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def manifest_file(path, groups):
    records = [r for rows in groups.values() for r in rows]
    table = participant_table([{'record_id': r, 'epoch_counts': [2] * 5} for r in records])
    assignment = {participant_id(r): partition for partition, rows in groups.items() for r in rows}
    manifest = build_manifest(table, assignment, {})
    write_json(path, manifest)
    return manifest


def make_run(path, manifest_path, manifest):
    path.mkdir(parents=True)
    (path / 'best_model.pth').write_bytes(path.name.encode('ascii'))
    (path / 'norm_stats.npz').write_bytes(b'normalization')
    config = {'SEQ_LEN': 20, 'processing_parameters': {},
              'fit_subjects': manifest['partitions']['train']['participants'],
              'split_manifest': str(manifest_path), 'split_manifest_sha256': manifest['manifest_sha256']}
    write_json(path / 'config.json', config)
    write_json(path / 'transition.json', default_transition().tolist())
    return config


def save_epochs(directory, run, manifest_path, manifest, partition):
    directory.mkdir(parents=True, exist_ok=True)
    records = manifest['partitions'][partition]['records']
    for record in records:
        truth = np.arange(10) % 5
        logits = (np.eye(5)[truth] * 3).astype(np.float32)
        np.savez_compressed(directory / f'{record}_epochs.npz', onset_sec=np.arange(10) * 30.,
            ground_truth=truth, argmax=truth, decoded=truth, logits=logits, scorable=np.ones(10, bool))
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    info = {'checkpoint_sha256': evaluate_run.checkpoint_sha256(run), 'split_manifest': str(manifest_path),
        'split_manifest_sha256': manifest['manifest_sha256'], 'partition': partition, 'records': records,
        'input_mode': 'raw', 'sequence_length': 20, 'tta': 1,
        'inference_contract': evaluate_run.inference_contract(config),
        'normalization_sha256': evaluate_run.sha256(run / 'norm_stats.npz'),
        'epoch_sha256': {r: evaluate_run.sha256(directory / f'{r}_epochs.npz') for r in records}}
    write_json(directory / 'inference_info.json', info)
    return info


@pytest.fixture
def evidence(tmp_path):
    path = tmp_path / 'split.json'
    manifest = manifest_file(path, {'train': ['SC9001E0'], 'validation': ['SC9011E0'], 'holdout': ['ST8191J0']})
    run = tmp_path / 'run'
    make_run(run, path, manifest)
    val = tmp_path / 'validation'
    info = save_epochs(val, run, path, manifest, 'validation')
    return run, path, manifest, val, info


def test_inference_provenance_rejects_switched_checkpoint_and_modified_epochs(evidence):
    run, path, manifest, val, info = evidence
    verified, _ = evaluate_run.validated_inference_info(val, run, 'validation')
    assert verified['saved_epoch_hash_binding'] == 'verified'
    write_json(val / 'inference_info.json', info | {'checkpoint_sha256': 'different'})
    with pytest.raises(ValueError, match='different checkpoint'):
        evaluate_run.validated_inference_info(val, run, 'validation')
    write_json(val / 'inference_info.json', info)
    with (val / 'SC9011E0_epochs.npz').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='epoch hash mismatch'):
        evaluate_run.validated_inference_info(val, run, 'validation')


def test_inference_provenance_rejects_partition_processing_and_stale_records(evidence):
    run, path, manifest, val, info = evidence
    with pytest.raises(ValueError, match='partition must be holdout'):
        evaluate_run.validated_inference_info(val, run, 'holdout')
    altered = dict(info['inference_contract'], sequence_stride=20)
    write_json(val / 'inference_info.json', info | {'inference_contract': altered})
    with pytest.raises(ValueError, match='processing contract'):
        evaluate_run.validated_inference_info(val, run, 'validation')
    write_json(val / 'inference_info.json', info)
    (val / 'SC9051E0_epochs.npz').write_bytes(b'stale')
    with pytest.raises(ValueError, match='metadata record set'):
        evaluate_run.validated_inference_info(val, run, 'validation')


def test_processing_uses_checkpoint_filters_and_rejects_unsupported_transforms():
    processing = {'input_mode': 'raw', 'fs': 100, 'epoch_sec': 30, 'lowcut': 1.0, 'highcut': 25.,
                  'order': 6, 'enable_notch': True, 'notch_freq': 40., 'epoch_normalize': None}
    kwargs = evaluate_run.record_processing({'processing_parameters': processing})
    assert kwargs['lowcut'] == 1.0 and kwargs['order'] == 6 and kwargs['notch_freq'] == 40.
    assert 'epoch_sec' not in kwargs and 'epoch_normalize' not in kwargs
    with pytest.raises(ValueError, match='not supported'):
        evaluate_run.record_processing({'processing_parameters': processing | {'clip_value': 3}})


def make_folds(tmp_path, duplicate=False):
    runs = tmp_path / 'runs'
    recipe = tmp_path / 'cv' / 'X'
    for fold in range(2):
        val_record = 'SC9011E0' if fold == 0 or duplicate else 'SC9021E0'
        train_records = ['SC9001E0', 'SC9021E0' if val_record == 'SC9011E0' else 'SC9011E0']
        path = tmp_path / f'fold_{fold}.json'
        manifest = manifest_file(path, {'train': train_records, 'validation': [val_record], 'holdout': ['ST8191J0']})
        run = runs / f'cv_X_f{fold}'
        make_run(run, path, manifest)
        save_epochs(recipe / f'f{fold}_val_epochs', run, path, manifest, 'validation')
    return runs, recipe


def test_oof_pool_checks_each_fold_and_binds_record_sources(tmp_path):
    runs, recipe = make_folds(tmp_path)
    target = cv_summary.pool(recipe, runs, 'X', [0, 1])
    info = json.loads((target / 'inference_info.json').read_text(encoding='utf-8'))
    assert info['records'] == ['SC9011E0', 'SC9021E0']
    assert info['record_sources']['SC9011E0']['member'] == 'cv_X_f0'
    ensemble = runs / 'ensemble'
    ensemble.mkdir()
    write_json(ensemble / 'ensemble.json', {'members': ['cv_X_f0', 'cv_X_f1'],
        'member_checkpoint_sha256': [evaluate_run.checkpoint_sha256(runs / f'cv_X_f{i}') for i in range(2)]})
    config = json.loads((runs / 'cv_X_f0' / 'config.json').read_text(encoding='utf-8'))
    write_json(ensemble / 'config.json', config)
    validated, scope = evaluate_run.validated_validation_info(target, ensemble)
    assert scope == 'pooled_fold_single_models' and validated['saved_epoch_hash_binding'] == 'verified'
    (target / 'SC9071E0_epochs.npz').write_bytes(b'stale')
    with pytest.raises(ValueError, match='stale records'):
        cv_summary.pool(recipe, runs, 'X', [0, 1])


def test_oof_pool_rejects_duplicate_fold_records_before_overwriting(tmp_path):
    runs, recipe = make_folds(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match='Duplicate out-of-fold'):
        cv_summary.pool(recipe, runs, 'X', [0, 1])
    assert not (recipe / 'oof_epochs').exists()


def test_frozen_report_rejects_changed_inference_contract(evidence, tmp_path):
    run, path, manifest, val, info = evidence
    frozen_path = tmp_path / 'frozen.json'
    evaluate_run.cmd_freeze(type('A', (), {'run_dir': str(run), 'val_dir': str(val),
        'output': str(frozen_path), 'target_accuracy': .9}))
    protocol = json.loads(frozen_path.read_text(encoding='utf-8'))
    assert protocol['calibration_prediction_scope'] == 'deployed_model_validation'
    assert protocol['inference_contract'] == info['inference_contract']
    hold = tmp_path / 'hold'
    save_epochs(hold, run, path, manifest, 'holdout')
    protocol['inference_contract']['tta'] = 3
    write_json(frozen_path, protocol)
    with pytest.raises(ValueError, match='frozen inference contract'):
        evaluate_run.cmd_report(type('A', (), {'run_dir': str(run), 'holdout_dir': str(hold),
            'frozen': str(frozen_path), 'split_manifest': str(path), 'output': str(tmp_path / 'report.json'),
            'n_boot': 10, 'seed': 0}))


def test_original_unhashed_inference_remains_explicitly_unbound(evidence):
    run, path, manifest, val, info = evidence
    for field in ('epoch_sha256', 'inference_contract', 'normalization_sha256', 'split_manifest_sha256'):
        info.pop(field)
    write_json(val / 'inference_info.json', info)
    verified, _ = evaluate_run.validated_inference_info(val, run, 'validation')
    assert verified['saved_epoch_hash_binding'] == 'absent_in_original_metadata'
