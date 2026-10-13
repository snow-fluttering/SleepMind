"""Independent review regressions for continuation, ensemble and protocol integrity."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pickle

import numpy as np
import pytest
import torch

from config import get_config
import predict
import reporting
import splits
import train


def _saved_contract(tmp_path):
    cfg = get_config()
    args = SimpleNamespace(ema_decay=0.0)
    split = {'manifest_sha256': 'frozen', 'fit_partitions': 'train', 'fit_participants': ['SC901'],
             'train_participants': ['SC901'], 'validation_participants': ['SC902'],
             'fit_record_runs': ['SC9011E0::segment_0'], 'validation_record_runs': ['SC9021E0::segment_0']}
    snapshot = {key: getattr(cfg, key) for key in train.CONFIG_KEYS}
    snapshot.update(processing_parameters=train.preprocessing_parameters(cfg), ema_decay=0.0)
    (tmp_path / 'config.json').write_text(json.dumps(snapshot), encoding='utf-8')
    (tmp_path / 'split.json').write_text(json.dumps(split), encoding='utf-8')
    (tmp_path / 'data_manifest.json').write_text('{"files": []}', encoding='utf-8')
    np.savez(tmp_path / 'norm_stats.npz', mean=1e-6, std=2e-5)
    return cfg, args, split


def test_resume_rejects_processing_split_and_normalization_changes(tmp_path):
    cfg, args, split = _saved_contract(tmp_path)
    train.validate_resume_contract(cfg, args, tmp_path, split, 1e-6, 2e-5)
    cfg.NUM_EPOCHS += 5  # Extending a pass budget preserves learned-data inputs.
    train.validate_resume_contract(cfg, args, tmp_path, split, 1e-6, 2e-5)
    cfg.INPUT_MODE = 'raw'
    with pytest.raises(ValueError, match='processing_parameters'):
        train.validate_resume_contract(cfg, args, tmp_path, split, 1e-6, 2e-5)
    cfg.INPUT_MODE = 'preprocessed'
    changed = copy.deepcopy(split)
    changed['fit_participants'] = ['SC902']
    with pytest.raises(ValueError, match='split.fit_participants'):
        train.validate_resume_contract(cfg, args, tmp_path, changed, 1e-6, 2e-5)
    with pytest.raises(ValueError, match='normalization_statistics'):
        train.validate_resume_contract(cfg, args, tmp_path, split, 1e-6, 3e-5)


def test_resume_rejects_changed_labels_even_if_eeg_statistics_are_unchanged(tmp_path):
    cfg, args, split = _saved_contract(tmp_path)
    files = [{'record': 'SC9011E0', 'part': 1, 'path': '/previous/data/labels.txt',
              'size': 12, 'sha256': 'a' * 64}]
    (tmp_path / 'data_manifest.json').write_text(json.dumps({'files': files}), encoding='utf-8')
    cache = tmp_path / 'cache.npz'
    current = copy.deepcopy(files)
    current[0]['path'] = '/relocated/data/labels.txt'
    np.savez(cache, metadata=json.dumps({'files': current}))
    train.validate_resume_contract(cfg, args, tmp_path, split, 1e-6, 2e-5, cache)
    current[0]['sha256'] = 'b' * 64
    np.savez(cache, metadata=json.dumps({'files': current}))
    with pytest.raises(ValueError, match='dataset_content'):
        train.validate_resume_contract(cfg, args, tmp_path, split, 1e-6, 2e-5, cache)


def test_extended_cosine_schedule_restores_nonzero_learning_rate_and_updates_weights():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    optimizer.zero_grad()
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]['lr'] == 0.0
    continuation = train.extend_cosine_schedule(scheduler, optimizer, total_passes=3)
    assert continuation['schedule_extended'] and optimizer.param_groups[0]['lr'] == pytest.approx(0.075)
    before = model.weight.detach().clone()
    optimizer.zero_grad()
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    assert not torch.equal(before, model.weight)
    scheduler.step()
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.025)


def _write_marker(path):
    Path(path).write_text('executed', encoding='ascii')


class _UnsafePickle:
    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return _write_marker, (str(self.path),)


def test_training_resume_blocks_pickle_execution(tmp_path, monkeypatch):
    cfg = get_config()
    cfg.TRAIN_DIR, cfg.RUNS_DIR, cfg.CACHE_DIR = str(tmp_path), str(tmp_path / 'runs'), str(tmp_path / 'cache')
    cfg.FEATURE_DIM, cfg.LSTM_HIDDEN, cfg.TRANSFORMER_LAYERS, cfg.TRANSFORMER_HEADS = 16, 8, 1, 2
    cfg.DEVICE = torch.device('cpu')
    monkeypatch.setattr(train, 'get_config', lambda: cfg)
    monkeypatch.setattr('data_preprocessing.scan_dataset', lambda _: {'SC9011E0': {}, 'SC9021E0': {}})
    rng = np.random.default_rng(5)
    data = rng.normal(size=(12, 3000)).astype(np.float32)
    ids = np.array(['SC9011E0::segment_0'] * 6 + ['SC9021E0::segment_0'] * 6)
    monkeypatch.setattr(train, 'build_training_set', lambda *a, **kw: (data, np.arange(12) % 5, ids))
    monkeypatch.setattr(train, 'plot_training_curve', lambda *args: None)
    monkeypatch.setattr(train, 'environment', lambda: {})
    argv = ['--epochs', '1', '--seq_len', '3', '--batch_size', '2', '--threads', '1',
            '--run_name', 'safe', '--no_cache', '--no_augment']
    train.main(argv)
    marker = tmp_path / 'marker.txt'
    torch.save({'model': _UnsafePickle(marker)}, tmp_path / 'runs' / 'safe' / 'last_state.pt')
    with pytest.raises(pickle.UnpicklingError):
        train.main([*argv, '--resume'])
    assert not marker.exists()


@pytest.mark.parametrize('members', [[], ['m', 'm'], ['../m'], ['bundle']])
def test_ensemble_rejects_empty_duplicate_or_nonlocal_members(tmp_path, members):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'ensemble.json').write_text(json.dumps({'members': members}), encoding='utf-8')
    with pytest.raises(ValueError, match='ensemble|Ensemble'):
        predict.ensemble_members(bundle)


def test_ensemble_refuses_member_checkpoint_replacement(tmp_path):
    member = tmp_path / 'm'
    member.mkdir()
    (member / 'best_model.pth').write_bytes(b'changed weights')
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'ensemble.json').write_text(json.dumps({'members': ['m'],
              'member_checkpoint_sha256': [hashlib.sha256(b'original weights').hexdigest()]}), encoding='utf-8')
    with pytest.raises(ValueError, match='declared hash'):
        predict.load_model(bundle)


def test_bound_protocol_refuses_changed_preprocessing_and_inference_settings():
    snapshot = {'SEQ_LEN': 20, 'processing_parameters': {'lowcut': 0.5, 'highcut': 30.0}}
    protocol = {'inference_contract': {'input_mode': 'raw', 'sequence_length': 20,
                'sequence_stride': 10, 'tta': 1, 'processing_parameters': snapshot['processing_parameters']}}
    assert predict.validate_protocol_contract(protocol, snapshot, 'raw', 1, None)
    assert not predict.validate_protocol_contract({}, snapshot, 'raw', 1, None)
    with pytest.raises(ValueError, match='input_mode'):
        predict.validate_protocol_contract(protocol, snapshot, 'preprocessed', 1, None)
    with pytest.raises(ValueError, match='tta'):
        predict.validate_protocol_contract(protocol, snapshot, 'raw', 3, None)
    with pytest.raises(ValueError, match='sequence_stride'):
        predict.validate_protocol_contract(protocol, snapshot, 'raw', 1, 5)
    with pytest.raises(ValueError, match='processing_parameters'):
        predict.validate_protocol_contract(protocol, {**snapshot, 'processing_parameters': {'lowcut': 1.0}}, 'raw', 1, None)


@pytest.mark.parametrize('changed_artifact', ['normalization', 'transition'])
def test_predict_refuses_replaced_protocol_bound_artifacts(tmp_path, monkeypatch, changed_artifact):
    cfg = get_config()
    cfg.RUNS_DIR, cfg.DEVICE = str(tmp_path / 'runs'), torch.device('cpu')
    run = tmp_path / 'runs' / 'bound'
    run.mkdir(parents=True)
    (run / 'best_model.pth').write_bytes(b'weights')
    (run / 'norm_stats.npz').write_bytes(b'normalization')
    transition = run / 'transition.json'
    transition.write_text(json.dumps(np.full((5, 5), 0.2).tolist()), encoding='utf-8')
    protocol = {'checkpoint_sha256': hashlib.sha256(b'weights').hexdigest(),
                'selected_variant': 'viterbi_learned_s1', 'selected_prior': 'learned', 'selected_strength': 1.0,
                'temperature': 1.0, 'abstention': {'confidence_threshold': None},
                'normalization_sha256': hashlib.sha256(b'normalization').hexdigest(),
                'transition_sha256': hashlib.sha256(transition.read_bytes()).hexdigest()}
    protocol[f'{changed_artifact}_sha256'] = '0' * 64
    frozen = tmp_path / 'frozen.json'
    frozen.write_text(json.dumps(protocol), encoding='utf-8')
    monkeypatch.setattr(predict, 'get_config', lambda: cfg)
    monkeypatch.setattr(predict, 'load_model', lambda *args: (torch.nn.Identity(), 0.0, 1.0,
        {'NORMALIZATION_EPSILON': 0.0, 'IMPORTED_LEGACY_CHECKPOINT': False}))
    monkeypatch.setattr(predict, 'scan_dataset', lambda *args: {'SC9011E0': {}})
    monkeypatch.setattr('sys.argv', ['predict.py', '--run_name', 'bound', '--protocol', str(frozen),
                                   '--output_dir', str(tmp_path / 'out')])
    with pytest.raises(SystemExit):
        predict.main()


def test_fold_calibration_transfer_does_not_claim_validated_ensemble_confidence():
    protocol = {'calibration_prediction_scope': 'pooled_fold_single_models'}
    confidence = predict.confidence_summary(np.array([0.7, 0.9]), np.ones(2, bool), 1.7, 0.8,
                                           protocol, is_ensemble=True)
    assert confidence['temperature_fit_applied'] and not confidence['calibrated']
    assert confidence['calibration_transferred_to_ensemble']
    assert 'not independently validated' in confidence['note']
    segments = [{'onset_sec': 0.0, 'end_sec': 60.0, 'label': 2}]
    summary = reporting.summarize(segments, {'total_seconds': 60.0, 'confidence': confidence})
    assert any('transferred from fold-model' in note for note in summary['limitations'])
    protocol['calibration_prediction_scope'] = 'deployed_ensemble'
    assert predict.confidence_summary(np.array([0.8]), np.ones(1, bool), 1.7, None, protocol, True)['calibrated']


def test_manifest_rejects_inconsistent_participant_owner_table():
    manifest = splits.make_split([{'record_id': 'SC9011E0', 'epoch_counts': [1] * 5},
                                 {'record_id': 'SC9021E0', 'epoch_counts': [1] * 5}], 2)
    manifest.pop('manifest_sha256')
    manifest['participants']['SC901']['partition'] = 'holdout'
    with pytest.raises(ValueError, match='different partition'):
        splits.verify_manifest(manifest)


def test_inventory_stage_counts_follow_names_instead_of_dictionary_order(tmp_path):
    counts = dict(REM=50, N3=40, Wake=10, N2=30, N1=20)
    (tmp_path / 'inventory.json').write_text(json.dumps({'records': [{'mode': 'raw', 'split': 'train',
        'record_id': 'SC9011E0', 'full_valid_epoch_counts': counts}]}), encoding='utf-8')
    assert splits.records_from_inventory(tmp_path / 'inventory.json')[0]['epoch_counts'] == [10, 20, 30, 40, 50]
