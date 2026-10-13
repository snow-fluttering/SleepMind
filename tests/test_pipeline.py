"""Read-to-report integration: raw parts -> trained tiny checkpoint -> hypnogram, NPZ, JSON, HTML, PNG."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from config import get_config
import predict
import train
from utils import evaluate

RATE, EPOCH = 100, 3000


def _write_raw_parts(folder, sid, seed, labels=True):
    """Two raw parts joined mid-epoch (part 1 ends 10 s into epoch 4), with a flat stretch."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    t = np.arange(12 * EPOCH) / RATE
    signal = 2e-5 * np.sin(2 * np.pi * 2 * t) + 5e-6 * rng.normal(size=t.size)
    signal[5 * EPOCH:9 * EPOCH] = 0.0  # flat stretch (the filter rings into its edges) must stay unscorable
    cut = 4 * EPOCH + 1000
    for number, chunk in ((1, signal[:cut]), (2, signal[cut:])):
        np.savetxt(folder / f'{sid}_EEGFpz-Cz_Part_{number}_of_2.txt', chunk, fmt='%.9g')
    if labels:
        stages = ['Sleep stage W', 'Sleep stage 1', 'Sleep stage 2', 'Sleep stage 3', 'Sleep stage R']
        rows = [(30 * i, 30 * i + 30, stages[(i // 2) % 5]) for i in range(12)]
        for number, (low, high) in ((1, (0, cut / RATE)), (2, (cut / RATE, 360))):
            lines = [f'{a} {b} {b - a} {d}' for a, b, d in rows if b > low and a < high]
            (folder / f'{sid}_Hypnogram_Data_Part_{number}_of_2.txt').write_text(
                'onset end duration description\n' + '\n'.join(lines) + '\n', encoding='utf-8')


def test_raw_input_to_consistent_report_artifacts(tmp_path, monkeypatch):
    train_dir, test_dir = tmp_path / 'train', tmp_path / 'test'
    for index, sid in enumerate(['SC9011E0', 'SC9021E0', 'SC9031E0']):
        _write_raw_parts(train_dir, sid, index)
    _write_raw_parts(test_dir, 'SC01', 9, labels=False)
    cfg = get_config()
    cfg.RAW_TRAIN_DIR, cfg.RUNS_DIR, cfg.CACHE_DIR = str(train_dir), str(tmp_path / 'runs'), str(tmp_path / 'cache')
    cfg.DEVICE = torch.device('cpu')
    monkeypatch.setattr(train, 'get_config', lambda: cfg)
    monkeypatch.setattr(predict, 'get_config', lambda: cfg)
    train.main(['--input_mode', 'raw', '--epochs', '1', '--batch_size', '2', '--seq_len', '3', '--threads', '2',
                '--val_ratio', '0.4', '--run_name', 'tiny', '--no_augment'])
    run = tmp_path / 'runs' / 'tiny'
    split = json.loads((run / 'split.json').read_text())
    assert set(split['train_participants']).isdisjoint(split['validation_participants'])
    checkpoint_hash = hashlib.sha256((run / 'best_model.pth').read_bytes()).hexdigest()
    protocol = {'checkpoint_sha256': checkpoint_hash, 'selected_variant': 'viterbi_learned_s1',
                'selected_prior': 'learned', 'selected_strength': 1.0, 'temperature': 1.7,
                'abstention': {'confidence_threshold': 0.5}}
    (tmp_path / 'protocol.json').write_text(json.dumps(protocol))
    out = tmp_path / 'out'
    argv = ['--run_name', 'tiny', '--input_mode', 'raw', '--test_dir', str(test_dir), '--output_dir', str(out),
            '--threads', '2', '--protocol', str(tmp_path / 'protocol.json')]
    monkeypatch.setattr('sys.argv', ['predict.py', *argv])
    predict.main()
    for name in ('SC01_report.json', 'SC01_report.html', 'SC01_hypnogram.png', 'SC01_hypnogram.tsv', 'SC01_epochs.npz'):
        assert (out / name).is_file(), name
    report = json.loads((out / 'SC01_report.json').read_text())
    arrays = np.load(out / 'SC01_epochs.npz')
    total = 12 * EPOCH / RATE
    assert report['recording_seconds'] == total
    assert len(arrays['decoded']) == 12 and np.array_equal(arrays['onset_sec'], np.arange(12) * 30.0)
    # The flat epoch and the part-cut epoch are handled explicitly rather than dropped or labeled Wake.
    assert not arrays['scorable'][7] and arrays['decoded'][7] == -1 and np.isnan(arrays['confidence'][7])
    assert arrays['scorable'][4] and arrays['decoded'][4] >= 0  # epoch crossing the part boundary is scored
    minutes = sum(report['stage_minutes'].values()) * 60 + report['unscorable_seconds']
    assert minutes == pytest.approx(total)
    assert report['unscorable_seconds'] == pytest.approx(30.0 * (~arrays['scorable']).sum()) and not arrays['scorable'].all()
    rows = [line.split('\t') for line in (out / 'SC01_hypnogram.tsv').read_text().splitlines()[1:]]
    assert float(rows[0][0]) == 0 and float(rows[-1][1]) == total
    assert all(abs(float(a[1]) - float(b[0])) < 1e-6 for a, b in zip(rows, rows[1:]))
    assert report['metadata']['checkpoint_sha256'] == checkpoint_hash
    assert report['metadata']['decoder'] == 'viterbi' and report['metadata']['transition_strength'] == 1.0
    assert report['confidence']['calibrated'] and report['confidence']['temperature'] == 1.7
    assert 'Low-confidence epochs' in (out / 'SC01_report.html').read_text()
    assert report['clinical_sleep_efficiency'] is None
    # A protocol for another checkpoint is refused.
    protocol['checkpoint_sha256'] = '0' * 64
    (tmp_path / 'protocol.json').write_text(json.dumps(protocol))
    with pytest.raises(SystemExit):
        predict.main()
    # Wake-only references are never turned into sleep-stage accuracy.
    assert evaluate(out / 'SC01_epochs.npz')['decoded']['annotation_scope'].startswith('No recognized')
