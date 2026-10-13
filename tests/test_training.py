"""Training regressions: participant isolation, train-only statistics, lazy windows, real optimization."""
import json

import numpy as np
import pytest
import torch

from config import get_config
from data_preprocessing import participant_id
from dataset import EpochBank, augment_batch, iter_batches, run_table, window_starts
from model import SeqSleepStagingModel, FocalLoss
import train


def test_validation_split_keeps_repeat_nights_together():
    participants = [participant_id(s) for s in ['SC9011E0', 'SC9012E0', 'ST8081J0', 'ST8082J0', 'SC9021E0']]
    fit, validation = train.participant_validation_split(participants, val_ratio=0.5, seed=1)
    assert set(fit).isdisjoint(validation) and set(fit) | set(validation) == set(participants)
    with pytest.raises(ValueError, match='at least two participants'):
        train.participant_validation_split(['SC901', 'SC901'])


def test_normalization_uses_unique_training_epochs_only():
    # Validation has a deliberately extreme amplitude and must not affect stats.
    X = np.array([[0, 2], [2, 4], [1000, 2000]], dtype=np.float32)
    y = np.array([0, 1, 4])
    sids = np.array(['SC9011E0::segment_0'] * 2 + ['SC9021E0::segment_0'])
    mean, std, labels, n = train.training_epoch_statistics(X, y, sids, {'SC901'})
    assert mean == pytest.approx(X[:2].mean())
    assert std == pytest.approx(X[:2].std())
    np.testing.assert_array_equal(labels, [0, 1])
    assert n == 2
    with pytest.raises(ValueError, match='zero or invalid variance'):
        train.training_epoch_statistics(np.ones_like(X), y, sids, {'SC901'})


def test_class_weights_and_transitions_are_train_only_and_gap_aware():
    labels = np.array([0, 1, 4, 4, 4, 4])
    sids = np.array(['SC9011E0::segment_0', 'SC9011E0::segment_0', 'SC9011E0::segment_1',
                     'SC9021E0::segment_0', 'SC9021E0::segment_0', 'SC9021E0::segment_0'])
    transition = train.training_only_transition(labels, sids, {'SC901'})
    # Only 0 -> 1 is observed: 1 -> 4 crosses a gap and 4 -> 4 is validation.
    assert transition[0, 1] == pytest.approx(2 / 6)
    np.testing.assert_allclose(transition[1:], np.full((4, 5), 0.2))
    weights = train.compute_class_weights(np.array([0, 0, 0, 1]), 5)
    assert weights[1] > weights[0] and float(weights.sum()) == pytest.approx(5)


def test_run_table_and_windows_never_cross_runs():
    run_ids = np.array(['a'] * 7 + ['b'] * 3 + ['c'] * 12)
    starts, lengths = run_table(run_ids)
    np.testing.assert_array_equal(starts, [0, 7, 10])
    np.testing.assert_array_equal(lengths, [7, 3, 12])
    for seed in range(5):
        windows = window_starts(starts, lengths, 4, rng=np.random.default_rng(seed))
        for w in windows:
            assert len(set(run_ids[w:w + 4])) == 1
        assert len(set(run_ids[windows])) == 2  # run b (3 epochs) is shorter than 4
    grid = window_starts(starts, lengths, 4, stride=2)
    np.testing.assert_array_equal(grid, [0, 2, 10, 12, 14, 16, 18])


def test_lazy_windows_share_one_bank_and_do_not_mutate_it():
    X = np.arange(10 * 6, dtype=np.float32).reshape(10, 6)
    y = np.arange(10) % 5
    original = X.copy()
    bank = EpochBank(X, y, mean=3.0, std=2.0)
    x, labels = bank.batch(np.array([0, 5]), 3)
    assert x.shape == (2, 3, 6) and labels.shape == (2, 3)
    np.testing.assert_allclose(x[1, 0].numpy(), (X[5] - 3) / 2)
    first = augment_batch(x, torch.Generator().manual_seed(1))
    second = augment_batch(x, torch.Generator().manual_seed(1))
    torch.testing.assert_close(first, second)  # seeded augmentation is reproducible
    np.testing.assert_array_equal(X, original)
    assert bank.X is X  # no copy of the epoch matrix
    batches = list(iter_batches(bank, np.array([0, 2, 4, 6]), 2, 3, order=np.array([3, 2, 1, 0])))
    assert [len(b[0]) for b in batches] == [3, 1]
    with pytest.raises(ValueError):
        EpochBank(X, y, 0.0, 0.0)


def test_real_model_trains_on_lazy_windows_and_validates_unique_epochs():
    torch.set_num_threads(2)
    train.set_seed(42)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 3000)).astype(np.float32)
    y = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1])
    bank = EpochBank(X, y, 0.0, 1.0)
    model = SeqSleepStagingModel(feat_dim=16, lstm_hidden=8, n_trans_layers=1, n_heads=2)
    before = model.classifier[-1].weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    starts = window_starts(np.array([0]), np.array([12]), 4, rng=np.random.default_rng(1))
    loss, count = train.train_pass(model, bank, starts, 4, 2, optimizer, FocalLoss(),
                                   torch.Generator().manual_seed(0), np.random.default_rng(0), True)
    assert np.isfinite(loss) and count == len(starts)
    assert not torch.equal(before, model.classifier[-1].weight)
    result = train.validate_runs(model, bank, np.array([0]), np.array([12]), 4, 'cpu', 5)
    assert result['total_epoch_positions'] == 12 and np.isfinite(result['negative_log_likelihood'])
    with pytest.raises(ValueError, match='No training windows'):
        train.train_pass(model, bank, np.empty(0, np.int64), 4, 2, optimizer, FocalLoss(),
                         torch.Generator().manual_seed(0), np.random.default_rng(0), False)


def _write_records(data_dir, ids, minutes=6):
    samples = np.arange(minutes * 6000) / 100
    stages = ['Sleep stage W', 'Sleep stage 1', 'Sleep stage 2', 'Sleep stage 3', 'Sleep stage R']
    for index, sid in enumerate(ids):
        eeg = (1e-5 * np.sin(2 * np.pi * (index + 3) * samples) + 2e-6 * np.cos(2 * np.pi * 0.8 * samples))
        np.savetxt(data_dir / f'{sid}_EEGFpz-Cz_Part_1_of_1.txt', eeg)
        rows = '\n'.join(f'{30 * i} {30 * i + 30} 30 {stages[(i // 3) % 5]}' for i in range(minutes * 2))
        (data_dir / f'{sid}_Hypnogram_Data_Part_1_of_1.txt').write_text(
            'onset end duration description\n' + rows + '\n', encoding='utf-8')


def test_training_cli_uses_manifest_without_loading_holdout(tmp_path, monkeypatch):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    ids = ['SC9011E0', 'SC9012E0', 'SC9021E0', 'SC9031E0', 'SC9041E0']
    _write_records(data_dir, ids)
    from splits import make_split
    manifest = make_split([{'record_id': r, 'epoch_counts': [3, 3, 3, 3, 3]} for r in ids], seed=1,
                          validation={'SC': 1}, holdout_extra={'SC': 1}, trials=5)
    manifest_path = tmp_path / 'split.json'
    manifest_path.write_text(json.dumps(manifest), encoding='ascii')
    cfg = get_config()
    cfg.TRAIN_DIR, cfg.RUNS_DIR, cfg.CACHE_DIR = str(data_dir), str(tmp_path / 'runs'), str(tmp_path / 'cache')
    cfg.DEVICE = torch.device('cpu')
    monkeypatch.setattr(train, 'get_config', lambda: cfg)
    argv = ['--epochs', '2', '--batch_size', '2', '--seq_len', '3', '--threads', '2', '--no_focal',
            '--no_augment', '--run_name', 'smoke', '--split_manifest', str(manifest_path), '--ema_decay', '0.9']
    train.main(argv)
    run = tmp_path / 'runs' / 'smoke'
    snapshot = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    split = json.loads((run / 'split.json').read_text(encoding='utf-8'))
    held_out = set(manifest['partitions']['holdout']['records'])
    loaded = {r.split('::')[0] for r in split['fit_record_runs'] + split['validation_record_runs']}
    assert held_out and not held_out & loaded
    assert split['holdout_loaded'] is False
    assert set(snapshot['train_subjects']).isdisjoint(snapshot['val_subjects'])
    assert snapshot['initialization'].startswith('fresh random')
    assert snapshot['split_manifest_sha256'] == manifest['manifest_sha256']
    assert snapshot['environment']['torch'] and snapshot['INPUT_MODE'] == 'preprocessed'
    assert (run / 'best_model.pth').is_file() and (run / 'run_info.json').is_file()
    matrix = np.asarray(json.loads((run / 'transition.json').read_text()))
    np.testing.assert_allclose(matrix.sum(axis=1), 1)
    log = json.loads((run / 'train_log.json').read_text())
    assert len(log['val_macro_f1']) == 2 and len(log['pass_seconds']) == 2
    # Resume continues the same run and keeps its artifacts; a reused name is rejected.
    train.main(['--epochs', '3', '--batch_size', '2', '--seq_len', '3', '--threads', '2', '--no_focal',
                '--no_augment', '--run_name', 'smoke', '--resume', '--split_manifest', str(manifest_path),
                '--ema_decay', '0.9'])
    assert len(json.loads((run / 'train_log.json').read_text())['train_loss']) == 3
    with pytest.raises(FileExistsError):
        train.main(['--epochs', '1', '--run_name', 'smoke', '--split_manifest', str(manifest_path)])


def test_weight_average_follows_decay_and_copies_integer_buffers():
    model = torch.nn.BatchNorm1d(3)
    ema = torch.nn.BatchNorm1d(3)
    with torch.no_grad():
        model.weight.fill_(2.0)
        ema.weight.fill_(1.0)
        model.num_batches_tracked.fill_(7)
    train.ema_update(ema, model, 0.75)
    torch.testing.assert_close(ema.weight, torch.full((3,), 1.25))
    assert int(ema.num_batches_tracked) == 7
