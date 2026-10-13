"""Mean-logit ensemble: exact per-member normalization, shared encoding equivalence and bundle hashing."""
import hashlib
import json

import numpy as np
import pytest
import torch

import predict
from model import SeqSleepStagingModel

TINY = dict(feat_dim=16, lstm_hidden=8, n_trans_layers=1, n_heads=2)


def _member_dir(root, name, seed, mean, std):
    torch.manual_seed(seed)
    model = SeqSleepStagingModel(**TINY)
    folder = root / name
    folder.mkdir()
    torch.save(model.state_dict(), folder / 'best_model.pth')
    np.savez(folder / 'norm_stats.npz', mean=mean, std=std, input_mode='raw')
    (folder / 'config.json').write_text(json.dumps({
        'SAMPLING_RATE': 100, 'EPOCH_DURATION': 30, 'NUM_CLASSES': 5, 'FEATURE_DIM': 16, 'LSTM_HIDDEN': 8,
        'TRANSFORMER_LAYERS': 1, 'TRANSFORMER_HEADS': 2, 'SEQ_LEN': 4}))
    (folder / 'data_manifest.json').write_text('{}')
    return folder, model.eval()


def test_ensemble_equals_mean_of_members_with_their_own_normalization(tmp_path):
    torch.set_num_threads(1)
    a_dir, a = _member_dir(tmp_path, 'm0', 1, 1e-6, 2e-5)
    b_dir, b = _member_dir(tmp_path, 'm1', 2, -3e-6, 2.4e-5)
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'ensemble.json').write_text(json.dumps({'members': ['m0', 'm1']}))
    (bundle / 'config.json').write_text((a_dir / 'config.json').read_text())
    ensemble, mean, std, config = predict.load_model(bundle)
    assert (mean, std) == (1e-6, 2e-5) and config['NORMALIZATION_EPSILON'] == 0.0
    raw = np.random.default_rng(0).normal(scale=2e-5, size=(9, 3000)).astype(np.float32)
    x = torch.from_numpy((raw - mean) / std)
    with torch.inference_mode():
        got = predict.predict_epochs(x.numpy(), ensemble, 4, 'cpu', batch_size=3)[0]
        direct = []
        for model, m, s in ((a, 1e-6, 2e-5), (b, -3e-6, 2.4e-5)):
            member_input = torch.from_numpy((raw - m) / s).float()
            direct.append(predict.predict_epochs(member_input.numpy(), model, 4, 'cpu', batch_size=3)[0])
        stacked = ensemble(x[:4].unsqueeze(0))[0].numpy()
    np.testing.assert_allclose(got, np.mean(direct, axis=0), atol=2e-5, rtol=2e-4)
    window = np.mean([m(torch.from_numpy((raw[:4] - mm) / ss).float().unsqueeze(0))[0].detach().numpy()
                      for m, mm, ss in ((a, 1e-6, 2e-5), (b, -3e-6, 2.4e-5))], axis=0)
    np.testing.assert_allclose(stacked, window, atol=2e-5, rtol=2e-4)


def test_bundle_hash_depends_on_every_member_and_order(tmp_path):
    for index, name in enumerate(['m0', 'm1']):
        _member_dir(tmp_path, name, index, 0.0, 1e-5)
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'ensemble.json').write_text(json.dumps({'members': ['m0', 'm1']}))
    first = predict.checkpoint_sha256(bundle)
    (bundle / 'ensemble.json').write_text(json.dumps({'members': ['m1', 'm0']}))
    assert predict.checkpoint_sha256(bundle) != first
    single = predict.checkpoint_sha256(tmp_path / 'm0')
    assert single == hashlib.sha256((tmp_path / 'm0' / 'best_model.pth').read_bytes()).hexdigest()
    (tmp_path / 'm1' / 'best_model.pth').write_bytes(b'changed')
    (bundle / 'ensemble.json').write_text(json.dumps({'members': ['m0', 'm1']}))
    assert predict.checkpoint_sha256(bundle) != first
