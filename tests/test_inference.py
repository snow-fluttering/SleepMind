import numpy as np
import torch
from model import SeqSleepStagingModel
from predict import predict_epochs, infer_record, frame_labels
from reporting import summarize, validate_segments
from readers import EEGRecording


def test_cached_epoch_encoder_matches_direct_overlap_inference():
    torch.set_num_threads(1)
    torch.manual_seed(3)
    model = SeqSleepStagingModel(feat_dim=16, lstm_hidden=8, n_trans_layers=1, n_heads=2).eval()
    frames = np.random.default_rng(2).normal(size=(7, 3000)).astype(np.float32)
    optimized, _ = predict_epochs(frames, model, 4, 'cpu', batch_size=3)
    total = np.zeros((7, 5))
    count = np.zeros(7)
    with torch.inference_mode():
        for start in [0, 2, 3]:
            logits = model(torch.from_numpy(frames[start:start + 4]).unsqueeze(0))[0].numpy()
            total[start:start + 4] += logits
            count[start:start + 4] += 1
    np.testing.assert_allclose(optimized, total / count[:, None], atol=2e-6, rtol=2e-5)


def test_unknown_and_partial_annotation_are_not_wake():
    segments = [(30, 90, 60, 'Sleep stage W'), (90, 120, 30, ''), (120, 150, 30, 'Movement time')]
    np.testing.assert_array_equal(frame_labels(segments, np.arange(5) * 30), [-1, 0, 0, -1, -1])
    np.testing.assert_array_equal(frame_labels([(10, 65, 55, 'W')], np.array([0, 30, 60])), [-1, 0, -1])
    np.testing.assert_array_equal(frame_labels([(0, 36, 36, 'W'), (36, 90, 54, 'Wake')], np.arange(3)*30), [0, 0, 0])


def test_short_run_uses_only_observed_context():
    torch.set_num_threads(1)
    model = SeqSleepStagingModel(feat_dim=16, lstm_hidden=8, n_trans_layers=1, n_heads=2).eval()
    frames = np.random.default_rng(6).normal(size=(2, 3000)).astype(np.float32)
    result, _ = predict_epochs(frames, model, 20, 'cpu')
    with torch.inference_mode():
        direct = model(torch.from_numpy(frames).unsqueeze(0))[0].numpy()
    np.testing.assert_allclose(result, direct, atol=2e-6, rtol=2e-5)


def test_fragments_flat_signal_and_tail_preserve_timeline():
    torch.set_num_threads(1)
    model = SeqSleepStagingModel(feat_dim=16, lstm_hidden=8, n_trans_layers=1, n_heads=2).eval()
    samples = np.zeros(12500, dtype=np.float32)
    samples[:3000] = np.random.default_rng(2).normal(scale=1e-5, size=3000)
    samples[9000:12000] = samples[:3000]
    record = {'eeg': samples, 'label_segments': [], 'blocks': [
        {'start_sample': 0, 'end_sample': 6000}, {'start_sample': 9000, 'end_sample': 12500}]}
    segments, arrays = infer_record(record, model, 0, 1e-5, seq_len=4, decoder='argmax')
    validate_segments(segments, 125)
    assert any(s['reason'] == 'missing_fragment' and s['onset_sec'] == 60 and s['end_sec'] == 90 for s in segments)
    assert any(s['reason'] == 'incomplete_epoch' and s['end_sec'] == 125 for s in segments)
    report = summarize(segments, {'total_seconds': 125})
    assert report['unscorable_seconds'] == 65
    assert report['clinical_sleep_efficiency'] is None
    assert len(arrays['decoded']) == 4
    np.testing.assert_array_equal(arrays['scorable'], [True, False, False, True])
    assert arrays['decoded'][2] == -1 and np.isnan(arrays['logits'][2]).all()


def test_reader_units_are_explicit():
    record = EEGRecording(np.array([10., 20.]), 100, 'EEG Fpz-Cz', 'uV')
    np.testing.assert_allclose(record.volts(), np.array([1e-5, 2e-5]), rtol=1e-6)
