"""Participant-level metrics, gap-aware decoding, calibration and the staged evaluation CLI."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score, f1_score

import evaluation as ev
from decoding import default_transition

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import evaluate_run  # noqa: E402


def test_confusion_metrics_match_sklearn_and_keep_five_fixed_classes():
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 4, 400)  # REM never occurs in the reference
    prediction = np.where(rng.random(400) < 0.7, truth, rng.integers(0, 5, 400))
    matrix = ev.confusion_counts(truth, prediction)
    metrics = ev.metrics_from_confusion(matrix)
    assert metrics['accuracy'] == pytest.approx(np.mean(truth == prediction))
    assert metrics['macro_f1_5class'] == pytest.approx(
        f1_score(truth, prediction, labels=range(5), average='macro', zero_division=0))
    assert metrics['cohen_kappa'] == pytest.approx(cohen_kappa_score(truth, prediction))
    assert metrics['per_stage']['REM']['support'] == 0 and metrics['per_stage']['REM']['recall'] == 0
    sleep = truth > 0
    assert metrics['sleep_only_macro_f1_excluding_reference_wake'] == pytest.approx(
        f1_score(truth[sleep], prediction[sleep], labels=[1, 2, 3, 4], average='macro', zero_division=0))


def test_unknown_unscorable_and_rejected_epochs_are_excluded_but_counted():
    truth = np.array([0, 1, -1, 2, 3])
    prediction = np.array([0, -1, 2, 2, 4])
    scorable = np.array([True, True, True, False, True])
    assert ev.confusion_counts(truth, prediction, scorable).sum() == 2
    assert ev.eligible_count(truth, scorable) == 3  # known and scorable, including the rejected epoch


def test_bootstrap_resamples_participants_reproducibly():
    cms = {f'P{i}': np.diag([10 + i] * 5) + np.ones((5, 5), int) * i for i in range(6)}
    a = ev.bootstrap_participants(cms, lambda m: ev.metrics_from_confusion(m)['accuracy'], 300, 1)
    b = ev.bootstrap_participants(cms, lambda m: ev.metrics_from_confusion(m)['accuracy'], 300, 1)
    assert a == b and a['lower'] <= a['upper'] and a['n_participants'] == 6
    assert ev.bootstrap_participants({'only': cms['P0']}, lambda m: 0.0)['lower'] is None
    paired = ev.paired_difference_bootstrap(cms, cms, n_boot=100)
    assert paired['difference'] == 0 and paired['lower'] == 0 and paired['upper'] == 0


def test_paired_interval_uses_equal_support_and_recovers_known_synthetic_gain():
    baseline = np.diag([16] * 5) + np.roll(np.diag([4] * 5), 1, axis=1)
    improved = np.diag([17] * 5) + np.roll(np.diag([3] * 5), 1, axis=1)
    old = {f'P{i}': baseline.copy() for i in range(6)}
    new = {f'P{i}': improved.copy() for i in range(6)}
    gain = ev.paired_difference_bootstrap(new, old, n_boot=200, seed=13)
    assert gain['difference'] == pytest.approx(0.05)
    assert gain['lower'] == pytest.approx(0.05) and gain['upper'] == pytest.approx(0.05)
    assert gain == ev.paired_difference_bootstrap(new, old, n_boot=200, seed=13)


@pytest.mark.parametrize('kind', ['participants', 'empty', 'counts', 'zero_counts', 'reference_support',
                                 'shape', 'negative', 'fractional', 'nonfinite'])
def test_paired_interval_rejects_incomparable_or_invalid_synthetic_counts(kind):
    counts = np.diag([10] * 5)
    old, new = {'A': counts.copy()}, {'A': counts.copy()}
    message = 'valid five-class confusion counts'
    if kind == 'participants':
        old['B'] = counts.copy()
        message = 'identical nonempty participant sets'
    elif kind == 'empty':
        old, new = {}, {}
        message = 'identical nonempty participant sets'
    elif kind == 'counts':
        new['A'][0, 0] += 1
        message = 'equal positive accepted counts'
    elif kind == 'zero_counts':
        old['A'].fill(0)
        new['A'].fill(0)
        message = 'equal positive accepted counts'
    elif kind == 'reference_support':
        new['A'][0, 0] -= 1
        new['A'][1, 1] += 1
        message = 'identical reference support'
    elif kind == 'shape':
        new['A'] = np.eye(4, dtype=int)
    elif kind == 'negative':
        new['A'][0, 1] = -1
    elif kind == 'fractional':
        new['A'] = new['A'].astype(float)
        new['A'][0, 0] += 0.5
    elif kind == 'nonfinite':
        new['A'] = new['A'].astype(float)
        new['A'][0, 0] = np.nan
    with pytest.raises(ValueError, match=message):
        ev.paired_difference_bootstrap(new, old, n_boot=10)


def test_decoding_resets_at_gaps_and_does_not_bridge_runs():
    scorable = np.array([True, True, False, True, True, True])
    assert ev.contiguous_runs(scorable) == [(0, 2), (3, 6)]
    logits = np.zeros((6, 5))
    logits[:, 2] = 0.01
    logits[2] = np.nan  # unscorable positions carry no logits and must be ignored
    forbidden = np.full((5, 5), 1.0)
    forbidden[:, 0] = 0.0  # no transition may enter Wake
    forbidden[0, 0] = 1.0
    decoded = ev.decode_record(logits, scorable, forbidden, 1.0)
    assert decoded[2] == -1 and (decoded[[0, 1, 3, 4, 5]] >= 0).all()
    assert (ev.decode_record(logits, scorable, None, 0.0)[[0, 1, 3, 4, 5]] == 2).all()
    diagnostics = ev.transition_diagnostics(np.array([0, 0, 1, 1, -1, 2]), np.array([1, 1, 1, 1, 0, 1]))
    assert diagnostics['stage_changes'] == 1  # 1 -> 2 across the gap is not counted


def test_temperature_recovers_known_overconfidence_and_ece_improves():
    rng = np.random.default_rng(1)
    n = 6000
    base = rng.normal(size=(n, 5))
    probabilities = ev.softmax(base)
    truth = np.array([rng.choice(5, p=p) for p in probabilities])
    overconfident = base * 3.0
    temperature = ev.fit_temperature(overconfident, truth)
    assert 2.5 < temperature < 3.6
    before = ev.calibration_summary(ev.softmax(overconfident), truth)
    after = ev.calibration_summary(ev.softmax(overconfident, temperature), truth)
    assert after['ece_top_label'] < before['ece_top_label'] and after['ece_top_label'] < 0.05
    curve = ev.risk_coverage(ev.softmax(overconfident, temperature), np.argmax(overconfident, 1), truth, [0.0, 0.9])
    assert curve[0]['coverage'] == 1.0 and curve[1]['coverage'] < 1.0


def _synthetic_epochs(directory, records, seed, informative=3.0):
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for record in records:
        n = 120
        truth = np.repeat(rng.integers(0, 5, n // 10), 10)
        logits = rng.normal(size=(n, 5)) + informative * np.eye(5)[truth]
        scorable = np.ones(n, bool)
        scorable[50] = False
        logits[50] = np.nan
        argmax = np.where(scorable, np.nan_to_num(logits).argmax(1), -1)
        np.savez_compressed(directory / f'{record}_epochs.npz', onset_sec=np.arange(n) * 30.0,
                            ground_truth=truth, argmax=argmax, decoded=argmax, logits=logits.astype(np.float32),
                            scorable=scorable)


def test_staged_protocol_freezes_on_validation_and_binds_the_checkpoint(tmp_path):
    from data_preprocessing import participant_id
    from splits import participant_table, build_manifest
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'best_model.pth').write_bytes(b'checkpoint-bytes')
    (run / 'norm_stats.npz').write_bytes(b'normalization-bytes')
    (run / 'transition.json').write_text(json.dumps(default_transition().tolist()))
    records = ['SC9001E0', 'SC9071E0', 'ST8011J0', 'ST8061J0', 'ST8062J0']
    table = participant_table([{'record_id': r, 'epoch_counts': [1] * 5} for r in records])
    assignment = {participant_id(r): ('train' if r == 'SC9001E0' else
        'holdout' if r.startswith('ST806') else 'validation') for r in records}
    manifest = build_manifest(table, assignment, {})
    manifest_path = tmp_path / 'm.json'
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    (run / 'config.json').write_text(json.dumps({'SEQ_LEN': 20, 'fit_subjects': ['SC900'],
        'split_manifest_sha256': manifest['manifest_sha256']}), encoding='utf-8')
    def metadata(directory, partition):
        (directory / 'inference_info.json').write_text(json.dumps({
            'checkpoint_sha256': evaluate_run.checkpoint_sha256(run), 'input_mode': 'raw',
            'sequence_length': 20, 'tta': 1, 'partition': partition,
            'records': manifest['partitions'][partition]['records'], 'split_manifest': str(manifest_path)}), encoding='utf-8')
    _synthetic_epochs(tmp_path / 'val', ['SC9071E0', 'ST8011J0'], 1)
    metadata(tmp_path / 'val', 'validation')
    frozen = tmp_path / 'frozen.json'
    args = type('A', (), dict(run_dir=str(run), val_dir=str(tmp_path / 'val'), output=str(frozen), target_accuracy=0.8))
    evaluate_run.cmd_freeze(args)
    protocol = json.loads(frozen.read_text())
    assert protocol['selected_variant'] in {evaluate_run.variant_name(*v) for v in evaluate_run.all_variants()}
    assert protocol['derived_from'].startswith('validation partition only')
    assert protocol['validation_calibration_after']['nll'] <= protocol['validation_calibration_before']['nll'] + 1e-9
    # Holdout inference must cover exactly the manifest's holdout records and the same checkpoint.
    _synthetic_epochs(tmp_path / 'hold', ['ST8061J0'], 2)
    report_args = type('R', (), dict(run_dir=str(run), holdout_dir=str(tmp_path / 'hold'), frozen=str(frozen),
                                     split_manifest=str(tmp_path / 'm.json'), output=str(tmp_path / 'out' / 'r.json'),
                                     n_boot=20, seed=0))
    with pytest.raises(ValueError, match='exactly the holdout records'):
        evaluate_run.cmd_report(report_args)
    _synthetic_epochs(tmp_path / 'hold', ['ST8061J0', 'ST8062J0'], 2)
    metadata(tmp_path / 'hold', 'holdout')
    evaluate_run.cmd_report(report_args)
    report = json.loads((tmp_path / 'out' / 'r.json').read_text())
    scope = report['scopes']['holdout_all']
    assert scope['argmax_metrics']['n'] == 2 * 119  # the unscorable epoch is excluded from accepted metrics
    assert scope['argmax_metrics']['eligible_reference_epochs'] == 2 * 119
    assert scope['argmax_metrics']['known_reference_epochs'] == 2 * 120 and scope['argmax_metrics']['known_but_unscorable_epochs'] == 2
    assert scope['variants']['argmax']['confidence_intervals_95_participant_bootstrap']['accuracy']['n_participants'] == 1
    assert 'holdout_requested_records' not in report['scopes']
    frozen_bytes = frozen.read_bytes()
    report_args.diagnostic_records = ' ST8061J0 '
    evaluate_run.cmd_report(report_args)
    diagnostic = json.loads((tmp_path / 'out' / 'r.json').read_text())['scopes']['holdout_requested_records']
    assert diagnostic['records'] == ['ST8061J0']
    assert diagnostic['argmax_metrics']['n'] == 119
    assert frozen.read_bytes() == frozen_bytes
    for records, message in [('', 'nonempty and unique'),
                             ('ST8061J0,ST8061J0', 'nonempty and unique'),
                             ('SC9071E0', 'subset of the declared holdout')]:
        report_args.diagnostic_records = records
        with pytest.raises(ValueError, match=message):
            evaluate_run.cmd_report(report_args)
    report_args.diagnostic_records = None
    (run / 'best_model.pth').write_bytes(b'different-checkpoint')
    with pytest.raises(ValueError, match='different checkpoint'):
        evaluate_run.cmd_report(report_args)
