"""Participant-level evaluation: confusion-matrix metrics, bootstrap, decoding and calibration.

All functions work on per-epoch label arrays produced by ``predict.infer_record``
(``-1`` marks unknown reference or no prediction). Each original epoch is counted
once. Metrics derived from pooled confusion matrices support participant-level
resampling because a participant's nights share one resampling unit.
"""
import numpy as np

from decoding import viterbi_decode

STAGES = ['W', 'N1', 'N2', 'N3', 'REM']


def confusion_counts(truth, prediction, scorable=None):
    """5 x 5 counts of known, scorable epochs with an accepted prediction (rows = truth)."""
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    valid = (truth >= 0) & (prediction >= 0)
    if scorable is not None:
        valid &= np.asarray(scorable, bool)
    matrix = np.zeros((5, 5), np.int64)
    np.add.at(matrix, (truth[valid], prediction[valid]), 1)
    return matrix


def eligible_count(truth, scorable=None):
    truth = np.asarray(truth)
    valid = truth >= 0
    if scorable is not None:
        valid &= np.asarray(scorable, bool)
    return int(valid.sum())


def metrics_from_confusion(matrix):
    """Accuracy, five-class macro F1 (absent stages score 0), kappa, per-stage P/R/F1/support."""
    matrix = np.asarray(matrix, np.float64)
    n = matrix.sum()
    if n == 0:
        return {'n': 0, 'accuracy': None, 'macro_f1_5class': None, 'cohen_kappa': None, 'per_stage': None}
    tp = np.diag(matrix)
    support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        f1 = np.where(support + predicted > 0, 2 * tp / (support + predicted), 0.0)
    accuracy = tp.sum() / n
    expected = float((support * predicted).sum() / n ** 2)
    kappa = (accuracy - expected) / (1 - expected) if expected < 1 - 1e-12 else None
    sleep = slice(1, 5)
    sleep_n = matrix[sleep, :].sum()
    return {
        'n': int(n), 'accuracy': float(accuracy), 'macro_f1_5class': float(f1.mean()), 'cohen_kappa': kappa,
        'sleep_only_macro_f1_excluding_reference_wake': float(_sleep_only_f1(matrix)) if sleep_n else None,
        'per_stage': {name: {'precision': float(precision[i]), 'recall': float(recall[i]),
                             'f1': float(f1[i]), 'support': int(support[i])}
                      for i, name in enumerate(STAGES)},
    }


def _sleep_only_f1(matrix):
    """Competition-style F1: only epochs whose reference is sleep (precision ignores false sleep on Wake)."""
    sub = matrix[1:5, :]
    tp = np.diag(matrix)[1:5]
    support = sub.sum(axis=1)
    predicted = sub.sum(axis=0)[1:5]
    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = np.where(support + predicted > 0, 2 * tp / (support + predicted), 0.0)
    return f1.mean()


def bootstrap_participants(per_participant, statistic, n_boot=2000, seed=0):
    """Percentile 95% interval resampling participants with replacement.

    ``per_participant`` maps participant -> payload (summed with ``+``);
    ``statistic`` maps the summed payload to a float or None.
    """
    names = sorted(per_participant)
    if len(names) < 2:
        return {'lower': None, 'upper': None, 'n_participants': len(names), 'n_boot': 0}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(names), len(names))
        total = sum(per_participant[names[i]] for i in pick)
        value = statistic(total)
        if value is not None:
            values.append(value)
    if not values:
        return {'lower': None, 'upper': None, 'n_participants': len(names), 'n_boot': 0}
    low, high = np.percentile(values, [2.5, 97.5])
    return {'lower': float(low), 'upper': float(high), 'n_participants': len(names), 'n_boot': len(values)}


def summarize_variant(per_participant_cm, per_participant_eligible, n_boot=2000, seed=0, per_participant_known=None):
    """Pooled metrics, participant-level intervals and per-participant accuracy for one decoder.

    ``eligible`` counts known reference epochs with usable signal; ``known`` additionally counts known
    epochs whose signal was flat or invalid (always unscorable, hence never part of accepted metrics).
    """
    pooled = sum(per_participant_cm.values())
    eligible = sum(per_participant_eligible.values())
    result = metrics_from_confusion(pooled)
    result['eligible_reference_epochs'] = int(eligible)
    if per_participant_known is not None:
        known = int(sum(per_participant_known.values()))
        result['known_reference_epochs'] = known
        result['known_but_unscorable_epochs'] = known - int(eligible)
        result['coverage_of_known_reference'] = result['n'] / known if known else None
    result['coverage_of_eligible_reference'] = result['n'] / eligible if eligible else None
    result['correct_fraction_of_eligible_reference'] = (float(np.trace(pooled)) / eligible) if eligible else None
    result['confusion_matrix'] = pooled.tolist()
    result['confidence_intervals_95_participant_bootstrap'] = {
        name: bootstrap_participants(per_participant_cm, lambda m, key=name: metrics_from_confusion(m)[key],
                                     n_boot, seed)
        for name in ('accuracy', 'macro_f1_5class', 'cohen_kappa')}
    result['per_participant'] = {
        p: {'n': int(cm.sum()), 'accuracy': metrics_from_confusion(cm)['accuracy'],
            'macro_f1_5class': metrics_from_confusion(cm)['macro_f1_5class'],
            'stage_support': cm.sum(axis=1).astype(int).tolist()}
        for p, cm in sorted(per_participant_cm.items())}
    return result


def paired_difference_bootstrap(cm_new, cm_base, key='accuracy', n_boot=2000, seed=0):
    """Bootstrap paired metrics with identical participants, counts and reference support.

    Confusion counts do not establish epoch identity or ordering. Callers must
    separately verify that both predictions cover the same reference epochs.
    """
    if not cm_new or set(cm_new) != set(cm_base):
        raise ValueError('Paired comparison requires identical nonempty participant sets')
    payload = {}
    for participant in sorted(cm_new):
        current, baseline = np.asarray(cm_new[participant]), np.asarray(cm_base[participant])
        for counts in (current, baseline):
            if (counts.shape != (5, 5) or counts.dtype.kind not in 'biuf' or not np.isfinite(counts).all() or
                    np.any(counts < 0) or np.any(counts != np.floor(counts))):
                raise ValueError('Paired comparison requires valid five-class confusion counts')
        if current.sum() <= 0 or current.sum() != baseline.sum():
            raise ValueError('Paired comparison requires equal positive accepted counts per participant')
        if not np.array_equal(current.sum(axis=1), baseline.sum(axis=1)):
            raise ValueError('Paired comparison requires identical reference support per participant')
        payload[participant] = np.stack([current, baseline])

    def statistic(total):
        a, b = metrics_from_confusion(total[0])[key], metrics_from_confusion(total[1])[key]
        return None if a is None or b is None else a - b

    point = statistic(sum(payload.values()))
    interval = bootstrap_participants(payload, statistic, n_boot, seed)
    return {'difference': point, **interval}


def contiguous_runs(scorable):
    """(start, end) of maximal consecutive scorable epochs; decoders never cross gaps."""
    padded = np.r_[False, np.asarray(scorable, bool), False]
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[::2], edges[1::2]))


def decode_record(logits, scorable, transition=None, strength=0.0):
    """Viterbi (or argmax when strength is 0) inside each contiguous scorable run."""
    decoded = np.full(len(logits), -1, np.int64)
    for begin, end in contiguous_runs(scorable):
        decoded[begin:end] = viterbi_decode(logits[begin:end], transition, strength)
    return decoded


def transition_diagnostics(decoded, scorable, truth=None):
    """Counts of stage changes, mean bout lengths and implausible switches in decoded hypnograms."""
    changes = np.zeros((5, 5), np.int64)
    bouts = {i: [] for i in range(5)}
    for begin, end in contiguous_runs(scorable):
        path = np.asarray(decoded[begin:end])
        if not len(path):
            continue
        np.add.at(changes, (path[:-1], path[1:]), 1)
        edges = np.r_[0, np.flatnonzero(path[1:] != path[:-1]) + 1, len(path)]
        for left, right in zip(edges[:-1], edges[1:]):
            bouts[int(path[left])].append(int(right - left))
    off = changes.copy()
    np.fill_diagonal(off, 0)
    return {'transition_counts': changes.tolist(), 'stage_changes': int(off.sum()),
            'mean_bout_epochs': {STAGES[i]: (float(np.mean(v)) if v else None) for i, v in bouts.items()}}


def softmax(logits, temperature=1.0):
    z = np.asarray(logits, np.float64) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits, truth, grid=None):
    """Scalar temperature minimizing negative log-likelihood on designated calibration data."""
    logits, truth = np.asarray(logits, np.float64), np.asarray(truth)
    if not len(truth):
        raise ValueError('Calibration requires labeled epochs')
    grid = np.exp(np.linspace(np.log(0.25), np.log(8.0), 200)) if grid is None else grid

    def nll(t):
        p = softmax(logits, t)
        return float(-np.log(np.clip(p[np.arange(len(truth)), truth], 1e-12, None)).mean())

    scores = [nll(t) for t in grid]
    return float(grid[int(np.argmin(scores))])


def calibration_summary(probabilities, truth, n_bins=15):
    """Top-label ECE, multiclass Brier score, NLL and reliability bins."""
    probabilities, truth = np.asarray(probabilities, np.float64), np.asarray(truth)
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == truth
    edges = np.linspace(1 / 5, 1.0, n_bins + 1)
    bins, ece = [], 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (confidence > left) & (confidence <= right) if left > edges[0] else (confidence >= left) & (confidence <= right)
        if mask.any():
            gap = abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
            ece += mask.mean() * gap
            bins.append({'low': float(left), 'high': float(right), 'count': int(mask.sum()),
                         'mean_confidence': float(confidence[mask].mean()),
                         'accuracy': float(correct[mask].mean())})
    onehot = np.eye(5)[truth]
    return {'ece_top_label': float(ece), 'brier_multiclass': float(((probabilities - onehot) ** 2).sum(axis=1).mean()),
            'nll': float(-np.log(np.clip(probabilities[np.arange(len(truth)), truth], 1e-12, None)).mean()),
            'mean_confidence': float(confidence.mean()), 'accuracy': float(correct.mean()),
            'n': int(len(truth)), 'bins': bins}


def risk_coverage(probabilities, prediction, truth, thresholds):
    """Accuracy on accepted epochs, coverage and correct-fraction of all eligible epochs per threshold."""
    probabilities = np.asarray(probabilities)
    confidence = probabilities.max(axis=1)
    correct = np.asarray(prediction) == np.asarray(truth)
    rows = []
    for threshold in thresholds:
        keep = confidence >= threshold
        rows.append({'threshold': float(threshold), 'coverage': float(keep.mean()),
                     'accuracy_on_accepted': float(correct[keep].mean()) if keep.any() else None,
                     'correct_fraction_of_eligible': float((correct & keep).mean())})
    return rows
