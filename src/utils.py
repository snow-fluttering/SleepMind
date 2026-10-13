"""Evaluate original epochs with explicit annotation and prediction coverage.

NPZ evaluation compares saved argmax and decoded stages against embedded
ground_truth labels. Segment evaluation aligns absolute timestamps on the
recording's 30-second grid; unknown annotations never become Wake.
"""
import argparse
import json
import math
from pathlib import Path
import re

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from data_preprocessing import map_description_to_label, parse_label_file


STAGE_NAMES = ['W', 'N1', 'N2', 'N3', 'REM']
CLASSES = list(range(5))


def _labels(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.number):
        raise ValueError(f'{name} must be a one-dimensional numeric label vector')
    if not np.isfinite(values).all() or (values != np.floor(values)).any():
        raise ValueError(f'{name} must contain finite integer labels')
    if ((values < -1) | (values > 4)).any():
        raise ValueError(f'{name} labels must be -1 or 0..4')
    return values.astype(np.int64, copy=False)


def _onsets(values, length, epoch_sec):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (length,) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('onset_sec must contain one finite nonnegative timestamp per epoch')
    if (np.diff(values) <= 0).any():
        raise ValueError('Epoch timestamps must be strictly increasing without duplicates')
    if not np.allclose(values / epoch_sec, np.round(values / epoch_sec), atol=1e-7, rtol=0):
        raise ValueError('Epoch timestamps must lie on the absolute recording epoch grid')
    return values


def _mask(values, length):
    if values is None:
        return np.ones(length, dtype=bool)
    values = np.asarray(values)
    if values.shape != (length,) or not np.isin(values, [0, 1, False, True]).all():
        raise ValueError('scorable must contain one boolean per epoch')
    return values.astype(bool)


def _stage_counts(labels):
    return {name: int(np.count_nonzero(labels == index)) for index, name in enumerate(STAGE_NAMES)}


def _classification_metrics(truth, prediction):
    """Use five fixed labels even when an evaluated subset has absent stages."""
    if not len(truth):
        return {'accuracy': None, 'macro_f1_5class': None, 'cohen_kappa': None,
                'classification_report': None, 'confusion_matrix': np.zeros((5, 5), int).tolist()}
    cm = confusion_matrix(truth, prediction, labels=CLASSES)
    accuracy = float(np.mean(truth == prediction))
    expected = float(np.dot(cm.sum(axis=0), cm.sum(axis=1)) / len(truth) ** 2)
    kappa = (accuracy - expected) / (1 - expected) if expected < 1 - 1e-12 else None
    return {
        'accuracy': accuracy,
        'macro_f1_5class': float(f1_score(truth, prediction, labels=CLASSES, average='macro', zero_division=0)),
        'cohen_kappa': kappa,
        'classification_report': classification_report(
            truth, prediction, labels=CLASSES, target_names=STAGE_NAMES, output_dict=True, zero_division=0),
        'confusion_matrix': cm.tolist(),
    }


def evaluate_arrays(ground_truth, predictions, scorable=None):
    """Return metrics plus abstentions; metrics cannot hide lost prediction coverage."""
    truth, prediction = _labels(ground_truth, 'ground_truth'), _labels(predictions, 'predictions')
    if len(truth) != len(prediction):
        raise ValueError('Reference and prediction lengths differ; truncation is forbidden')
    valid = _mask(scorable, len(truth))
    known = truth >= 0
    eligible = known & valid
    scored = eligible & (prediction >= 0)
    rejected = eligible & (prediction < 0)
    sleep = eligible & (truth > 0)
    known_count, eligible_count, scored_count = map(int, (known.sum(), eligible.sum(), scored.sum()))
    result = {
        'total_epoch_positions': int(len(truth)),
        'known_reference_epochs': known_count,
        'unknown_reference_epochs': int((~known).sum()),
        'unscorable_signal_epochs': int((~valid).sum()),
        'known_but_unscorable_epochs': int((known & ~valid).sum()),
        'eligible_reference_epochs': eligible_count,
        'scored_epochs': scored_count,
        'rejected_predicted_epochs': int(rejected.sum()),
        'all_rejected_prediction_epochs': int((prediction < 0).sum()),
        'prediction_coverage_on_eligible_reference': scored_count / eligible_count if eligible_count else None,
        'correct_fraction_of_eligible_reference': int((scored & (prediction == truth)).sum()) / eligible_count if eligible_count else None,
        'reference_stage_counts': _stage_counts(truth[eligible]),
        'scored_reference_stage_counts': _stage_counts(truth[scored]),
        'predicted_stage_counts': _stage_counts(prediction[valid & (prediction >= 0)]),
        'rejected_by_reference_stage': _stage_counts(truth[rejected]),
        'confusion_stage_order': STAGE_NAMES,
        'sleep_reference_epochs': int(sleep.sum()),
        'sleep_stage_evaluation_available': bool(sleep.any()),
        'sleep_only_macro_f1_excluding_reference_wake': None,
        'macro_f1_5class_with_rejections_as_errors': None,
    }
    result.update(_classification_metrics(truth[scored], prediction[scored]))
    if eligible_count:
        result['macro_f1_5class_with_rejections_as_errors'] = float(f1_score(
            truth[eligible], prediction[eligible], labels=CLASSES, average='macro', zero_division=0))
    if sleep.any():
        result['sleep_only_macro_f1_excluding_reference_wake'] = float(f1_score(
            truth[sleep], prediction[sleep], labels=[1, 2, 3, 4], average='macro', zero_division=0))
        result['annotation_scope'] = 'Observed known reference intervals only'
    elif known_count:
        result['annotation_scope'] = 'Known Wake intervals only; sleep-stage quality cannot be evaluated'
    else:
        result['annotation_scope'] = 'No recognized reference annotations; quality cannot be evaluated'
    result['metric_notes'] = [
        'Accuracy, kappa, report, and confusion matrix use known, scorable epochs with accepted predictions.',
        'Coverage and rejection-aware F1 expose missing predictions; unknown reference labels are excluded.',
        'Five-class macro-F1 includes all five fixed labels, including stages with zero support.',
        'Sleep-only F1 excludes actual Wake and does not penalize false sleep predictions on actual Wake.',
        'Scores on partial supplied annotations are not full-recording or independent test accuracy.',
    ]
    return result


def _existing_path(value, name):
    if value is None or not str(value).strip():
        raise ValueError(f'{name} must be a nonempty existing file or directory')
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f'{name} does not exist: {path}')
    return path


def evaluate_epoch_files(path, epoch_sec=30):
    """Evaluate *_epochs.npz without reading withheld reference stage labels."""
    source = _existing_path(path, 'pred_dir')
    files = [source] if source.is_file() else sorted(source.rglob('*_epochs.npz'))
    if not files:
        raise ValueError(f'No epoch NPZ files found: {source}')
    records, aggregate = {}, {'ground_truth': [], 'argmax': [], 'decoded': [], 'scorable': []}
    disagreements = 0
    for file in files:
        record_id = file.stem.removesuffix('_epochs')
        if record_id in records:
            raise ValueError(f'Duplicate recording output: {record_id}')
        with np.load(file, allow_pickle=False) as archive:
            missing = {'ground_truth', 'argmax', 'decoded', 'onset_sec'} - set(archive.files)
            if missing:
                raise ValueError(f'Missing epoch arrays in {file.name}: {sorted(missing)}')
            truth = _labels(archive['ground_truth'], 'ground_truth')
            argmax = _labels(archive['argmax'], 'argmax')
            decoded = _labels(archive['decoded'], 'decoded')
            if len(argmax) != len(truth) or len(decoded) != len(truth):
                raise ValueError(f'Mismatched epoch array lengths: {file.name}')
            onsets = _onsets(archive['onset_sec'], len(truth), epoch_sec)
            scorable = _mask(archive['scorable'] if 'scorable' in archive.files else None, len(truth))
        shared = scorable & (argmax >= 0) & (decoded >= 0)
        changes = int((shared & (argmax != decoded)).sum())
        disagreements += changes
        records[record_id] = {
            'source_file': str(file.resolve()), 'epoch_positions': int(len(truth)),
            'missing_epoch_grid_positions_before_last': int(round(onsets[-1] / epoch_sec) + 1 - len(onsets)) if len(onsets) else 0,
            'argmax_decoded_disagreements': changes,
            'prediction_acceptance_disagreements': int(((argmax >= 0) != (decoded >= 0)).sum()),
            'argmax': evaluate_arrays(truth, argmax, scorable),
            'decoded': evaluate_arrays(truth, decoded, scorable),
        }
        for name, values in [('ground_truth', truth), ('argmax', argmax), ('decoded', decoded), ('scorable', scorable)]:
            aggregate[name].append(values)
    joined = {name: np.concatenate(values) for name, values in aggregate.items()}
    return {
        'format': 'epoch_npz', 'epoch_seconds': epoch_sec, 'record_count': len(records),
        'argmax_decoded_disagreements': disagreements, 'records': records,
        'argmax': evaluate_arrays(joined['ground_truth'], joined['argmax'], joined['scorable']),
        'decoded': evaluate_arrays(joined['ground_truth'], joined['decoded'], joined['scorable']),
    }


def _segment_files(value, name):
    source = _existing_path(value, name)
    candidates = [source] if source.is_file() else sorted(source.rglob('*'))
    grouped = {}
    for file in candidates:
        if not file.is_file() or file.suffix.lower() not in ('.txt', '.tsv'):
            continue
        match = re.fullmatch(r'(.+)_Hypnogram(?:_Data_Part_(\d+)_of_(\d+))?', file.stem, re.I)
        if match:
            grouped.setdefault(match.group(1), []).append((file, match.group(2)))
    if not grouped:
        raise ValueError(f'No hypnogram TXT/TSV files found: {source}')
    result = {}
    for record_id, values in grouped.items():
        whole = [file for file, part in values if part is None]
        if len(whole) > 1:
            raise ValueError(f'Duplicate whole-record hypnograms for {record_id}')
        result[record_id] = whole or [file for file, _ in values]
    return result


def _intervals(files):
    rows = []
    for file in files:
        rows.extend((onset, end, map_description_to_label(description))
                    for onset, end, _, description in parse_label_file(file) if end > onset)
    intervals = []
    for start, end, label in sorted(rows):
        if intervals and start < intervals[-1][1] - 1e-6:
            raise ValueError('Hypnogram intervals overlap or repeat; timestamp alignment is ambiguous')
        if intervals and abs(start - intervals[-1][1]) <= 1e-6 and label == intervals[-1][2]:
            intervals[-1] = (intervals[-1][0], end, label)
        else:
            intervals.append((start, end, label))
    return intervals


def _labels_on_grid(intervals, starts, epoch_sec):
    labels = np.full(len(starts), -1, dtype=np.int64)
    for start, end, label in intervals:
        if label >= 0:
            covered = (starts >= start - 1e-6) & (starts + epoch_sec <= end + 1e-6)
            labels[covered] = label
    return labels


def load_pred_file_as_array(path, epoch_sec=30):
    """Expand a segment file on its absolute grid, retaining unknown positions."""
    intervals = _intervals([_existing_path(path, 'path')])
    count = int(math.floor(max((end for _, end, _ in intervals), default=0) / epoch_sec))
    return _labels_on_grid(intervals, np.arange(count) * epoch_sec, epoch_sec)


def evaluate_segment_files(pred_dir, gt_dir, epoch_sec=30):
    predictions, references = _segment_files(pred_dir, 'pred_dir'), _segment_files(gt_dir, 'gt_dir')
    shared = sorted(set(predictions) & set(references))
    if not shared:
        raise ValueError('No matching recording IDs between prediction and reference files')
    records, all_truth, all_prediction = {}, [], []
    # Missing whole-record predictions must reduce coverage, just like missing
    # intervals. Predictions without references remain explicitly unscored.
    for record_id in sorted(set(predictions) | set(references)):
        predicted = _intervals(predictions.get(record_id, []))
        reference = _intervals(references.get(record_id, []))
        end = max((right for _, right, _ in predicted + reference), default=0)
        starts = np.arange(int(math.floor(end / epoch_sec))) * epoch_sec
        truth = _labels_on_grid(reference, starts, epoch_sec)
        prediction = _labels_on_grid(predicted, starts, epoch_sec)
        records[record_id] = evaluate_arrays(truth, prediction)
        records[record_id]['reference_only_known_epoch_positions'] = int(((truth >= 0) & (prediction < 0)).sum())
        records[record_id]['prediction_without_known_reference_positions'] = int(((truth < 0) & (prediction >= 0)).sum())
        all_truth.append(truth)
        all_prediction.append(prediction)
    result = evaluate_arrays(np.concatenate(all_truth), np.concatenate(all_prediction))
    return {'format': 'absolute_timestamp_segments', 'epoch_seconds': epoch_sec,
            'record_count': len(records), 'records': records, 'predictions': result,
            'prediction_recordings_without_reference': sorted(set(predictions) - set(references)),
            'reference_recordings_without_prediction': sorted(set(references) - set(predictions))}


def evaluate(pred_dir, gt_dir=None, epoch_sec=30):
    """Select embedded-reference NPZ evaluation or explicit-reference segments."""
    if not np.isfinite(epoch_sec) or epoch_sec <= 0:
        raise ValueError('epoch_sec must be finite and positive')
    path = _existing_path(pred_dir, 'pred_dir')
    if gt_dir is None:
        return evaluate_epoch_files(path, epoch_sec)
    return evaluate_segment_files(path, gt_dir, epoch_sec)


def plot_hypnogram(label_sequence, save_path, title='Hypnogram', epoch_sec=30):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = _labels(label_sequence, 'label_sequence')
    remap = {0: 4, 4: 3, 1: 2, 2: 1, 3: 0, -1: np.nan}
    figure, axis = plt.subplots(figsize=(12, 3))
    axis.step(np.arange(len(labels)) * epoch_sec / 3600, [remap[int(x)] for x in labels], where='post')
    axis.set(yticks=range(5), yticklabels=['N3', 'N2', 'N1', 'REM', 'W'], xlabel='Time (hours)', ylabel='Stage', title=title)
    axis.grid(alpha=.3)
    figure.tight_layout()
    figure.savefig(save_path, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pred_dir', required=True, help='Epoch NPZ file/directory, or hypnogram file/directory')
    parser.add_argument('--gt_dir', help='Reference hypnogram file/directory for segment evaluation')
    parser.add_argument('--output', help='Save the complete JSON metrics')
    parser.add_argument('--epoch_sec', type=float, default=30)
    args = parser.parse_args()
    result = evaluate(args.pred_dir, args.gt_dir, args.epoch_sec)
    serialized = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(serialized + '\n', encoding='utf-8')
        summary = {'output': str(destination.resolve()), 'record_count': result['record_count']}
        for method in ('argmax', 'decoded', 'predictions'):
            if method in result:
                summary[method] = {key: result[method][key] for key in (
                    'accuracy', 'macro_f1_5class', 'scored_epochs', 'rejected_predicted_epochs',
                    'prediction_coverage_on_eligible_reference', 'sleep_stage_evaluation_available', 'annotation_scope')}
        print(json.dumps(summary, indent=2, allow_nan=False))
    else:
        print(serialized)


if __name__ == '__main__':
    main()
