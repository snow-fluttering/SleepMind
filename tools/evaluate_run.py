"""Staged evaluation of a fresh run: infer -> freeze (validation only) -> report (holdout).

* ``infer``   whole-record raw inference for a manifest partition or record list; saves per-record
              epoch NPZ files with logits, labels, validity and absolute onsets.
* ``freeze``  uses validation inference only: chooses the decoding variant by validation five-class
              macro F1, fits a temperature on validation, derives an abstention threshold, and
              writes ``frozen_protocol.json`` before any holdout inference exists.
* ``report``  applies the frozen protocol to holdout inference and writes metrics with
              participant-level bootstrap intervals, scopes, calibration and decoder ablations.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data_preprocessing import scan_dataset, load_record, load_record_cached, participant_id
from decoding import default_transition
from evaluation import (STAGES, confusion_counts, eligible_count, summarize_variant, decode_record,
                        transition_diagnostics, fit_temperature, softmax, calibration_summary,
                        risk_coverage, paired_difference_bootstrap, metrics_from_confusion)
from predict import load_model, infer_record, checkpoint_sha256, ensemble_members
from sysinfo import environment, peak_rss_bytes
from splits import load_manifest

STRENGTHS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
PRIORS = ('heuristic', 'learned')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact_path(value):
    """Resolve project-relative evidence after a checkout has been relocated."""
    path = Path(value)
    if path.exists():
        return path
    root = Path(__file__).resolve().parents[1]
    if not path.is_absolute():
        return root / path
    if 'SleepMind' in path.parts:
        index = len(path.parts) - 1 - list(reversed(path.parts)).index('SleepMind')
        return root.joinpath(*path.parts[index + 1:])
    return path


def inference_contract(config):
    sequence_length = int(config.get('SEQ_LEN', 20))
    return {'input_mode': 'raw', 'sequence_length': sequence_length,
            'sequence_stride': max(1, sequence_length // 2), 'tta': 1,
            'processing_parameters': config.get('processing_parameters', {})}


def record_processing(config):
    processing = dict(config.get('processing_parameters', {}))
    if processing.get('input_mode', 'raw') != 'raw':
        raise ValueError('Staged raw evaluation requires a raw-trained processing contract')
    if processing.get('epoch_normalize') is not None or processing.get('clip_value') is not None:
        raise ValueError('Custom epoch normalization/clipping is not supported by inference')
    accepted = set(inspect.signature(load_record).parameters) - {'subject_parts'}
    return {k: v for k, v in processing.items() if k in accepted} | {'input_mode': 'raw'}


def check_epoch_metadata(directory, expected_records, epoch_hashes=None):
    paths = {p.name.removesuffix('_epochs.npz'): p for p in Path(directory).glob('*_epochs.npz')}
    expected = list(expected_records)
    if len(expected) != len(set(expected)):
        raise ValueError('Inference metadata contains duplicate records')
    if set(paths) != set(expected):
        raise ValueError('Inference files differ from the metadata record set')
    if epoch_hashes is not None:
        if set(epoch_hashes) != set(paths):
            raise ValueError('Epoch hash table differs from the inference record set')
        for record, path in paths.items():
            if sha256(path) != epoch_hashes[record]:
                raise ValueError(f'Inference epoch hash mismatch: {record}')


def validated_inference_info(directory, run_dir, partition, manifest_path=None):
    """Bind ordinary inference to its checkpoint, split, contract, record set and saved file hashes.

    Older metadata can lack per-file hashes. It can be checked against
    checkpoint/split/records, but its weaker integrity coverage remains explicit.
    """
    path = Path(directory) / 'inference_info.json'
    if not path.exists():
        raise ValueError(f'Missing inference provenance: {path}')
    info = json.loads(path.read_text(encoding='utf-8'))
    if info.get('kind') == 'oof_validation':
        raise ValueError('Pooled out-of-fold inference requires fold-specific validation')
    if info.get('checkpoint_sha256') != checkpoint_sha256(run_dir):
        raise ValueError('Inference belongs to a different checkpoint')
    if info.get('partition') != partition:
        raise ValueError(f'Inference partition must be {partition}')
    manifest = load_manifest(artifact_path(manifest_path or info['split_manifest']))
    recorded_manifest = load_manifest(artifact_path(info['split_manifest']))
    if recorded_manifest['manifest_sha256'] != manifest['manifest_sha256']:
        raise ValueError('Inference split manifest differs from the requested split')
    if info.get('split_manifest_sha256', manifest['manifest_sha256']) != manifest['manifest_sha256']:
        raise ValueError('Inference split manifest hash mismatch')
    expected = manifest['partitions'][partition]['records']
    if set(info.get('records', [])) != set(expected):
        raise ValueError('Inference records differ from the declared split partition')
    check_epoch_metadata(directory, info['records'], info.get('epoch_sha256'))
    config = json.loads((Path(run_dir) / 'config.json').read_text(encoding='utf-8'))
    if any(participant_id(r) in config.get('fit_subjects', []) for r in expected):
        raise ValueError('Evaluation participant was used for model fitting')
    if not (Path(run_dir) / 'ensemble.json').exists():
        if config.get('split_manifest_sha256', manifest['manifest_sha256']) != manifest['manifest_sha256']:
            raise ValueError('Checkpoint split manifest differs from inference provenance')
    contract = inference_contract(config)
    if 'inference_contract' in info:
        if info['inference_contract'] != contract:
            raise ValueError('Inference processing contract differs from its checkpoint configuration')
    else:
        for field in ('input_mode', 'sequence_length', 'tta'):
            if info.get(field) != contract[field]:
                raise ValueError(f'Original inference {field} differs from the checkpoint contract')
        # The original implementation used load_record defaults, regardless of custom configuration.
        defaults = {name: p.default for name, p in inspect.signature(load_record).parameters.items()
                    if p.default is not inspect.Parameter.empty}
        processing = record_processing(config)
        if any(value != defaults[key] for key, value in processing.items() if key != 'input_mode'):
            raise ValueError('Original inference metadata cannot verify non-default processing')
    if info.get('normalization_sha256') is not None:
        if info['normalization_sha256'] != sha256(Path(run_dir) / 'norm_stats.npz'):
            raise ValueError('Inference normalization hash mismatch')
    info = dict(info)
    info['verified_contract'] = contract
    info['saved_epoch_hash_binding'] = 'verified' if 'epoch_sha256' in info else 'absent_in_original_metadata'
    return info, manifest


def validated_validation_info(directory, run_dir):
    path = Path(directory) / 'inference_info.json'
    if not path.exists():
        raise ValueError(f'Missing inference provenance: {path}')
    info = json.loads(path.read_text(encoding='utf-8'))
    if info.get('kind') != 'oof_validation':
        ordinary, _ = validated_inference_info(directory, run_dir, 'validation')
        return ordinary, 'deployed_ensemble' if (Path(run_dir) / 'ensemble.json').exists() else 'deployed_model_validation'
    members, ensemble = ensemble_members(run_dir)
    hashes = ensemble.get('member_checkpoint_sha256', [])
    if len(hashes) != len(members):
        raise ValueError('Ensemble member checkpoint hash table is incomplete')
    expected = dict(zip(members, hashes))
    if info.get('member_checkpoint_sha256') != expected:
        raise ValueError('Out-of-fold checkpoints differ from the deployed ensemble members')
    check_epoch_metadata(directory, info['records'], info['epoch_sha256'])
    sources = info.get('record_sources', {})
    if set(sources) != set(info['records']):
        raise ValueError('Out-of-fold record provenance is incomplete')
    for member, digest in expected.items():
        member_dir = Path(run_dir).parent / member
        if checkpoint_sha256(member_dir) != digest:
            raise ValueError('Out-of-fold member checkpoint hash mismatch')
        config = json.loads((member_dir / 'config.json').read_text(encoding='utf-8'))
        manifest = load_manifest(artifact_path(config['split_manifest']))
        member_records = {r for r, source in sources.items() if source['member'] == member}
        if member_records != set(manifest['partitions']['validation']['records']):
            raise ValueError('Out-of-fold records differ from their fold validation partition')
        if any(participant_id(r) in config['fit_subjects'] for r in member_records):
            raise ValueError('Out-of-fold validation participant was used for model fitting')
        for record in member_records:
            if sources[record]['checkpoint_sha256'] != digest:
                raise ValueError('Out-of-fold record checkpoint hash mismatch')
            if sources[record]['split_manifest_sha256'] != manifest['manifest_sha256']:
                raise ValueError('Out-of-fold split manifest hash mismatch')
    if set(source['member'] for source in sources.values()) != set(expected):
        raise ValueError('Out-of-fold provenance has missing or unknown ensemble members')
    config = json.loads((Path(run_dir) / 'config.json').read_text(encoding='utf-8'))
    if info.get('inference_contract') != inference_contract(config):
        raise ValueError('Out-of-fold inference contract differs from the deployed ensemble')
    return info | {'verified_contract': info['inference_contract'], 'saved_epoch_hash_binding': 'verified'}, 'pooled_fold_single_models'


def variant_name(prior, strength):
    return 'argmax' if prior is None else f'viterbi_{prior}_s{strength:g}'


def all_variants():
    return [(None, 0.0)] + [(prior, s) for prior in PRIORS for s in STRENGTHS]


def partition_records(manifest_path, partition):
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    return list(manifest['partitions'][partition]['records'])


def cmd_infer(args):
    import torch
    torch.set_num_threads(args.threads)
    run_dir = Path(args.run_dir)
    device = torch.device(args.device)
    model, mean, std, snapshot = load_model(run_dir, device)
    records = args.records.split(',') if args.records else partition_records(args.split_manifest, args.partition)
    manifest = load_manifest(args.split_manifest)
    if len(records) != len(set(records)):
        raise ValueError('Repeated inference records are not allowed')
    subjects = scan_dataset(args.data_dir)
    missing = [r for r in records if r not in subjects]
    if missing:
        raise ValueError(f'Records not found in {args.data_dir}: {missing}')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    existing = {p.name.removesuffix('_epochs.npz') for p in out.glob('*_epochs.npz')}
    if existing - set(records):
        raise ValueError('Output directory contains stale inference records')
    seq_len = snapshot.get('SEQ_LEN', 20)
    processing = record_processing(snapshot)
    contract = inference_contract(snapshot)
    timings, began = {}, time.perf_counter()
    for sid in records:
        start = time.perf_counter()
        record = load_record_cached(sid, subjects[sid], args.record_cache, **processing)
        loaded = time.perf_counter()
        _, arrays = infer_record(record, model, mean, std, seq_len, device, 1, 'argmax', None, 1.0,
                                 args.batch_size, contract['sequence_stride'], snapshot['NORMALIZATION_EPSILON'])
        np.savez_compressed(out / f'{sid}_epochs.npz', **arrays)
        timings[sid] = {'load_seconds': loaded - start, 'inference_seconds': time.perf_counter() - loaded,
                        'epoch_positions': int(len(arrays['argmax']))}
        print(f'{sid}: {timings[sid]["epoch_positions"]} epochs, infer {timings[sid]["inference_seconds"]:.1f}s', flush=True)
    (out / 'inference_info.json').write_text(json.dumps({
        'run_dir': str(run_dir), 'checkpoint_sha256': checkpoint_sha256(run_dir),
        'input_mode': 'raw', 'sequence_length': seq_len, 'tta': 1, 'batch_size': args.batch_size,
        'inference_contract': contract, 'normalization_sha256': sha256(run_dir / 'norm_stats.npz'),
        'epoch_sha256': {sid: sha256(out / f'{sid}_epochs.npz') for sid in records},
        'device': str(device),
        'split_manifest': args.split_manifest, 'partition': args.partition, 'records': records,
        'split_manifest_sha256': manifest['manifest_sha256'],
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'timings': timings, 'total_seconds': time.perf_counter() - began,
        'peak_rss_bytes': peak_rss_bytes(), 'environment': environment()}, indent=2), encoding='utf-8')


def load_epochs(directory):
    data = {}
    for path in sorted(Path(directory).glob('*_epochs.npz')):
        with np.load(path, allow_pickle=False) as archive:
            data[path.name.removesuffix('_epochs.npz')] = {k: archive[k] for k in archive.files}
    if not data:
        raise ValueError(f'No epoch NPZ files in {directory}')
    return data


def transition_for(run_dir, prior):
    if prior == 'learned':
        return np.asarray(json.loads((Path(run_dir) / 'transition.json').read_text(encoding='utf-8')))
    return default_transition()


def evaluate_variants(epochs, run_dir, variants, records=None, transitions=None):
    """Per-variant participant confusion matrices for the chosen records."""
    result = {}
    for prior, strength in variants:
        cms, eligible, known = {}, {}, {}
        for sid, a in epochs.items():
            if records is not None and sid not in records:
                continue
            transition = None
            if prior:
                transition = (np.asarray(transitions[sid]) if transitions and prior == 'learned'
                              else transition_for(run_dir, prior))
            decoded = decode_record(a['logits'], a['scorable'], transition, strength) if prior else a['argmax']
            p = participant_id(sid)
            cm = confusion_counts(a['ground_truth'], decoded, a['scorable'])
            cms[p] = cms.get(p, 0) + cm
            eligible[p] = eligible.get(p, 0) + eligible_count(a['ground_truth'], a['scorable'])
            known[p] = known.get(p, 0) + eligible_count(a['ground_truth'])
        result[variant_name(prior, strength)] = (cms, eligible, known)
    return result


def pooled_labeled(epochs, records=None):
    logits, truth = [], []
    for sid, a in epochs.items():
        if records is not None and sid not in records:
            continue
        keep = (a['ground_truth'] >= 0) & a['scorable'] & np.isfinite(a['logits']).all(axis=1)
        logits.append(a['logits'][keep])
        truth.append(a['ground_truth'][keep])
    return np.concatenate(logits), np.concatenate(truth)


def cmd_freeze(args):
    info, calibration_scope = validated_validation_info(args.val_dir, args.run_dir)
    epochs = load_epochs(args.val_dir)
    variants = all_variants()
    # Out-of-fold validation: each record's learned prior comes from its own fold's training participants.
    by_record = Path(args.val_dir) / 'transitions_by_record.json'
    transitions = json.loads(by_record.read_text(encoding='utf-8')) if by_record.exists() else None
    if calibration_scope == 'pooled_fold_single_models':
        if not transitions or set(transitions) != set(epochs):
            raise ValueError('Out-of-fold learned transition provenance is incomplete')
        for sid, source in info['record_sources'].items():
            member_transition = transition_for(Path(args.run_dir).parent / source['member'], 'learned')
            if not np.array_equal(np.asarray(transitions[sid]), member_transition):
                raise ValueError('Out-of-fold learned transition differs from its training fold')
    table = evaluate_variants(epochs, args.run_dir, variants, None, transitions)
    rows = {}
    for name, (cms, _, _) in table.items():
        m = metrics_from_confusion(sum(cms.values()))
        rows[name] = {'accuracy': m['accuracy'], 'macro_f1_5class': m['macro_f1_5class'], 'cohen_kappa': m['cohen_kappa']}
    order = [variant_name(*v) for v in variants]
    best = max(order, key=lambda n: (round(rows[n]['macro_f1_5class'], 6), -order.index(n)))
    logits, truth = pooled_labeled(epochs)
    temperature = fit_temperature(logits, truth)
    calibrated = softmax(logits, temperature)
    thresholds = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    curve = risk_coverage(calibrated, calibrated.argmax(1), truth, thresholds)
    eligible_rows = [r for r in curve if r['accuracy_on_accepted'] is not None and r['coverage'] >= 0.5
                     and r['accuracy_on_accepted'] >= args.target_accuracy]
    abstain = min((r['threshold'] for r in eligible_rows), default=None)
    prior_name = best.split('_')[1] if best != 'argmax' else None
    frozen = {
        'derived_from': 'validation partition only; holdout data are excluded from this computation',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'inference_contract': info['verified_contract'],
        'validation_inference_info_sha256': sha256(Path(args.val_dir) / 'inference_info.json'),
        'validation_saved_epoch_hash_binding': info['saved_epoch_hash_binding'],
        'calibration_prediction_scope': calibration_scope,
        'normalization_sha256': sha256(Path(args.run_dir) / 'norm_stats.npz'),
        'run_dir': str(args.run_dir), 'checkpoint_sha256': checkpoint_sha256(args.run_dir),
        'validation_dir': str(args.val_dir),
        'selection_criterion': 'validation pooled five-class macro F1; ties prefer the earlier (weaker) variant',
        'selected_variant': best, 'selected_prior': prior_name,
        'selected_strength': None if prior_name is None else float(best.rsplit('_s', 1)[1]),
        'validation_variant_scores': rows, 'temperature': temperature,
        'temperature_fit': 'scalar NLL minimisation on validation epochs',
        'abstention': {'target_accuracy_on_accepted': args.target_accuracy, 'minimum_coverage': 0.5,
                       'confidence_threshold': abstain, 'validation_curve': curve,
                       'note': 'Exploratory; confidence-based rejection is not a validated artifact detector.'},
        'validation_calibration_before': calibration_summary(softmax(logits), truth),
        'validation_calibration_after': calibration_summary(calibrated, truth),
        'learned_transition_source': str(Path(args.run_dir) / 'transition.json'),
        'validation_learned_prior': 'per-fold training participants' if transitions else 'run transition.json',
        'transition_sha256': sha256(Path(args.run_dir) / 'transition.json') if prior_name == 'learned' else None,
    }
    Path(args.output).write_text(json.dumps(frozen, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: frozen[k] for k in ('selected_variant', 'temperature')} | {
        'abstention_threshold': abstain, 'best_val': rows[best]}, indent=2))


def plot_reliability(before, after, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, summary, title in ((axes[0], before, 'Uncalibrated'), (axes[1], after, 'Validation-fit temperature')):
        ax.plot([0.2, 1], [0.2, 1], '--', color='#888')
        ax.plot([b['mean_confidence'] for b in summary['bins']], [b['accuracy'] for b in summary['bins']],
                'o-', color='#1e7293')
        ax.set(xlabel='Mean confidence', title=f'{title} (ECE {summary["ece_top_label"]:.3f})')
        ax.grid(alpha=.3)
    axes[0].set_ylabel('Accuracy (holdout)')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def cmd_report(args):
    frozen = json.loads(Path(args.frozen).read_text(encoding='utf-8'))
    if frozen['checkpoint_sha256'] != checkpoint_sha256(args.run_dir):
        raise ValueError('Frozen protocol belongs to a different checkpoint')
    epochs = load_epochs(args.holdout_dir)
    manifest = load_manifest(args.split_manifest)
    holdout_records = set(manifest['partitions']['holdout']['records'])
    if set(epochs) != holdout_records:
        raise ValueError(f'Holdout inference must cover exactly the holdout records: {sorted(set(epochs) ^ holdout_records)}')
    info, _ = validated_inference_info(args.holdout_dir, args.run_dir, 'holdout', args.split_manifest)
    if frozen.get('inference_contract') is not None and frozen['inference_contract'] != info['verified_contract']:
        raise ValueError('Holdout inference differs from the frozen inference contract')
    if frozen.get('transition_sha256') is not None:
        if frozen['transition_sha256'] != sha256(Path(args.run_dir) / 'transition.json'):
            raise ValueError('Learned transition differs from the frozen protocol')
    if frozen.get('normalization_sha256') is not None:
        if frozen['normalization_sha256'] != sha256(Path(args.run_dir) / 'norm_stats.npz'):
            raise ValueError('Normalization differs from the frozen protocol')
    variants = all_variants()
    names = {variant_name(*v): v for v in variants}
    selected = frozen['selected_variant']
    scopes = {'holdout_all': None,
              'holdout_ST_cohort': {r for r in holdout_records if r.startswith('ST')},
              'holdout_SC_cohort': {r for r in holdout_records if r.startswith('SC')}}
    diagnostic_text = getattr(args, 'diagnostic_records', None)
    if diagnostic_text is not None:
        diagnostic_records = [r.strip() for r in diagnostic_text.split(',') if r.strip()]
        if not diagnostic_records or len(diagnostic_records) != len(set(diagnostic_records)):
            raise ValueError('Diagnostic records must be nonempty and unique')
        if not set(diagnostic_records) <= holdout_records:
            raise ValueError('Diagnostic records must be a subset of the declared holdout')
        scopes['holdout_requested_records'] = set(diagnostic_records)
    result = {'purpose': 'Participant-disjoint holdout evaluation of a freshly trained model; thresholds and decoder '
                         'were frozen on validation before holdout inference.',
              'run_dir': str(args.run_dir), 'checkpoint_sha256': frozen['checkpoint_sha256'],
              'split_manifest_sha256': manifest['manifest_sha256'], 'frozen_protocol': args.frozen,
              'frozen_protocol_sha256': sha256(args.frozen), 'selected_variant': selected, 'scopes': {}}
    result['inference_provenance'] = {
        'inference_info_sha256': sha256(Path(args.holdout_dir) / 'inference_info.json'),
        'checkpoint_split_and_records_verified': True,
        'saved_epoch_hash_binding': info['saved_epoch_hash_binding'],
        'protocol_inference_contract_bound': 'inference_contract' in frozen,
        'inference_contract': info['verified_contract']}
    base_cm = {}
    for scope, subset in scopes.items():
        if subset is not None and not subset & set(epochs):
            result['scopes'][scope] = {'records': [], 'note': 'No inference records in this scope'}
            continue
        table = evaluate_variants(epochs, args.run_dir, variants, subset)
        entry = {'records': sorted(subset if subset is not None else holdout_records), 'variants': {}}
        for name, (cms, eligible, known) in table.items():
            entry['variants'][name] = summarize_variant(cms, eligible, args.n_boot, args.seed, known)
        entry['selected_variant_metrics'] = entry['variants'][selected]
        entry['argmax_metrics'] = entry['variants']['argmax']
        entry['heuristic_viterbi_s1_metrics'] = entry['variants']['viterbi_heuristic_s1']
        base_cm[scope] = {n: table[n][0] for n in ('argmax', 'viterbi_heuristic_s1', selected)}
        result['scopes'][scope] = entry
    # Decoder effect on stage transitions for the selected and baseline decoders (holdout_all).
    diagnostics = {}
    for name in ('argmax', 'viterbi_heuristic_s1', 'viterbi_learned_s1', selected):
        prior, strength = names[name]
        counts = None
        stage_changes = 0
        for sid, a in epochs.items():
            decoded = decode_record(a['logits'], a['scorable'], transition_for(args.run_dir, prior), strength) if prior else a['argmax']
            d = transition_diagnostics(decoded, a['scorable'])
            stage_changes += d['stage_changes']
        diagnostics[name] = {'stage_changes_total': stage_changes}
    truth_changes = 0
    for a in epochs.values():
        keep = a['scorable'] & (a['ground_truth'] >= 0)
        truth_changes += transition_diagnostics(np.where(keep, a['ground_truth'], 0), keep)['stage_changes']
    diagnostics['reference_known_stage_changes'] = truth_changes
    result['stage_change_diagnostics'] = diagnostics
    # Calibration and abstention on holdout, frozen temperature.
    logits, truth = pooled_labeled(epochs)
    temperature = frozen['temperature']
    before, after = calibration_summary(softmax(logits), truth), calibration_summary(softmax(logits, temperature), truth)
    thresholds = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    calibrated = softmax(logits, temperature)
    result['calibration'] = {'temperature_frozen_on_validation': temperature, 'before': before, 'after': after,
                             'risk_coverage_argmax_temperature_scaled': risk_coverage(
                                 calibrated, calibrated.argmax(1), truth, thresholds),
                             'frozen_abstention_threshold': frozen['abstention']['confidence_threshold']}
    scope = frozen.get('calibration_prediction_scope')
    if scope is None and (Path(args.run_dir) / 'ensemble.json').exists():
        scope = 'pooled_fold_single_models' if frozen.get('validation_learned_prior') == 'per-fold training participants' else 'unrecorded'
    result['calibration']['prediction_scope_fit'] = scope or 'unrecorded'
    result['calibration']['deployed_ensemble_calibration_validated'] = (
        scope == 'deployed_ensemble' if (Path(args.run_dir) / 'ensemble.json').exists() else None)
    if (Path(args.run_dir) / 'ensemble.json').exists() and scope != 'deployed_ensemble':
        result['calibration']['limitation'] = ('Temperature and abstention were fitted to another prediction '
            'distribution; transfer to mean-logit ensemble predictions is exploratory and unvalidated.')
    thr = frozen['abstention']['confidence_threshold']
    if thr is not None:
        result['calibration']['at_frozen_threshold'] = risk_coverage(calibrated, calibrated.argmax(1), truth, [thr])[0]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    plot_reliability(before, after, out.with_name(out.stem + '_reliability.png'))
    short = {s: {n: {k: result['scopes'][s]['variants'][n][k] for k in ('n', 'accuracy', 'macro_f1_5class')}
                 for n in ('argmax', 'viterbi_heuristic_s1', selected)} for s in scopes if 'variants' in result['scopes'][s]}
    print(json.dumps(short, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    infer = sub.add_parser('infer')
    infer.add_argument('--run_dir', required=True)
    infer.add_argument('--split_manifest', default='reports/split_manifest.json')
    infer.add_argument('--partition', choices=['validation', 'holdout', 'train'])
    infer.add_argument('--records', help='Comma-separated record IDs instead of a partition')
    infer.add_argument('--data_dir', default='data/raw/train')
    infer.add_argument('--record_cache', default='cache/records_raw')
    infer.add_argument('--output_dir', required=True)
    infer.add_argument('--threads', type=int, default=8)
    infer.add_argument('--device', default='cpu', help='cpu or cuda; the frozen analysis is device-agnostic')
    infer.add_argument('--batch_size', type=int, default=64)
    freeze = sub.add_parser('freeze')
    freeze.add_argument('--run_dir', required=True)
    freeze.add_argument('--val_dir', required=True)
    freeze.add_argument('--output', required=True)
    freeze.add_argument('--target_accuracy', type=float, default=0.90)
    report = sub.add_parser('report')
    report.add_argument('--run_dir', required=True)
    report.add_argument('--holdout_dir', required=True)
    report.add_argument('--frozen', required=True)
    report.add_argument('--split_manifest', default='reports/split_manifest.json')
    report.add_argument('--output', required=True)
    report.add_argument('--n_boot', type=int, default=2000)
    report.add_argument('--seed', type=int, default=42)
    report.add_argument('--diagnostic_records',
                        help='Optional explicit comma-separated holdout record subset; descriptive diagnostics only.')
    args = parser.parse_args()
    if args.command == 'infer':
        if bool(args.records) == bool(args.partition):
            parser.error('Provide exactly one of --records or --partition')
        cmd_infer(args)
    elif args.command == 'freeze':
        cmd_freeze(args)
    else:
        cmd_report(args)


if __name__ == '__main__':
    main()
