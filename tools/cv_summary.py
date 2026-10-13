"""Pool out-of-fold validation inference per recipe and summarize cross-validated quality.

Writes, per recipe, ``oof_epochs/`` (all development records, each predicted by the model that never saw
its participant) with ``transitions_by_record.json`` (each fold's own training-only learned prior), and a
JSON summary with participant-level bootstrap intervals and paired differences against a reference recipe.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_preprocessing import participant_id
from evaluation import confusion_counts, eligible_count, summarize_variant, paired_difference_bootstrap
from evaluate_run import load_epochs, validated_inference_info, sha256


def pool(recipe_dir, runs_dir, recipe, folds):
    target = recipe_dir / 'oof_epochs'
    transitions, files, record_sources, checkpoints = {}, {}, {}, {}
    contract, holdout, development = None, None, None
    if len(folds) != len(set(folds)):
        raise ValueError('Duplicate cross-validation folds')
    for fold in folds:
        source = recipe_dir / f'f{fold}_val_epochs'
        member = f'cv_{recipe}_f{fold}'
        run = runs_dir / member
        info, manifest = validated_inference_info(source, run, 'validation')
        current_holdout = set(manifest['partitions']['holdout']['records'])
        current_development = (set(manifest['partitions']['train']['records']) |
                               set(manifest['partitions']['validation']['records']))
        if contract is not None and contract != info['verified_contract']:
            raise ValueError('Cross-validation inference contracts differ')
        if holdout is not None and (holdout != current_holdout or development != current_development):
            raise ValueError('Cross-validation development or holdout partitions differ')
        contract, holdout, development = info['verified_contract'], current_holdout, current_development
        checkpoints[member] = info['checkpoint_sha256']
        matrix = json.loads((run / 'transition.json').read_text(encoding='utf-8'))
        for path in source.glob('*_epochs.npz'):
            record = path.name.removesuffix('_epochs.npz')
            if record in files:
                raise ValueError(f'Duplicate out-of-fold validation record: {record}')
            files[record] = path
            transitions[record] = matrix
            record_sources[record] = {'member': member, 'fold': fold,
                'checkpoint_sha256': info['checkpoint_sha256'],
                'split_manifest_sha256': manifest['manifest_sha256'],
                'source_inference_info_sha256': sha256(source / 'inference_info.json'),
                'source_saved_epoch_hash_binding': info['saved_epoch_hash_binding']}
    if not files:
        raise ValueError('No out-of-fold validation records')
    target.mkdir(exist_ok=True)
    existing = {p.name.removesuffix('_epochs.npz') for p in target.glob('*_epochs.npz')}
    if existing - set(files):
        raise ValueError('Out-of-fold output contains stale records; use a fresh output directory')
    for record, path in files.items():
        shutil.copy(path, target / path.name)
    (target / 'transitions_by_record.json').write_text(json.dumps(transitions), encoding='ascii')
    (target / 'inference_info.json').write_text(json.dumps({
        'kind': 'oof_validation', 'partition': 'validation', 'records': sorted(files),
        'record_sources': record_sources, 'member_checkpoint_sha256': checkpoints,
        'inference_contract': contract,
        'epoch_sha256': {record: sha256(target / path.name) for record, path in files.items()},
        'note': 'Each record is predicted only by its own participant-disjoint validation-fold model; '
                'pooled single-model calibration is not deployed-ensemble calibration.'}, indent=2), encoding='utf-8')
    return target


def oof_confusions(epochs):
    cms, eligible, known = {}, {}, {}
    for sid, a in epochs.items():
        p = participant_id(sid)
        cms[p] = cms.get(p, 0) + confusion_counts(a['ground_truth'], a['argmax'], a['scorable'])
        eligible[p] = eligible.get(p, 0) + eligible_count(a['ground_truth'], a['scorable'])
        known[p] = known.get(p, 0) + eligible_count(a['ground_truth'])
    return cms, eligible, known


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipes', default='B,EMA,CE,REG,L40,P80')
    parser.add_argument('--reference', default='B')
    parser.add_argument('--cv_runs', default='reports/cv_runs')
    parser.add_argument('--runs_dir', default='runs')
    parser.add_argument('--folds', default='0,1,2,3,4')
    parser.add_argument('--n_boot', type=int, default=2000)
    parser.add_argument('--output', default='reports/cv_summary.json')
    args = parser.parse_args()
    folds = [int(f) for f in args.folds.split(',')]
    base, results, cms_by = Path(args.cv_runs), {}, {}
    for recipe in args.recipes.split(','):
        target = pool(base / recipe, Path(args.runs_dir), recipe, folds)
        epochs = load_epochs(target)
        cms, eligible, known = oof_confusions(epochs)
        cms_by[recipe] = cms
        summary = summarize_variant(cms, eligible, args.n_boot, 0, known)
        fold_scores = []
        for fold in folds:
            log = json.loads((Path(args.runs_dir) / f'cv_{recipe}_f{fold}' / 'train_log.json').read_text(encoding='utf-8'))
            fold_scores.append({'fold': fold, 'best_pass': log['best_pass'], 'val_macro_f1': max(log['val_macro_f1']),
                                'passes': len(log['train_loss'])})
        results[recipe] = {'oof_argmax': {k: summary[k] for k in (
            'n', 'accuracy', 'macro_f1_5class', 'cohen_kappa', 'per_stage', 'known_reference_epochs',
            'eligible_reference_epochs', 'coverage_of_known_reference', 'confidence_intervals_95_participant_bootstrap')},
            'per_participant': summary['per_participant'], 'folds': fold_scores,
            'mean_fold_best_pass_macro_f1': float(np.mean([f['val_macro_f1'] for f in fold_scores]))}
        print(recipe, round(summary['accuracy'], 4), round(summary['macro_f1_5class'], 4), flush=True)
    ref = args.reference
    for recipe in results:
        if recipe != ref:
            for key in ('accuracy', 'macro_f1_5class'):
                results[recipe][f'paired_{key}_vs_{ref}'] = paired_difference_bootstrap(
                    cms_by[recipe], cms_by[ref], key, args.n_boot, 0)
    out = {'reference_recipe': ref, 'scope': 'Out-of-fold argmax, 18 development participants, raw input, TTA=1',
           'selection_rule': 'highest pooled OOF five-class macro F1; differences below 0.005 are ties -> simpler recipe',
           'recipes': results}
    Path(args.output).write_text(json.dumps(out, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
