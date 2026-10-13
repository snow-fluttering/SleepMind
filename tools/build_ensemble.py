"""Bundle several trained runs into a mean-logit ensemble directory usable by predict/evaluate tools.

The bundle stores no weights of its own: ``ensemble.json`` lists member runs (siblings in the runs
directory) and their checkpoint hashes. Normalization of the first member defines the bundle input scale;
every member is converted back to its own normalization exactly. The bundle's learned transition prior is
fit on the contiguous training runs of all participants that the members were trained on (development
participants only; the holdout is never read).
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data_preprocessing import build_training_set, participant_id
from decoding import estimate_transition
from predict import checkpoint_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--members', required=True, help='Comma-separated member run names')
    parser.add_argument('--name', required=True)
    parser.add_argument('--runs_dir', default='runs')
    parser.add_argument('--train_dir', default='data/raw/train')
    parser.add_argument('--record_cache', default='cache/records_raw')
    args = parser.parse_args()
    runs = Path(args.runs_dir)
    members = args.members.split(',')
    configs = [json.loads((runs / m / 'config.json').read_text(encoding='utf-8')) for m in members]
    for key in ('FEATURE_DIM', 'LSTM_HIDDEN', 'TRANSFORMER_LAYERS', 'TRANSFORMER_HEADS', 'SEQ_LEN', 'INPUT_MODE'):
        if len({c[key] for c in configs}) != 1:
            raise ValueError(f'Members disagree on {key}')
    manifests = {c['split_manifest_sha256'] for c in configs}
    fit = sorted({p for c in configs for p in c['fit_subjects']})
    holdout_seen = [m for m in members if json.loads((runs / m / 'split.json').read_text())['holdout_loaded']]
    if holdout_seen:
        raise ValueError(f'Members that loaded the holdout: {holdout_seen}')
    out = runs / args.name
    out.mkdir(exist_ok=False)
    shutil.copy(runs / members[0] / 'norm_stats.npz', out / 'norm_stats.npz')
    records = sorted({r.split('::')[0] for m in members for r in
                      json.loads((runs / m / 'split.json').read_text())['fit_record_runs']})
    processing = configs[0]['processing_parameters']
    X, y, sids = build_training_set(args.train_dir, **processing, cache_path=None, records=records,
                                    record_cache_dir=args.record_cache, verbose=False)
    keep = np.isin([participant_id(s) for s in sids], fit)
    transition = estimate_transition(y[keep], sids[keep], smoothing=1.0)
    (out / 'transition.json').write_text(json.dumps(transition.tolist(), indent=2), encoding='utf-8')
    config = dict(configs[0])
    config.update({
        'run_name': args.name, 'ENSEMBLE_MEMBERS': members, 'fit_subjects': fit,
        'initialization': 'ensemble of fresh random-initialization members',
        'ensemble_aggregation': 'mean of member logits (each member sees its own normalization)',
        'member_split_manifest_sha256': sorted(manifests),
        'transition_prior_source': 'Contiguous runs of all member-training participants (development only).',
    })
    (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    description = {'members': members, 'aggregation': 'mean_logits',
                   'member_checkpoint_sha256': [hashlib.sha256((runs / m / 'best_model.pth').read_bytes()).hexdigest()
                                                for m in members]}
    (out / 'ensemble.json').write_text(json.dumps(description, indent=2), encoding='utf-8')
    description['bundle_checkpoint_sha256'] = checkpoint_sha256(out)
    (out / 'ensemble.json').write_text(json.dumps(description, indent=2), encoding='utf-8')
    print(json.dumps({'bundle': str(out), 'members': members, 'fit_participants': len(fit),
                      'bundle_checkpoint_sha256': description['bundle_checkpoint_sha256']}, indent=2))


if __name__ == '__main__':
    main()
