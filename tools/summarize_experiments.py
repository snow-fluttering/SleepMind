"""Collect per-run validation results, provenance and timings into one machine-readable table."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize_run(run_dir):
    run = Path(run_dir)
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    log = json.loads((run / 'train_log.json').read_text(encoding='utf-8'))
    info = json.loads((run / 'run_info.json').read_text(encoding='utf-8')) if (run / 'run_info.json').exists() else {}
    best = log['best_pass']
    row = {
        'run': run.name, 'best_pass': best, 'passes_completed': len(log['train_loss']),
        'fit_partitions': config.get('fit_partitions'), 'seed': config['SEED'],
        'recipe': {key: config[key] for key in (
            'USE_FOCAL_LOSS', 'FOCAL_GAMMA', 'LABEL_SMOOTHING', 'USE_CLASS_WEIGHTS', 'LEARNING_RATE', 'WEIGHT_DECAY',
            'CNN_DROPOUT', 'FC_DROPOUT', 'SEQ_LEN', 'BATCH_SIZE', 'NUM_EPOCHS', 'USE_DATA_AUGMENT')},
        'ema_decay': config.get('ema_decay', 0.0),
        'split_manifest_sha256': config.get('split_manifest_sha256'),
        'checkpoint_sha256': hashlib.sha256((run / 'best_model.pth').read_bytes()).hexdigest(),
        'mean_pass_seconds': info.get('mean_pass_seconds'), 'wall_seconds': info.get('wall_seconds_this_invocation'),
        'peak_rss_bytes': info.get('peak_rss_bytes'), 'threads': config.get('threads'),
        'data_manifest_sha256': config.get('data_manifest_sha256'),
    }
    if log.get('val_macro_f1'):
        index = best - 1
        row['validation_at_best_pass'] = {key: log[name][index] for key, name in (
            ('accuracy', 'val_accuracy'), ('macro_f1_5class', 'val_macro_f1'), ('cohen_kappa', 'val_kappa'),
            ('negative_log_likelihood', 'val_nll'))}
        row['validation_last_pass'] = {key: log[name][-1] for key, name in (
            ('accuracy', 'val_accuracy'), ('macro_f1_5class', 'val_macro_f1'))}
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows = [summarize_run(run) for run in args.runs]
    Path(args.output).write_text(json.dumps({'runs': rows}, indent=2, allow_nan=False), encoding='utf-8')
    for row in rows:
        v = row.get('validation_at_best_pass', {})
        print(row['run'], 'best pass', row['best_pass'], {k: round(x, 4) for k, x in v.items()})


if __name__ == '__main__':
    main()
