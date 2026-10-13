"""Train candidate recipes on every cross-validation fold and save out-of-fold validation inference.

Runs are resumable: a fold with a finished ``run_info.json`` and saved out-of-fold epochs is skipped.
Every model starts from fresh random initialization; the holdout partition is never loaded.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

COMMON = ['--input_mode', 'raw', '--record_cache', 'cache/records_raw', '--epochs', '40', '--seed', '42',
          '--threads', '8', '--device', 'cuda']
RECIPES = {  # name -> extra train.py arguments relative to base recipe B
    'B': [],
    'EMA': ['--ema_decay', '0.995'],
    'CE': ['--no_focal'],
    'REG': ['--weight_decay', '0.01', '--cnn_dropout', '0.6'],
    'L40': ['--seq_len', '40', '--batch_size', '16'],
    'P80': ['--epochs', '80'],
}


def run(command):
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode:
        raise SystemExit(f'Command failed ({completed.returncode}): {" ".join(map(str, command))}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipes', default=','.join(RECIPES))
    parser.add_argument('--folds', default='0,1,2,3,4')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', default='reports/cv_runs')
    args = parser.parse_args()
    python = sys.executable
    for name in args.recipes.split(','):
        for fold in [int(f) for f in args.folds.split(',')]:
            run_name = f'cv_{name}_f{fold}'
            run_dir = ROOT / 'runs' / run_name
            epochs_dir = ROOT / args.out / name / f'f{fold}_val_epochs'
            if not (run_dir / 'run_info.json').exists():
                began = time.perf_counter()
                command = [python, 'src/train.py', *COMMON, '--device', args.device,
                           '--split_manifest', f'reports/cv/fold_{fold}.json', '--run_name', run_name]
                # argparse keeps the last occurrence, so recipe arguments appended here override COMMON.
                run(command + RECIPES[name])
                print(f'[cv] trained {run_name} in {time.perf_counter() - began:.0f}s', flush=True)
            if not (epochs_dir / 'inference_info.json').exists():
                run([python, 'tools/evaluate_run.py', 'infer', '--run_dir', f'runs/{run_name}',
                     '--split_manifest', f'reports/cv/fold_{fold}.json', '--partition', 'validation',
                     '--device', args.device, '--output_dir', str(epochs_dir.relative_to(ROOT))])
            print(f'[cv] {run_name} complete', flush=True)
    summary = {'recipes': {k: v for k, v in RECIPES.items() if k in args.recipes.split(',')}, 'common': COMMON}
    Path(ROOT / args.out).mkdir(parents=True, exist_ok=True)
    Path(ROOT / args.out / 'recipes.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
