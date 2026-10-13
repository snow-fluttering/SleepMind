"""Quality/cost ablation of inference context, stride and test-time augmentation on a given partition.

Only run this on the validation partition when choosing settings. Each configuration reports pooled
accuracy and five-class macro F1 of argmax predictions (each original epoch once) and wall-clock time.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data_preprocessing import scan_dataset, load_record_cached, participant_id
from evaluation import confusion_counts, metrics_from_confusion
from predict import load_model, infer_record
from sysinfo import environment

CONFIGS = [  # (sequence length, stride, TTA shifts)
    (20, 10, 1), (20, 5, 1), (20, 20, 1), (40, 20, 1), (60, 30, 1), (20, 10, 3), (20, 10, 5)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--split_manifest', default='reports/split_manifest.json')
    parser.add_argument('--partition', default='validation', choices=['validation', 'train', 'holdout'])
    parser.add_argument('--data_dir', default='data/raw/train')
    parser.add_argument('--record_cache', default='cache/records_raw')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    manifest = json.loads(Path(args.split_manifest).read_text(encoding='utf-8'))
    records = manifest['partitions'][args.partition]['records']
    subjects = scan_dataset(args.data_dir)
    device = torch.device(args.device)
    model, mean, std, snapshot = load_model(args.run_dir, device)
    loaded = {sid: load_record_cached(sid, subjects[sid], args.record_cache, input_mode='raw') for sid in records}
    rows = []
    for seq_len, stride, tta in CONFIGS:
        cm, began = 0, time.perf_counter()
        for sid, record in loaded.items():
            _, arrays = infer_record(record, model, mean, std, seq_len, device, tta, 'argmax', None, 1.0,
                                     args.batch_size, stride, snapshot['NORMALIZATION_EPSILON'])
            cm = cm + confusion_counts(arrays['ground_truth'], arrays['argmax'], arrays['scorable'])
        seconds = time.perf_counter() - began
        metrics = metrics_from_confusion(cm)
        rows.append({'sequence_length': seq_len, 'stride': stride, 'tta_shifts': tta, 'seconds': seconds,
                     'epochs': metrics['n'], 'accuracy': metrics['accuracy'], 'macro_f1_5class': metrics['macro_f1_5class'],
                     'cohen_kappa': metrics['cohen_kappa']})
        print(rows[-1], flush=True)
    result = {'run_dir': args.run_dir, 'partition': args.partition, 'trained_sequence_length': snapshot.get('SEQ_LEN'),
              'threads': args.threads, 'device': args.device, 'environment': environment(), 'configurations': rows,
              'note': 'Selection must use the validation partition; the holdout is evaluated once with the frozen choice.'}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
