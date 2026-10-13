"""Warm the per-record joined signal cache (raw mode filters each continuous block once)."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from data_preprocessing import scan_dataset, load_record_cached


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', default='data/raw/train')
    parser.add_argument('--input_mode', choices=['raw', 'preprocessed'], default='raw')
    parser.add_argument('--cache_dir', default='cache/records_raw')
    parser.add_argument('--output', default=None, help='Optional JSON timing report')
    args = parser.parse_args()
    rows = []
    for sid, parts in sorted(scan_dataset(args.data_dir).items()):
        began = time.perf_counter()
        record = load_record_cached(sid, parts, args.cache_dir, input_mode=args.input_mode)
        rows.append({'record': sid, 'samples': int(len(record['eeg'])), 'seconds': time.perf_counter() - began})
        print(f"{sid}: {rows[-1]['samples']} samples, {rows[-1]['seconds']:.1f}s", flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(rows, indent=2), encoding='ascii')


if __name__ == '__main__':
    main()
