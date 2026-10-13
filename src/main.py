"""Command dispatcher. Run from the project root with python src/main.py."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['train', 'predict', 'evaluate', 'all'])
    args, remaining = parser.parse_known_args()
    scripts = {'train': 'train.py', 'predict': 'predict.py', 'evaluate': 'utils.py'}
    stages = ['train', 'predict'] if args.stage == 'all' else [args.stage]
    for stage in stages:
        forwarded = remaining
        if args.stage == 'all':
            common = {'--run_name', '--input_mode', '--batch_size', '--threads'}
            train_values = {'--train_dir', '--epochs', '--lr', '--weight_decay', '--seq_len', '--val_ratio', '--seed', '--device', '--split_manifest',
                            '--fit_partitions', '--infer_batch_size', '--label_smoothing', '--cnn_dropout', '--fc_dropout', '--patience', '--record_cache'}
            predict_values = {'--seq_stride', '--protocol', '--test_dir', '--output_dir', '--decoder', '--transition_path', '--transition_strength', '--part_offsets', '--tta', '--record_id', '--source_file', '--unit'}
            train_flags = {'--no_cache', '--no_focal', '--no_augment', '--no_class_weights', '--resume', '--set_latest'}
            allowed = common | (train_values | train_flags if stage == 'train' else predict_values)
            forwarded = []
            index = 0
            while index < len(remaining):
                token = remaining[index]
                name = token.split('=', 1)[0]
                if name not in common | train_values | predict_values | train_flags:
                    parser.error(f'Unknown all-stage option: {name}')
                width = 1 if '=' in token or name in train_flags else 2
                if width == 2 and index + 1 >= len(remaining):
                    parser.error(f'Missing value for {name}')
                if name in allowed:
                    forwarded.extend(remaining[index:index+width])
                index += width
        subprocess.run([sys.executable, str(Path(__file__).with_name(scripts[stage])), *forwarded], check=True)


if __name__ == '__main__':
    main()
