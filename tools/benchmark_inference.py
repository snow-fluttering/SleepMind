"""CPU/CUDA inference benchmark: direct overlapping windows vs shared epoch encoding.

Every configuration runs in a fresh worker process so peak resident memory is
isolated. Timings use ``time.perf_counter`` after one untimed warmup call and
report median/min/max over the requested repetitions. CUDA is measured only if
``torch.cuda.is_available()``; hardware and device availability are recorded.
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def worker(args):
    import numpy as np
    import torch
    from data_preprocessing import scan_dataset, load_record
    from decoding import viterbi_decode
    from predict import load_model, predict_epochs
    from sysinfo import peak_rss_bytes
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    model, mean, std, config = load_model(args.run_dir, device)
    seq_len, stride = config.get('SEQ_LEN', 20), config.get('SEQ_LEN', 20) // 2
    parts = scan_dataset(args.data_dir)[args.record]
    began = time.perf_counter()
    record = load_record(parts, input_mode='raw')  # text parse, part join and one zero-phase filter
    load_filter_seconds = time.perf_counter() - began
    epoch_len = 3000
    frames = np.asarray(record['eeg'])[:args.epochs * epoch_len].reshape(-1, epoch_len)
    frames = ((frames - mean) / (std + config['NORMALIZATION_EPSILON'])).astype(np.float32)
    n = len(frames)

    def direct():
        """Re-encode every overlapping window for the shared-encoder ablation."""
        starts = list(range(0, n - seq_len + 1, stride))
        if starts[-1] != n - seq_len:
            starts.append(n - seq_len)
        total, count = np.zeros((n, 5)), np.zeros(n)
        with torch.inference_mode():
            for begin in range(0, len(starts), args.batch_size):
                chunk = starts[begin:begin + args.batch_size]
                batch = torch.from_numpy(np.stack([frames[s:s + seq_len] for s in chunk])).to(device)
                out = model(batch).cpu().numpy()
                for s, o in zip(chunk, out):
                    total[s:s + seq_len] += o
                    count[s:s + seq_len] += 1
        return (total / count[:, None]).astype(np.float32)

    def shared():
        return predict_epochs(frames, model, seq_len, device, 1, stride, args.batch_size)[0]

    def shared_bf16():
        with torch.autocast(device.type, dtype=torch.bfloat16):
            return predict_epochs(frames, model, seq_len, device, 1, stride, args.batch_size)[0]

    functions = {'direct_windows': direct, 'shared_encoding': shared, 'shared_encoding_bf16_autocast': shared_bf16}
    run = functions[args.method]
    reference = shared() if args.method != 'shared_encoding' else None
    run()  # warmup
    sync = torch.cuda.synchronize if device.type == 'cuda' else (lambda: None)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    times, output = [], None
    for _ in range(args.repetitions):
        sync()
        began = time.perf_counter()
        output = run()
        sync()
        times.append(time.perf_counter() - began)
    result = {'method': args.method, 'threads': args.threads, 'batch_size': args.batch_size, 'epochs': n,
              'device': str(device), 'repetitions': args.repetitions, 'warmup_runs': 1, 'tta': 1,
              'seconds_median': statistics.median(times), 'seconds_min': min(times), 'seconds_max': max(times),
              'epochs_per_second_median': n / statistics.median(times),
              'load_parse_filter_seconds_once': load_filter_seconds, 'peak_rss_bytes': peak_rss_bytes(),
              'peak_cuda_memory_bytes': int(torch.cuda.max_memory_allocated()) if device.type == 'cuda' else None,
              'members': len(getattr(model, 'members', [model]))}
    if reference is not None:
        result['argmax_disagreements_vs_shared_fp32'] = int((output.argmax(1) != reference.argmax(1)).sum())
        result['max_abs_logit_difference_vs_shared_fp32'] = float(np.abs(output - reference).max())
    else:
        began = time.perf_counter()
        direct_output = direct()
        result['direct_reference_seconds'] = time.perf_counter() - began
        result['argmax_disagreements_vs_direct'] = int((output.argmax(1) != direct_output.argmax(1)).sum())
        result['max_abs_logit_difference_vs_direct'] = float(np.abs(output - direct_output).max())
        began = time.perf_counter()
        viterbi_decode(output, None, 1.0)
        result['viterbi_decode_seconds'] = time.perf_counter() - began
    print('RESULT ' + json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--data_dir', default='data/raw/train')
    parser.add_argument('--record', required=True, help='Explicit record ID from the supplied data directory.')
    parser.add_argument('--epochs', type=int, default=1100)
    parser.add_argument('--threads', default='1,4,8')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--output', default=None)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--method', default='shared_encoding')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--devices', default='cpu', help='Comma-separated devices to sweep (cpu,cuda).')
    args = parser.parse_args()
    if args.worker:
        args.threads = int(args.threads)
        worker(args)
        return
    from sysinfo import environment
    from predict import checkpoint_sha256
    checkpoint_before = checkpoint_sha256(Path(args.run_dir))
    results = []
    plan = [(device, int(t)) for device in args.devices.split(',') for t in
            (args.threads.split(',') if device == 'cpu' else ['1'])]
    for device, threads in plan:
        for method in ('direct_windows', 'shared_encoding', 'shared_encoding_bf16_autocast'):
            command = [sys.executable, __file__, '--worker', '--run_dir', args.run_dir, '--data_dir', args.data_dir,
                       '--record', args.record, '--epochs', str(args.epochs), '--threads', str(threads),
                       '--batch_size', str(args.batch_size), '--repetitions', str(args.repetitions),
                       '--method', method, '--device', device]
            completed = subprocess.run(command, capture_output=True, text=True, check=True)
            line = next(l for l in completed.stdout.splitlines() if l.startswith('RESULT '))
            results.append(json.loads(line[7:]))
            print(device, results[-1]['method'], threads, round(results[-1]['seconds_median'], 3), flush=True)
    speedups = {}
    for device, threads in plan:
        by = {r['method']: r for r in results if r['threads'] == threads and r['device'].startswith(device)}
        speedups[f'{device}_threads{threads}'] = {
            'shared_over_direct': by['direct_windows']['seconds_median'] / by['shared_encoding']['seconds_median'],
            'bf16_over_shared_fp32': by['shared_encoding']['seconds_median'] / by['shared_encoding_bf16_autocast']['seconds_median']}
    measured_cuda = any(r['device'].startswith('cuda') for r in results)
    if checkpoint_sha256(Path(args.run_dir)) != checkpoint_before:
        raise ValueError('Checkpoint changed during the benchmark')
    report = {'scope': 'Encoder + temporal inference on one record slice; text parse/filter reported separately; '
                       'decoding and report rendering are timed in the pipeline benchmark.',
              'run_dir': args.run_dir, 'checkpoint_sha256': checkpoint_before,
              'record': args.record, 'environment': environment(),
              'results': results, 'speedups_median': speedups,
              'gpu_note': ('CUDA measurements are included for the recorded GPU.' if measured_cuda else
                           'No CUDA configuration was measured; no GPU performance claim is made.')}
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(speedups, indent=2))


if __name__ == '__main__':
    main()
