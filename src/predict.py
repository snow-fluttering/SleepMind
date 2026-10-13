"""Reproducible offline inference with shared epoch encoding, gaps and reports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from config import get_config
from model import SeqSleepStagingModel
from data_preprocessing import load_record, scan_dataset, parse_label_file, map_description_to_label
from decoding import viterbi_decode, default_transition
from reporting import validate_segments, write_report


def resolve_run_dir(runs_dir, run_name=None):
    root = Path(runs_dir)
    if run_name:
        result = root / run_name
    elif (root / 'latest.txt').exists():
        result = root / (root / 'latest.txt').read_text(encoding='utf-8').strip()
    else:
        candidates = sorted(root.glob('run_*'), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError('No saved run found')
        result = candidates[0]
    if not result.is_dir():
        raise FileNotFoundError(f'Run directory does not exist: {result}')
    return str(result)


class _EnsembleEncoder(torch.nn.Module):
    """Concatenate member epoch features; each member receives its own normalization exactly."""

    def __init__(self, members, bundle_mean, bundle_std):
        super().__init__()
        self.encoders = torch.nn.ModuleList([m.epoch_enc for m, _, _ in members])
        # Input arrives as (x - bundle_mean) / bundle_std; member i expects (x - mean_i) / std_i.
        self.scale = [bundle_std / std for _, _, std in members]
        self.shift = [(bundle_mean - mean) / std for _, mean, std in members]

    def forward(self, x):
        return torch.cat([encoder(x * scale + shift) for encoder, scale, shift in
                          zip(self.encoders, self.scale, self.shift)], dim=-1)


class _EnsembleSequence(torch.nn.Module):
    def __init__(self, members):
        super().__init__()
        self.encoders = torch.nn.ModuleList([m.seq_enc for m, _, _ in members])

    def forward(self, h):
        return torch.cat([enc(part) for enc, part in zip(self.encoders, h.chunk(len(self.encoders), dim=-1))], dim=-1)


class _EnsembleHead(torch.nn.Module):
    def __init__(self, members):
        super().__init__()
        self.heads = torch.nn.ModuleList([m.classifier for m, _, _ in members])

    def forward(self, h):
        parts = h.chunk(len(self.heads), dim=-1)
        return torch.stack([head(part) for head, part in zip(self.heads, parts)]).mean(dim=0)


class EnsembleModel(torch.nn.Module):
    """Mean-logit ensemble exposing the encoder / sequence / classifier split used by shared-encoding inference."""

    def __init__(self, members, bundle_mean, bundle_std):
        super().__init__()
        self.members = torch.nn.ModuleList([m for m, _, _ in members])
        self.epoch_enc = _EnsembleEncoder(members, bundle_mean, bundle_std)
        self.seq_enc = _EnsembleSequence(members)
        self.classifier = _EnsembleHead(members)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        batch, length, samples = x.shape
        features = self.epoch_enc(x.reshape(batch * length, samples)).view(batch, length, -1)
        return self.classifier(self.seq_enc(features))


def checkpoint_sha256(run_dir):
    """SHA-256 of the checkpoint, or of the ordered member hashes for an ensemble bundle."""
    run_dir = Path(run_dir)
    if (run_dir / 'ensemble.json').exists():
        members = ensemble_members(run_dir)[0]
        hashes = [hashlib.sha256((run_dir.parent / name / 'best_model.pth').read_bytes()).hexdigest() for name in members]
        return hashlib.sha256('\n'.join(hashes).encode('ascii')).hexdigest()
    return hashlib.sha256((run_dir / 'best_model.pth').read_bytes()).hexdigest()


def ensemble_members(run_dir):
    """Validate a nonrecursive bundle of distinct sibling member directories."""
    run_dir = Path(run_dir)
    description = json.loads((run_dir / 'ensemble.json').read_text(encoding='utf-8'))
    members = description.get('members')
    if not isinstance(members, list) or not members or len(set(members)) != len(members):
        raise ValueError('An ensemble requires nonempty distinct member names')
    for name in members:
        if not isinstance(name, str) or Path(name).name != name or name in ('.', '..', run_dir.name):
            raise ValueError('Ensemble members must be distinct sibling directory names')
        if (run_dir.parent / name / 'ensemble.json').exists():
            raise ValueError('Nested ensemble bundles are unsupported')
    if description.get('aggregation', 'mean_logits') != 'mean_logits':
        raise ValueError('Only mean-logit ensemble aggregation is supported')
    return members, description


def load_model(run_dir, device='cpu'):
    run_dir = Path(run_dir)
    if (run_dir / 'ensemble.json').exists():
        return _load_ensemble(run_dir, device)
    config = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    if config.get('SAMPLING_RATE', 100) != 100 or config.get('EPOCH_DURATION', 30) != 30 or config.get('NUM_CLASSES', 5) != 5:
        raise ValueError('This pipeline requires a 100 Hz, 30 second, five-stage checkpoint')
    model = SeqSleepStagingModel(
        num_classes=config.get('NUM_CLASSES', 5), feat_dim=config.get('FEATURE_DIM', 256),
        lstm_hidden=config.get('LSTM_HIDDEN', 128), n_trans_layers=config.get('TRANSFORMER_LAYERS', 2),
        n_heads=config.get('TRANSFORMER_HEADS', 8), cnn_dropout=config.get('CNN_DROPOUT', .5),
        attn_dropout=config.get('ATTN_DROPOUT', .1), fc_dropout=config.get('FC_DROPOUT', .3),
    ).to(device)
    model.load_state_dict(torch.load(run_dir / 'best_model.pth', map_location=device, weights_only=True))
    model.eval()
    with np.load(run_dir / 'norm_stats.npz', allow_pickle=False) as stats:
        mean, std = float(stats['mean']), float(stats['std'])
        config['NORMALIZATION_EPSILON'] = 0.0 if 'input_mode' in stats else 1e-8
    config['IMPORTED_LEGACY_CHECKPOINT'] = not (run_dir / 'data_manifest.json').exists()
    if not np.isfinite([mean, std]).all() or std <= 0:
        raise ValueError('Invalid checkpoint normalization statistics')
    return model, mean, std, config


def _load_ensemble(run_dir, device):
    names, description = ensemble_members(run_dir)
    declared_hashes = description.get('member_checkpoint_sha256')
    if declared_hashes is not None and declared_hashes != [checkpoint_sha256(run_dir.parent / name) for name in names]:
        raise ValueError('An ensemble member checkpoint differs from its declared hash')
    bundle = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    members = []
    for name in names:
        model, mean, std, config = load_model(run_dir.parent / name, device)
        if config['NORMALIZATION_EPSILON'] != 0.0:
            raise ValueError('Ensemble members must use the new exact normalization contract')
        for key in ('FEATURE_DIM', 'LSTM_HIDDEN', 'TRANSFORMER_LAYERS', 'TRANSFORMER_HEADS',
                    'SEQ_LEN', 'INPUT_MODE', 'processing_parameters'):
            if config.get(key) != bundle.get(key):
                raise ValueError(f'Ensemble bundle and member {name} disagree on {key}')
        members.append((model, mean, std))
    bundle_mean, bundle_std = members[0][1], members[0][2]
    ensemble = EnsembleModel(members, bundle_mean, bundle_std).to(device).eval()
    bundle['NORMALIZATION_EPSILON'] = 0.0
    bundle['IMPORTED_LEGACY_CHECKPOINT'] = False
    bundle['ENSEMBLE_MEMBERS'] = names
    return ensemble, bundle_mean, bundle_std, bundle


@torch.inference_mode()
def predict_epochs(X_frames, model, seq_len, device, tta_n=1, seq_stride=None, batch_size=32, timings=None):
    """Encode each frame once per TTA shift, then batch overlapping sequence windows.

    ``timings`` (optional dict) accumulates ``encode_seconds`` and ``temporal_seconds``.
    """
    X_frames = np.asarray(X_frames, dtype=np.float32)
    if X_frames.ndim != 2 or not np.isfinite(X_frames).all():
        raise ValueError('Expected finite epoch sample matrix')
    n = len(X_frames)
    if seq_len < 1 or seq_len > 512 or batch_size <= 0 or not 1 <= tta_n <= 7:
        raise ValueError('Sequence length must be 1..512, batch size positive and TTA 1..7')
    if not n:
        return np.empty((0, 5), np.float32), np.empty(0, np.int64)
    stride = seq_stride or max(1, seq_len // 2)
    if not 1 <= stride <= seq_len:
        raise ValueError('Stride must be 1..sequence length')
    window_len = min(seq_len, n)
    starts = list(range(0, n - window_len + 1, min(stride, window_len)))
    if starts[-1] != n - window_len:
        starts.append(n - window_len)
    accumulator = np.zeros((n, 5), dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    shifts = [0, -50, 50, -25, 25, -12, 12][:tta_n]
    clock = time.perf_counter
    for shift in shifts:
        features = []
        mark = clock()
        for begin in range(0, n, batch_size):
            samples = torch.as_tensor(X_frames[begin:begin + batch_size], device=device)
            if shift:
                samples = torch.roll(samples, shift, dims=-1)
            features.append(model.epoch_enc(samples).cpu())
        encoded = torch.cat(features)
        if timings is not None:
            timings['encode_seconds'] = timings.get('encode_seconds', 0.0) + clock() - mark
        mark = clock()
        for begin in range(0, len(starts), batch_size):
            positions = starts[begin:begin + batch_size]
            windows = torch.stack([encoded[index:index + window_len] for index in positions]).to(device)
            outputs = model.classifier(model.seq_enc(windows)).float().cpu().numpy()
            for position, output in zip(positions, outputs):
                accumulator[position:position + window_len] += output
                counts[position:position + window_len] += 1
        if timings is not None:
            timings['temporal_seconds'] = timings.get('temporal_seconds', 0.0) + clock() - mark
    if (counts[:n] == 0).any():
        raise RuntimeError('Uncovered epochs in sequence inference')
    logits = (accumulator[:n] / counts[:n, None]).astype(np.float32)
    return logits, logits.argmax(axis=1).astype(np.int64)


def frame_labels(segments, starts, epoch_sec=30):
    """Only full epochs covered by a recognized annotation are supervised."""
    labels = np.full(len(starts), -1, dtype=np.int64)
    merged = []
    for onset, end, duration, description in sorted(segments, key=lambda row: (row[0], row[1])):
        if end <= onset:
            continue
        label = map_description_to_label(description)
        if merged and onset < merged[-1][1] - 1e-6:
            raise ValueError('Overlapping ground truth annotations')
        if merged and abs(merged[-1][1] - onset) < 1e-6 and merged[-1][2] == label:
            merged[-1] = (merged[-1][0], end, label)
        else:
            merged.append((onset, end, label))
    for onset, end, label in merged:
        if label >= 0 and end > onset:
            mask = (starts >= onset - 1e-6) & (starts + epoch_sec <= end + 1e-6)
            conflict = mask & (labels >= 0) & (labels != label)
            if conflict.any():
                raise ValueError('Conflicting ground truth annotations')
            labels[mask] = label
    return labels


def _append_segment(segments, start, end, label, reason='scored'):
    if end <= start:
        return
    if segments and segments[-1]['label'] == label and segments[-1]['reason'] == reason and abs(segments[-1]['end_sec'] - start) < 1e-6:
        segments[-1]['end_sec'] = float(end)
    else:
        segments.append({'onset_sec': float(start), 'end_sec': float(end), 'label': int(label), 'reason': reason})


def infer_record(record, model, mean, std, seq_len=20, device='cpu', tta_n=1,
                 decoder='viterbi', transition=None, strength=1.0, batch_size=32, seq_stride=None,
                 normalization_epsilon=1e-8, temperature=1.0, timings=None):
    fs, epoch_sec = 100, 30
    epoch_len = fs * epoch_sec
    if not np.isfinite([mean, std, normalization_epsilon]).all() or std <= 0 or normalization_epsilon < 0:
        raise ValueError('Normalization requires finite mean, positive std and nonnegative epsilon')
    segments = []
    total_frames = len(record['eeg']) // epoch_len
    global_starts = np.arange(total_frames, dtype=np.float64) * epoch_sec
    decoded_all = np.full(total_frames, -1, dtype=np.int64)
    raw_all = np.full(total_frames, -1, dtype=np.int64)
    logits_all = np.full((total_frames, 5), np.nan, dtype=np.float32)
    scorable_all = np.zeros(total_frames, dtype=bool)
    cursor = 0.0
    for block in record['blocks']:
        start_sample, end_sample = block['start_sample'], block['end_sample']
        start_sec, end_sec = start_sample / fs, end_sample / fs
        _append_segment(segments, cursor, start_sec, -1, 'missing_fragment')
        aligned_start = ((start_sample + epoch_len - 1) // epoch_len) * epoch_len
        aligned_end = end_sample // epoch_len * epoch_len
        if aligned_end <= aligned_start:
            _append_segment(segments, start_sec, end_sec, -1, 'incomplete_epoch')
            cursor = end_sec
            continue
        _append_segment(segments, start_sec, aligned_start / fs, -1, 'incomplete_epoch')
        frames = record['eeg'][aligned_start:aligned_end].reshape(-1, epoch_len)
        starts = np.arange(aligned_start, aligned_end, epoch_len) / fs
        valid = np.isfinite(frames).all(axis=1) & (np.std(frames, axis=1) > 1e-10)
        logits = np.full((len(frames), 5), np.nan, dtype=np.float32)
        raw = np.full(len(frames), -1, dtype=np.int64)
        decoded = raw.copy()
        edges = np.flatnonzero(np.diff(np.r_[False, valid, False]))
        for begin, end in zip(edges[::2], edges[1::2]):
            output, argmax = predict_epochs((frames[begin:end] - mean) / (std + normalization_epsilon), model,
                                           seq_len, device, tta_n, seq_stride, batch_size, timings)
            logits[begin:end], raw[begin:end] = output, argmax
            mark = time.perf_counter()
            decoded[begin:end] = (viterbi_decode(output, transition, strength)
                                  if decoder == 'viterbi' else argmax)
            if timings is not None:
                timings['decode_seconds'] = timings.get('decode_seconds', 0.0) + time.perf_counter() - mark
        for start, label in zip(starts, decoded):
            _append_segment(segments, start, start + epoch_sec, label, 'flat_or_invalid_signal' if label == -1 else 'scored')
        _append_segment(segments, aligned_end / fs, end_sec, -1, 'incomplete_epoch')
        indices = (starts / epoch_sec).astype(np.int64)
        decoded_all[indices], raw_all[indices], logits_all[indices] = decoded, raw, logits
        scorable_all[indices] = valid
        cursor = end_sec
    total = len(record['eeg']) / fs
    _append_segment(segments, cursor, total, -1, 'missing_fragment')
    validate_segments(segments, total)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    confidence = np.full(total_frames, np.nan, dtype=np.float32)
    if scorable_all.any():
        z = logits_all[scorable_all].astype(np.float64) / temperature
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        confidence[scorable_all] = p.max(axis=1) / p.sum(axis=1)
    return segments, {
        'onset_sec': global_starts, 'decoded': decoded_all, 'confidence': confidence,
        'argmax': raw_all, 'logits': logits_all, 'scorable': scorable_all,
        'ground_truth': frame_labels(record['label_segments'], global_starts),
    }


def confidence_summary(confidence, scorable, temperature, threshold, protocol, is_ensemble=False):
    """Per-recording confidence statistics; calibrated only when a frozen protocol supplied the temperature."""
    values = confidence[scorable]
    transferred = (protocol is not None and is_ensemble and
                   protocol.get('calibration_prediction_scope') != 'deployed_ensemble')
    summary = {'mean_top_probability': float(values.mean()) if len(values) else None,
               'temperature': temperature, 'calibrated': protocol is not None and not transferred,
               'temperature_fit_applied': protocol is not None,
               'calibration_prediction_scope': (protocol or {}).get('calibration_prediction_scope', 'unspecified'),
               'calibration_transferred_to_ensemble': transferred,
               'low_confidence_threshold': threshold,
               'low_confidence_epochs': int((values < threshold).sum()) if threshold is not None else None,
               'scorable_epochs': int(len(values))}
    summary['note'] = ('The temperature was transferred from fold-model predictions to the deployed ensemble; '
                       'ensemble calibration and threshold accuracy are not independently validated. '
                       'Low-confidence epochs are flagged, not removed.' if transferred else
                       'Temperature fit on validation participants only; low-confidence epochs are flagged, not removed. '
                       'This is not a validated artifact detector.' if protocol else
                       'Uncalibrated softmax confidence; not a probability of correctness.')
    return summary


def validate_protocol_contract(protocol, snapshot, input_mode, tta, seq_stride):
    """Return False for older unbound protocols; reject changed bound inference settings."""
    contract = protocol.get('inference_contract')
    if contract is None:
        return False
    length = snapshot.get('SEQ_LEN', 20)
    expected = {'input_mode': input_mode, 'sequence_length': length,
                'sequence_stride': seq_stride or max(1, length // 2), 'tta': tta,
                'processing_parameters': snapshot.get('processing_parameters', {})}
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f'Frozen protocol inference contract differs on {key}')
    return True


def write_segments(segments, path):
    descriptions = ['Sleep stage W', 'Sleep stage 1', 'Sleep stage 2', 'Sleep stage 3', 'Sleep stage R']
    with Path(path).open('w', encoding='utf-8') as output:
        output.write('onset_sec\tend_sec\tduration_sec\tdescription\n')
        for segment in segments:
            start, end, label = segment['onset_sec'], segment['end_sec'], segment['label']
            description = descriptions[label] if label >= 0 else 'Unknown'
            output.write(f'{start:.6f}\t{end:.6f}\t{end-start:.6f}\t{description}\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test_dir')
    parser.add_argument('--run_name', default=None)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--input_mode', choices=['raw', 'preprocessed'], default='preprocessed')
    parser.add_argument('--decoder', choices=['argmax', 'viterbi'], default='viterbi')
    parser.add_argument('--transition_path', help='JSON 5x5 training-only transition matrix')
    parser.add_argument('--protocol', help='Frozen validation protocol JSON (decoder, prior, strength, temperature, abstention threshold)')
    parser.add_argument('--part_offsets', help='JSON {record_id: {part_number: absolute_start_sample}} for missing parts')
    parser.add_argument('--transition_strength', type=float, default=1.0)
    parser.add_argument('--tta', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seq_stride', type=int)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--record_id', help='Restrict to one recording')
    parser.add_argument('--source_file', help='Standalone single-channel TXT input for a BCI adapter')
    parser.add_argument('--unit', choices=['V', 'uV'], default='V')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    torch.set_num_threads(args.threads)
    cfg = get_config()
    run_dir = resolve_run_dir(cfg.RUNS_DIR, args.run_name)
    model, mean, std, snapshot = load_model(run_dir, cfg.DEVICE)
    processing = snapshot.get('processing_parameters', {})
    if processing.get('epoch_normalize') is not None or processing.get('clip_value') is not None:
        raise ValueError('Checkpoint uses extra epoch processing unsupported by this inference implementation')
    filter_settings = {key: processing[key] for key in ['fs', 'lowcut', 'highcut', 'order', 'enable_notch', 'notch', 'notch_freq', 'notch_q', 'detrend_type'] if key in processing}
    data_dir = args.test_dir or str(Path(cfg.PROJECT_ROOT) / 'data' / args.input_mode / 'test')
    output_dir = Path(args.output_dir or Path(run_dir) / f'predictions_{args.input_mode}')
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.source_file:
        from readers import TextEEGReader
        from data_preprocessing import apply_filter
        input_record = TextEEGReader(unit=args.unit, preprocessed=args.input_mode == 'preprocessed').read(args.source_file)
        samples = input_record.volts()
        if args.input_mode == 'raw':
            standalone_settings = {key: value for key, value in filter_settings.items() if key != 'notch'}
            if filter_settings.get('notch') is not None:
                standalone_settings.update(enable_notch=True, notch_freq=filter_settings['notch'])
            samples = apply_filter(samples, **standalone_settings)
        record_source = {'eeg': samples, 'label_segments': [], 'part_metadata': [], 'input_mode': args.input_mode,
                         'blocks': [{'start_sample': 0, 'end_sample': len(samples)}]}
        subjects = {args.record_id or 'bci_recording': None}
    else:
        if args.unit != 'V':
            parser.error('Dataset units are V; --unit applies only to --source_file')
        subjects = scan_dataset(data_dir)
        if args.record_id:
            subjects = {sid: parts for sid, parts in subjects.items() if sid == args.record_id}
    if not subjects:
        raise ValueError(f'No EEG records found: {data_dir}')
    transition_path = args.transition_path or (str(Path(run_dir) / 'transition.json') if (Path(run_dir) / 'transition.json').exists() else None)
    transition = json.loads(Path(transition_path).read_text(encoding='utf-8')) if transition_path else default_transition()
    checkpoint_hash = checkpoint_sha256(run_dir)
    protocol, temperature, low_confidence = None, 1.0, None
    protocol_contract_bound = False
    if args.protocol:
        protocol = json.loads(Path(args.protocol).read_text(encoding='utf-8'))
        if protocol['checkpoint_sha256'] != checkpoint_hash:
            parser.error('The frozen protocol belongs to a different checkpoint')
        if protocol.get('normalization_sha256') is not None:
            if hashlib.sha256((Path(run_dir) / 'norm_stats.npz').read_bytes()).hexdigest() != protocol['normalization_sha256']:
                parser.error('The normalization statistics differ from the frozen protocol')
        protocol_contract_bound = validate_protocol_contract(protocol, snapshot, args.input_mode, args.tta, args.seq_stride)
        args.decoder = 'argmax' if protocol['selected_variant'] == 'argmax' else 'viterbi'
        args.transition_strength = protocol['selected_strength'] or 0.0
        if protocol['selected_prior'] == 'heuristic':
            transition, transition_path = default_transition(), None
        if protocol.get('transition_sha256') is not None:
            if transition_path is None or hashlib.sha256(Path(transition_path).read_bytes()).hexdigest() != protocol['transition_sha256']:
                parser.error('The learned transition matrix differs from the frozen protocol')
        temperature = protocol['temperature']
        low_confidence = protocol['abstention']['confidence_threshold']
    part_offsets = json.loads(Path(args.part_offsets).read_text(encoding='utf-8')) if args.part_offsets else {}
    for sid, parts in sorted(subjects.items()):
        start_time = time.perf_counter()
        record = record_source if parts is None else load_record(parts, input_mode=args.input_mode,
                                                                part_start_samples=part_offsets.get(sid), **filter_settings)
        load_seconds = time.perf_counter() - start_time
        stage_seconds = {}
        segments, arrays = infer_record(record, model, mean, std, snapshot.get('SEQ_LEN', 20), cfg.DEVICE,
                                        args.tta, args.decoder, transition, args.transition_strength, args.batch_size, args.seq_stride,
                                        snapshot['NORMALIZATION_EPSILON'], temperature, stage_seconds)
        stage_seconds['load_parse_filter_seconds'] = load_seconds
        write_segments(segments, output_dir / f'{sid}_hypnogram.tsv')
        np.savez_compressed(output_dir / f'{sid}_epochs.npz', **arrays)
        for part in record['part_metadata']:
            partial = []
            for segment in segments:
                left, right = max(segment['onset_sec'], part['start_sec']), min(segment['end_sec'], part['end_sec'])
                if right > left:
                    partial.append({**segment, 'onset_sec': left, 'end_sec': right})
            write_segments(partial, output_dir / f"{sid}_Hypnogram_Data_Part_{part['part']}_of_5.txt")
        elapsed = time.perf_counter() - start_time
        metadata = {'record_id': sid, 'input_mode': args.input_mode, 'data_dir': os.path.abspath(data_dir),
                    'total_seconds': len(record['eeg']) / 100, 'sampling_rate_hz': 100, 'epoch_seconds': 30,
                    'checkpoint_sha256': checkpoint_hash, 'run_dir': os.path.abspath(run_dir),
                    'sequence_length': snapshot.get('SEQ_LEN', 20), 'sequence_stride': args.seq_stride or snapshot.get('SEQ_LEN', 20) // 2,
                    'tta': args.tta, 'decoder': args.decoder, 'transition_strength': args.transition_strength,
                    'transition_prior': np.asarray(transition).tolist(),
                    'transition_source': transition_path or 'Unvalidated soft heuristic',
                    'transition_is_heuristic': transition_path is None,
                    'imported_legacy_checkpoint': snapshot['IMPORTED_LEGACY_CHECKPOINT'],
                    'checkpoint_input_mode': snapshot.get('INPUT_MODE', 'unknown'),
                    'normalization_epsilon': snapshot['NORMALIZATION_EPSILON'],
                    'known_annotation_classes': sorted(set(arrays['ground_truth'][arrays['ground_truth'] >= 0].tolist())),
                    'source_file': os.path.abspath(args.source_file) if args.source_file else None,
                    'channel': 'EEG Fpz-Cz', 'input_unit': args.unit,
                    'preprocessing': 'No filtering' if args.input_mode == 'preprocessed' else 'Whole continuous block SOS Butterworth forward-backward',
                    'processing_parameters': processing or {'fs':100,'epoch_sec':30,'lowcut':0.5,'highcut':30.0,'order':4,'enable_notch':False},
                    'threads': args.threads, 'batch_size': args.batch_size, 'device': str(cfg.DEVICE),
                    'torch_version': torch.__version__, 'elapsed_seconds': elapsed,
                    'stage_seconds_before_report_rendering': stage_seconds,
                    'part_metadata': record['part_metadata'], 'annotation_override': False,
                    'protocol': os.path.abspath(args.protocol) if args.protocol else None,
                    'protocol_inference_contract_bound': protocol_contract_bound,
                    'protocol_sha256': hashlib.sha256(Path(args.protocol).read_bytes()).hexdigest() if args.protocol else None,
                    'confidence': confidence_summary(arrays['confidence'], arrays['scorable'], temperature,
                                                     low_confidence, protocol, bool(snapshot.get('ENSEMBLE_MEMBERS')))}
        write_report(segments, metadata, output_dir)
        print(f'{sid}: {len(arrays["decoded"])} epochs; {elapsed:.2f}s; report saved to {output_dir}', flush=True)


if __name__ == '__main__':
    main()
