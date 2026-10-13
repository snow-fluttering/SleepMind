"""Participant-disjoint sequence training with lazy windows and reproducible artifacts.

A pass samples non-overlapping length-L windows with a random phase per
contiguous run, so each original epoch is visited about once per pass without
materializing overlapping sequence arrays. Model selection uses the unique-epoch
five-class macro F1 of whole contiguous validation runs (shared epoch encoding),
not overlapping training windows. The final holdout partition of a split
manifest is never loaded.
"""
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(__file__))
from config import get_config
from model import SeqSleepStagingModel, FocalLoss
from dataset import EpochBank, run_table, window_starts, iter_batches
from data_preprocessing import build_training_set, participant_id
from decoding import estimate_transition
from predict import predict_epochs
from splits import load_manifest
from sysinfo import environment, peak_rss_bytes
from utils import evaluate_arrays

STAGE_NAMES = ['W', 'N1', 'N2', 'N3', 'REM']
SELECTION_METRIC = 'validation unique-epoch five-class macro F1 (argmax, whole contiguous runs)'
SOURCE_FILES = ('train.py', 'model.py', 'dataset.py', 'config.py', 'data_preprocessing.py',
                'predict.py', 'decoding.py', 'splits.py', 'utils.py')


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def participant_validation_split(participants, val_ratio=0.15, seed=42):
    """Exploratory fallback without a manifest: whole participants only, no final holdout."""
    if not 0 < val_ratio < 1:
        raise ValueError('val_ratio must be strictly between zero and one.')
    unique = np.unique(np.asarray(participants))
    if len(unique) < 2:
        raise ValueError('A participant-disjoint split requires at least two participants.')
    np.random.default_rng(seed).shuffle(unique)
    n_val = min(len(unique) - 1, max(1, int(len(unique) * val_ratio)))
    return sorted(unique[n_val:].tolist()), sorted(unique[:n_val].tolist())


def training_epoch_statistics(X, y, sids, train_participants):
    """Mean/std over each original training epoch once, accumulated per run."""
    participants = np.asarray([participant_id(sid) for sid in sids])
    mask = np.isin(participants, list(train_participants))
    if not mask.any():
        raise ValueError('No original epochs belong to the selected training participants.')
    count, mean, squared_deviations = 0, 0.0, 0.0
    for sid in np.unique(sids[mask]):
        values = X[mask & (sids == sid)].astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite EEG samples in training run {sid}.')
        chunk_count, chunk_mean, chunk_variance = values.size, float(values.mean()), float(values.var())
        delta = chunk_mean - mean
        combined = count + chunk_count
        squared_deviations += chunk_variance * chunk_count + delta * delta * count * chunk_count / combined
        mean += delta * chunk_count / combined
        count = combined
    std = float(np.sqrt(max(0.0, squared_deviations / count)))
    if not np.isfinite(std) or std <= np.finfo(np.float32).tiny:
        raise ValueError('Training EEG has zero or invalid variance.')
    return float(mean), std, y[mask], int(mask.sum())


def training_only_transition(y, sids, train_participants):
    """Smoothed transition prior from contiguous training runs only."""
    mask = np.asarray([participant_id(sid) in train_participants for sid in sids])
    if not mask.any():
        raise ValueError('Transition estimation requires original training epochs.')
    return estimate_transition(y[mask], sids[mask], smoothing=1.0)


def compute_class_weights(flat_y, num_classes):
    flat_y = np.asarray(flat_y).reshape(-1)
    if not len(flat_y) or np.any(flat_y < 0) or np.any(flat_y >= num_classes):
        raise ValueError('Class weights require nonempty valid class labels.')
    counts = np.bincount(flat_y, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = 1.0 / counts
    return torch.tensor(weights / weights.sum() * num_classes, dtype=torch.float32)


def sequence_loss(criterion, logits, y):
    return criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


@torch.no_grad()
def ema_update(ema, model, decay):
    """Exponential moving average of floating-point parameters and buffers (integer buffers are copied)."""
    for averaged, current in zip(ema.state_dict().values(), model.state_dict().values()):
        if averaged.dtype.is_floating_point:
            averaged.mul_(decay).add_(current.detach(), alpha=1 - decay)
        else:
            averaged.copy_(current)


def refresh_recurrent_weights(model):
    """Restore contiguous recurrent weights after copying or loading on CUDA."""
    for module in model.modules():
        if isinstance(module, nn.RNNBase):
            module.flatten_parameters()


def extend_cosine_schedule(scheduler, optimizer, total_passes):
    """Extending a completed schedule must not train an extra pass at zero learning rate."""
    previous = scheduler.T_max
    if total_passes > previous:
        scheduler.T_max = total_passes
        rates = [float(base * (1 + np.cos(np.pi * scheduler.last_epoch / total_passes)) / 2)
                 for base in scheduler.base_lrs]
        for group, rate in zip(optimizer.param_groups, rates):
            group['lr'] = rate
        scheduler._last_lr = rates
    return {'original_schedule_passes': previous, 'requested_schedule_passes': total_passes,
            'completed_passes_at_resume': scheduler.last_epoch, 'schedule_extended': total_passes > previous,
            'note': 'An extended horizon changes the future schedule; this is not an exact replay of an uninterrupted run.'}


def validate_resume_contract(cfg, args, run_dir, split, mean, std, cache_path=None):
    """A continuation cannot silently change its data, normalization or model contract."""
    run_dir = Path(run_dir)
    snapshot = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    saved_split = json.loads((run_dir / 'split.json').read_text(encoding='utf-8'))
    # Pass count and early-stop patience may be extended; all learned inputs remain fixed.
    keys = [key for key in CONFIG_KEYS if key not in
            {'TRAIN_DIR', 'NUM_EPOCHS', 'EARLY_STOPPING_PATIENCE'}]
    mismatches = [key for key in keys if snapshot.get(key) != getattr(cfg, key)]
    if snapshot.get('processing_parameters') != preprocessing_parameters(cfg):
        mismatches.append('processing_parameters')
    if snapshot.get('ema_decay', 0.0) != args.ema_decay:
        mismatches.append('ema_decay')
    for key in ('manifest_sha256', 'fit_partitions', 'fit_participants', 'train_participants',
                'validation_participants', 'fit_record_runs', 'validation_record_runs'):
        if saved_split.get(key) != split.get(key):
            mismatches.append(f'split.{key}')
    with np.load(run_dir / 'norm_stats.npz', allow_pickle=False) as normalization:
        if float(normalization['mean']) != mean or float(normalization['std']) != std:
            mismatches.append('normalization_statistics')
    saved_data = json.loads((run_dir / 'data_manifest.json').read_text(encoding='utf-8'))
    if saved_data.get('files'):
        if cache_path is not None:
            with np.load(cache_path, allow_pickle=False) as cached:
                current_files = json.loads(str(cached['metadata'].item()))['files']
        else:
            from data_preprocessing import scan_dataset, _file_sha256
            current_files = []
            wanted = {sid.split('::')[0] for sid in split['fit_record_runs'] + split['validation_record_runs']}
            for sid, parts in scan_dataset(cfg.TRAIN_DIR).items():
                if sid in wanted:
                    for part, pair in parts.items():
                        for path in pair:
                            if path is not None:
                                current_files.append({'path': path, 'record': sid, 'part': part,
                                                      'sha256': _file_sha256(path), 'size': Path(path).stat().st_size})
        signature = lambda rows: sorted((row['record'], row['part'], Path(row['path']).name,
                                        row['size'], row['sha256']) for row in rows)
        if signature(saved_data['files']) != signature(current_files):
            mismatches.append('dataset_content')
    if mismatches:
        raise ValueError('Resume contract differs from the saved run: ' + ', '.join(mismatches))


def train_pass(model, bank, starts, seq_len, batch_size, optimizer, criterion, generator, rng, augment,
               ema=None, ema_decay=0.0, device='cpu'):
    """One shuffled pass over sampled windows; returns mean loss and window count."""
    model.train()
    order = rng.permutation(len(starts))
    total, count = 0.0, 0
    for x, y in iter_batches(bank, starts, seq_len, batch_size, order, augment, generator):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = sequence_loss(criterion, model(x), y)
        if not torch.isfinite(loss):
            raise RuntimeError('Encountered nonfinite loss; inspect input data and normalization.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if ema is not None:
            ema_update(ema, model, ema_decay)
        total += float(loss.item()) * len(x)
        count += len(x)
    if not count:
        raise ValueError('No training windows: every contiguous run is shorter than seq_len.')
    return total / count, count


def validate_runs(model, bank, run_starts, run_lengths, seq_len, device, batch_size):
    """Unique-epoch metrics over whole contiguous runs with shared epoch encoding."""
    model.eval()
    truth, prediction, loss_terms = [], [], []
    for begin, length in zip(run_starts, run_lengths):
        frames = (bank.X[begin:begin + length] - np.float32(bank.mean)) / np.float32(bank.std)
        logits, argmax = predict_epochs(frames, model, seq_len, device, 1, max(1, seq_len // 2), batch_size)
        truth.append(bank.y[begin:begin + length])
        prediction.append(argmax)
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_p = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        loss_terms.append(-log_p[np.arange(length), bank.y[begin:begin + length]])
    truth, prediction = np.concatenate(truth), np.concatenate(prediction)
    result = evaluate_arrays(truth, prediction)
    result['negative_log_likelihood'] = float(np.concatenate(loss_terms).mean())
    return result


def setup_run_dir(base_dir, run_name=None):
    """Create a fresh run directory, refusing accidental checkpoint overwrites."""
    if run_name is None:
        run_name = 'run_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if Path(run_name).name != run_name or run_name in ('.', '..'):
        raise ValueError('run_name must be a directory name without path separators.')
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=False)
    return run_dir, run_name


def preprocessing_parameters(cfg):
    """Explicit input processing contract stored with every new checkpoint."""
    return {
        'fs': cfg.SAMPLING_RATE, 'epoch_sec': cfg.EPOCH_DURATION,
        'lowcut': cfg.LOW_FREQ, 'highcut': cfg.HIGH_FREQ, 'order': cfg.FILTER_ORDER,
        'enable_notch': False, 'notch': cfg.NOTCH_FREQ, 'notch_freq': 50.0,
        'notch_q': 30.0, 'detrend_type': None, 'epoch_normalize': None,
        'clip_value': None, 'input_mode': cfg.INPUT_MODE, 'label_timebase': 'global',
    }


CONFIG_KEYS = [
    'TRAIN_DIR', 'INPUT_MODE', 'SAMPLING_RATE', 'EPOCH_DURATION', 'LOW_FREQ', 'HIGH_FREQ',
    'FILTER_ORDER', 'NOTCH_FREQ', 'TTA_N', 'SEQ_LEN', 'BATCH_SIZE', 'LEARNING_RATE',
    'WEIGHT_DECAY', 'NUM_EPOCHS', 'EARLY_STOPPING_PATIENCE', 'VAL_RATIO', 'FEATURE_DIM',
    'LSTM_HIDDEN', 'TRANSFORMER_LAYERS', 'TRANSFORMER_HEADS', 'CNN_DROPOUT', 'ATTN_DROPOUT',
    'FC_DROPOUT', 'USE_FOCAL_LOSS', 'FOCAL_GAMMA', 'LABEL_SMOOTHING', 'USE_CLASS_WEIGHTS',
    'USE_DATA_AUGMENT', 'SEED', 'NUM_CLASSES',
]


def save_config_snapshot(cfg, run_dir, extra):
    snapshot = {key: getattr(cfg, key) for key in CONFIG_KEYS}
    snapshot.update({
        'validation_metric_scope': SELECTION_METRIC,
        'normalization_scope': 'Unique original epochs from training participants only.',
        'processing_parameters': preprocessing_parameters(cfg),
        'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in SOURCE_FILES},
        'environment': environment(),
    })
    snapshot.update(extra)
    Path(run_dir, 'config.json').write_text(json.dumps(snapshot, indent=2, default=str), encoding='utf-8')


def save_data_manifest(cache_path, run_dir, processing):
    if cache_path and Path(cache_path).is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            manifest = json.loads(str(cached['metadata'].item()))
            manifest['cache_key'] = str(cached['cache_key'].item())
    else:
        manifest = {'processing': processing, 'files': []}
    manifest['fingerprint_method'] = 'SHA-256 of every source text file (see files) and processing parameters.'
    serialized = json.dumps(manifest, sort_keys=True, indent=2)
    Path(run_dir, 'data_manifest.json').write_text(serialized, encoding='utf-8')
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def plot_training_curve(log, run_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        passes = np.arange(1, len(log['train_loss']) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(passes, log['train_loss'], label='train windows')
        if log['val_nll']:
            axes[0].plot(passes, log['val_nll'], label='validation NLL (unique epochs)')
        axes[0].set(xlabel='Pass', ylabel='Loss', title='Loss')
        if log['val_macro_f1']:
            axes[1].plot(passes, log['val_macro_f1'], label='validation five-class macro F1')
            axes[1].plot(passes, log['val_accuracy'], label='validation accuracy')
        axes[1].set(xlabel='Pass', ylabel='Score', title='Validation (unique epochs, argmax)')
        for ax in axes:
            ax.legend()
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, 'training_curve.png'), dpi=120)
        plt.close(fig)
    except Exception as exc:  # plotting is optional diagnostics
        print(f'[warn] Could not draw training curves: {exc}')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train_dir', type=str, default=None)
    parser.add_argument('--input_mode', choices=['raw', 'preprocessed'], default=None)
    parser.add_argument('--split_manifest', default=None,
                        help='Frozen participant manifest; omit only for exploratory runs without a holdout.')
    parser.add_argument('--fit_partitions', choices=['train', 'train+validation'], default='train',
                        help='train+validation refits on both with a fixed pass count and no model selection.')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None, help='Number of passes.')
    parser.add_argument('--batch_size', type=int, default=None, help='Windows per training step.')
    parser.add_argument('--infer_batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--seq_len', type=int, default=None)
    parser.add_argument('--label_smoothing', type=float, default=None)
    parser.add_argument('--cnn_dropout', type=float, default=None)
    parser.add_argument('--fc_dropout', type=float, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--ema_decay', type=float, default=0.0,
                        help='Per-step weight EMA decay in [0, 1); the averaged weights are validated and saved.')
    parser.add_argument('--val_ratio', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--threads', type=int, default=None)
    parser.add_argument('--record_cache', default=None, help='Directory for joined per-record signal caches.')
    parser.add_argument('--resume', action='store_true', help='Continue an existing run_name from last_state.pt.')
    parser.add_argument('--set_latest', action='store_true', help='Point runs/latest.txt at this run when done.')
    parser.add_argument('--no_cache', action='store_true')
    parser.add_argument('--no_focal', action='store_true')
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--no_class_weights', action='store_true')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = get_config()
    if args.input_mode:
        cfg.INPUT_MODE = args.input_mode
    if cfg.INPUT_MODE == 'raw':
        cfg.TRAIN_DIR = cfg.RAW_TRAIN_DIR
    if args.train_dir:
        cfg.TRAIN_DIR = os.path.abspath(args.train_dir)
    for argument, key in [
        ('epochs', 'NUM_EPOCHS'), ('batch_size', 'BATCH_SIZE'), ('lr', 'LEARNING_RATE'),
        ('seq_len', 'SEQ_LEN'), ('val_ratio', 'VAL_RATIO'), ('seed', 'SEED'),
        ('weight_decay', 'WEIGHT_DECAY'), ('label_smoothing', 'LABEL_SMOOTHING'),
        ('cnn_dropout', 'CNN_DROPOUT'), ('fc_dropout', 'FC_DROPOUT'),
        ('patience', 'EARLY_STOPPING_PATIENCE'),
    ]:
        if getattr(args, argument) is not None:
            setattr(cfg, key, getattr(args, argument))
    if args.device:
        cfg.DEVICE = torch.device(args.device)
    cfg.USE_FOCAL_LOSS = cfg.USE_FOCAL_LOSS and not args.no_focal
    cfg.USE_DATA_AUGMENT = cfg.USE_DATA_AUGMENT and not args.no_augment
    cfg.USE_CLASS_WEIGHTS = cfg.USE_CLASS_WEIGHTS and not args.no_class_weights
    if not 0 <= args.ema_decay < 1:
        parser.error('ema_decay must lie in [0, 1).')
    for key in ['NUM_EPOCHS', 'BATCH_SIZE', 'SEQ_LEN', 'LEARNING_RATE']:
        if getattr(cfg, key) <= 0:
            parser.error(f'{key} must be positive.')
    if cfg.SEQ_LEN > 512 or not 0 < cfg.VAL_RATIO < 1 or args.infer_batch_size <= 0:
        parser.error('Require seq_len <= 512, 0 < val_ratio < 1 and a positive inference batch size.')
    if args.threads is not None:
        if args.threads <= 0:
            parser.error('threads must be positive.')
        torch.set_num_threads(args.threads)
    if cfg.DEVICE.type == 'cuda' and not torch.cuda.is_available():
        parser.error('A CUDA device was requested but CUDA is unavailable.')
    if args.fit_partitions == 'train+validation' and not args.split_manifest:
        parser.error('--fit_partitions train+validation requires --split_manifest.')
    began = time.perf_counter()
    set_seed(cfg.SEED)
    os.makedirs(cfg.CACHE_DIR, exist_ok=True)
    print(f'[env] Device: {cfg.DEVICE}; input mode: {cfg.INPUT_MODE}; threads: {torch.get_num_threads()}', flush=True)

    manifest = load_manifest(args.split_manifest) if args.split_manifest else None
    from data_preprocessing import scan_dataset
    available = sorted(scan_dataset(cfg.TRAIN_DIR))
    if manifest:
        wanted = manifest['partitions']['train']['records'] + manifest['partitions']['validation']['records']
        records = [r for r in wanted if r in set(available)]
        if len(records) != len(wanted):
            parser.error('Some manifest train/validation records are absent from the training directory.')
    else:
        records = available
    cache = None if args.no_cache else os.path.join(
        cfg.CACHE_DIR, f'train_{cfg.INPUT_MODE}_{hashlib.sha256(",".join(records).encode()).hexdigest()[:10]}.npz')
    processing = preprocessing_parameters(cfg)
    data_began = time.perf_counter()
    X, y, sids = build_training_set(cfg.TRAIN_DIR, **processing, cache_path=cache, records=records,
                                    record_cache_dir=args.record_cache)
    data_seconds = time.perf_counter() - data_began
    if not len(y):
        raise ValueError('No valid labeled epochs were loaded from the training directory.')
    participants = np.asarray([participant_id(s) for s in sids])
    if manifest:
        validation_participants = list(manifest['partitions']['validation']['participants'])
        train_participants = list(manifest['partitions']['train']['participants'])
    else:
        train_participants, validation_participants = participant_validation_split(
            participants, cfg.VAL_RATIO, cfg.SEED)
    fit_participants = (train_participants + validation_participants
                        if args.fit_partitions == 'train+validation' else train_participants)
    if set(train_participants) & set(validation_participants):
        raise ValueError('Training and validation participants overlap.')
    fit_mask = np.isin(participants, fit_participants)
    val_mask = np.isin(participants, validation_participants)
    mean, std, fit_y, n_fit_epochs = training_epoch_statistics(X, y, sids, fit_participants)
    transition = training_only_transition(y, sids, set(fit_participants))
    class_weights = (compute_class_weights(fit_y, cfg.NUM_CLASSES).to(cfg.DEVICE)
                     if cfg.USE_CLASS_WEIGHTS else None)
    bank = EpochBank(X, y, mean, std)
    run_starts, run_lengths = run_table(sids)
    run_participant = participants[run_starts]
    fit_runs = np.isin(run_participant, fit_participants)
    val_runs = np.isin(run_participant, validation_participants)
    n_short = int((run_lengths[fit_runs] < cfg.SEQ_LEN).sum())
    split = {
        'seed': cfg.SEED, 'manifest': args.split_manifest, 'fit_partitions': args.fit_partitions,
        'manifest_sha256': manifest['manifest_sha256'] if manifest else None,
        'train_participants': sorted(train_participants), 'validation_participants': sorted(validation_participants),
        'fit_participants': sorted(fit_participants),
        'fit_record_runs': sorted(set(str(s) for s in sids[fit_mask])),
        'validation_record_runs': sorted(set(str(s) for s in sids[val_mask])),
        'holdout_loaded': False,
        'fit_runs_shorter_than_seq_len': n_short,
        'fit_epochs_in_runs_shorter_than_seq_len': int(run_lengths[fit_runs][run_lengths[fit_runs] < cfg.SEQ_LEN].sum()),
    }
    print(f'[data] Epochs: fit {n_fit_epochs}, validation {int(val_mask.sum())}; '
          f'fit class counts {np.bincount(fit_y, minlength=5).tolist()}; load {data_seconds:.1f}s', flush=True)
    print(f'[data] Validation participants: {sorted(validation_participants)}', flush=True)
    print(f'[data] Training-only normalization: mean={mean:.4e}, std={std:.4e}', flush=True)

    model = SeqSleepStagingModel(
        num_classes=cfg.NUM_CLASSES, feat_dim=cfg.FEATURE_DIM, lstm_hidden=cfg.LSTM_HIDDEN,
        n_trans_layers=cfg.TRANSFORMER_LAYERS, n_heads=cfg.TRANSFORMER_HEADS,
        cnn_dropout=cfg.CNN_DROPOUT, attn_dropout=cfg.ATTN_DROPOUT, fc_dropout=cfg.FC_DROPOUT,
    ).to(cfg.DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    criterion = (FocalLoss(alpha=class_weights, gamma=cfg.FOCAL_GAMMA, label_smoothing=cfg.LABEL_SMOOTHING)
                 if cfg.USE_FOCAL_LOSS else nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.LABEL_SMOOTHING))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS)
    ema = copy.deepcopy(model).eval() if args.ema_decay > 0 else None
    if ema is not None:
        refresh_recurrent_weights(ema)
        for parameter in ema.parameters():
            parameter.requires_grad_(False)

    continuation = None
    if args.resume:
        if not args.run_name:
            parser.error('--resume requires --run_name.')
        if Path(args.run_name).name != args.run_name or args.run_name in ('.', '..'):
            parser.error('run_name must be a directory name without path separators.')
        run_dir, run_name = os.path.join(cfg.RUNS_DIR, args.run_name), args.run_name
        validate_resume_contract(cfg, args, run_dir, split, mean, std, cache)
        state = torch.load(os.path.join(run_dir, 'last_state.pt'), map_location=cfg.DEVICE, weights_only=True)
        model.load_state_dict(state['model'])
        refresh_recurrent_weights(model)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        continuation = extend_cosine_schedule(scheduler, optimizer, cfg.NUM_EPOCHS)
        if ema is not None:
            ema.load_state_dict(state['ema'])
            refresh_recurrent_weights(ema)
        log, best_score, best_epoch, patience = state['log'], state['best_score'], state['best_epoch'], state['patience']
        first_pass = len(log['train_loss']) + 1
        print(f'[resume] Continuing {run_name} at pass {first_pass} (RNG streams are reseeded).', flush=True)
        history_path = Path(run_dir, 'continuation_history.json')
        history = json.loads(history_path.read_text(encoding='utf-8')) if history_path.exists() else []
        history.append({**continuation, 'seed': cfg.SEED, 'rng_streams_reseeded': True,
                        'requested_total_passes': cfg.NUM_EPOCHS,
                        'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                          for name in SOURCE_FILES}})
        history_path.write_text(json.dumps(history, indent=2), encoding='utf-8')
    else:
        run_dir, run_name = setup_run_dir(cfg.RUNS_DIR, args.run_name)
        log = {'train_loss': [], 'val_nll': [], 'val_macro_f1': [], 'val_accuracy': [], 'val_kappa': [],
               'lr': [], 'pass_seconds': [], 'validation_seconds': [], 'train_windows': [], 'peak_rss_bytes': []}
        best_score, best_epoch, patience, first_pass = -1.0, 0, 0, 1
        np.savez(os.path.join(run_dir, 'norm_stats.npz'), mean=mean, std=std, input_mode=cfg.INPUT_MODE,
                 n_original_training_epochs=n_fit_epochs)
        Path(run_dir, 'transition.json').write_text(json.dumps(transition.tolist(), indent=2), encoding='utf-8')
        Path(run_dir, 'class_weights.json').write_text(json.dumps(
            None if class_weights is None else class_weights.cpu().tolist()), encoding='utf-8')
        Path(run_dir, 'split.json').write_text(json.dumps(split, indent=2), encoding='utf-8')
        data_hash = save_data_manifest(cache, run_dir, processing)
        save_config_snapshot(cfg, run_dir, {
            'run_name': run_name, 'val_subjects': sorted(validation_participants),
            'train_subjects': sorted(train_participants), 'fit_subjects': sorted(fit_participants),
            'model_params_M': round(n_params, 3), 'n_original_training_epochs': n_fit_epochs,
            'cache_path': cache, 'threads': torch.get_num_threads(), 'data_manifest_sha256': data_hash,
            'split_manifest': args.split_manifest, 'split_manifest_sha256': split['manifest_sha256'],
            'fit_partitions': args.fit_partitions, 'infer_batch_size': args.infer_batch_size,
            'ema_decay': args.ema_decay,
            'selection_metric': SELECTION_METRIC if args.fit_partitions == 'train' else 'none (fixed pass count)',
            'initialization': 'fresh random initialization (no imported checkpoint)',
            'transition_prior_source': 'Contiguous runs of fit participants only.', 'transition_smoothing': 1.0,
            'filter_applied': cfg.INPUT_MODE == 'raw', 'record_cache': args.record_cache,
            'filter_implementation': 'Continuous-block zero-phase Butterworth second-order sections.',
            'data_loading_seconds': data_seconds,
            'window_sampling': 'Random phase per contiguous run, non-overlapping windows, one pass visits each epoch about once.',
        })
    print(f'[run] {run_dir}; parameters {n_params:.2f}M', flush=True)

    generator = torch.Generator().manual_seed(cfg.SEED + first_pass)
    rng = np.random.default_rng(cfg.SEED + first_pass)
    val_starts, val_lengths = run_starts[val_runs], run_lengths[val_runs]
    use_validation = args.fit_partitions == 'train' and len(val_starts) > 0
    for current in range(first_pass, cfg.NUM_EPOCHS + 1):
        pass_began = time.perf_counter()
        starts = window_starts(run_starts[fit_runs], run_lengths[fit_runs], cfg.SEQ_LEN, rng=rng)
        learning_rate = optimizer.param_groups[0]['lr']
        loss, n_windows = train_pass(model, bank, starts, cfg.SEQ_LEN, cfg.BATCH_SIZE, optimizer,
                                     criterion, generator, rng, cfg.USE_DATA_AUGMENT, ema, args.ema_decay, cfg.DEVICE)
        evaluated = ema if ema is not None else model
        scheduler.step()
        train_seconds = time.perf_counter() - pass_began
        val_began = time.perf_counter()
        if use_validation:
            val = validate_runs(evaluated, bank, val_starts, val_lengths, cfg.SEQ_LEN, cfg.DEVICE, args.infer_batch_size)
            score = val['macro_f1_5class']
        else:
            val, score = None, float(current)  # fixed-pass refit keeps the latest weights
        val_seconds = time.perf_counter() - val_began
        for key, value in [('train_loss', loss), ('lr', learning_rate), ('pass_seconds', train_seconds),
                           ('validation_seconds', val_seconds), ('train_windows', n_windows),
                           ('peak_rss_bytes', peak_rss_bytes())]:
            log[key].append(value)
        if val:
            for key, value in [('val_nll', val['negative_log_likelihood']), ('val_macro_f1', val['macro_f1_5class']),
                               ('val_accuracy', val['accuracy']), ('val_kappa', val['cohen_kappa'])]:
                log[key].append(value)
        improved = score > best_score
        if improved:
            best_score, best_epoch, patience = score, current, 0
            torch.save(evaluated.state_dict(), os.path.join(run_dir, 'best_model.pth'))
            if val:
                Path(run_dir, 'val_report.json').write_text(json.dumps(
                    {'run': run_name, 'best_pass': current, 'selection_metric': SELECTION_METRIC, **val},
                    indent=2, allow_nan=False), encoding='utf-8')
        else:
            patience += 1
        message = (f'[{current:03d}/{cfg.NUM_EPOCHS}] lr={learning_rate:.2e} loss={loss:.4f} windows={n_windows}')
        if val:
            message += (f' | val acc={val["accuracy"]:.4f} mF1={val["macro_f1_5class"]:.4f} '
                        f'kappa={val["cohen_kappa"]:.4f} nll={val["negative_log_likelihood"]:.4f}')
        print(message + f'{" *" if improved and val else ""} | train {train_seconds:.0f}s val {val_seconds:.0f}s', flush=True)
        Path(run_dir, 'train_log.json').write_text(json.dumps(
            {'best_pass': best_epoch, 'best_score': best_score, 'selection_metric': SELECTION_METRIC, **log},
            indent=2), encoding='utf-8')
        torch.save({'model': model.state_dict(), 'ema': None if ema is None else ema.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'log': log, 'best_score': best_score,
                    'best_epoch': best_epoch, 'patience': patience}, os.path.join(run_dir, 'last_state.pt'))
        plot_training_curve(log, run_dir)
        if use_validation and patience >= cfg.EARLY_STOPPING_PATIENCE:
            print(f'[early stop] No validation macro-F1 improvement for {patience} passes.', flush=True)
            break
    if not use_validation:
        torch.save((ema if ema is not None else model).state_dict(), os.path.join(run_dir, 'best_model.pth'))
    Path(run_dir, 'run_info.json').write_text(json.dumps({
        'device': str(cfg.DEVICE),
        'peak_cuda_memory_bytes': (int(torch.cuda.max_memory_allocated()) if cfg.DEVICE.type == 'cuda' else None),
        'wall_seconds_this_invocation': time.perf_counter() - began, 'passes_completed': len(log['train_loss']),
        'best_pass': best_epoch, 'peak_rss_bytes': peak_rss_bytes(), 'environment': environment(),
        'checkpoint_sha256': hashlib.sha256(Path(run_dir, 'best_model.pth').read_bytes()).hexdigest(),
        'continuation': continuation,
        'mean_pass_seconds': float(np.mean(log['pass_seconds'])),
        'mean_validation_seconds': float(np.mean(log['validation_seconds'])),
    }, indent=2), encoding='utf-8')
    if args.set_latest:
        Path(cfg.RUNS_DIR, 'latest.txt').write_text(run_name, encoding='utf-8')
    print(f'[done] best pass {best_epoch}; checkpoint {os.path.join(run_dir, "best_model.pth")}', flush=True)


if __name__ == '__main__':
    main()
