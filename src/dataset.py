"""Lazy contiguous-window sampling over a single epoch bank.

Every original epoch is stored once as a ``(N, samples)`` float32 matrix.
Windows are described by start indices into that matrix and are assembled per
batch, so no duplicate sequence arrays are materialized. A window never crosses
the boundary between contiguous runs (record, signal block or label gap).
"""
import numpy as np
import torch


def run_table(run_ids):
    """Return ``(starts, lengths)`` of maximal blocks with equal consecutive run ids."""
    run_ids = np.asarray(run_ids)
    if run_ids.ndim != 1:
        raise ValueError('run_ids must be one-dimensional')
    if not len(run_ids):
        return np.empty(0, np.int64), np.empty(0, np.int64)
    edges = np.r_[0, np.flatnonzero(run_ids[1:] != run_ids[:-1]) + 1, len(run_ids)]
    return edges[:-1].astype(np.int64), np.diff(edges).astype(np.int64)


def window_starts(run_starts, run_lengths, seq_len, stride=None, rng=None):
    """Start indices of length-``seq_len`` windows inside runs.

    With ``rng`` each run receives a random phase and non-overlapping windows
    (one pass visits nearly every epoch once). Without it the grid is
    deterministic with the given stride (default: non-overlapping). Runs shorter
    than ``seq_len`` yield no window.
    """
    if seq_len <= 0 or (stride is not None and stride <= 0):
        raise ValueError('seq_len and stride must be positive')
    step = stride or seq_len
    starts = []
    for begin, length in zip(run_starts, run_lengths):
        if length < seq_len:
            continue
        phase = int(rng.integers(0, min(step, length - seq_len + 1))) if rng is not None else 0
        starts.append(np.arange(begin + phase, begin + length - seq_len + 1, step))
    return np.concatenate(starts).astype(np.int64) if starts else np.empty(0, np.int64)


class EpochBank:
    """Normalized batch assembly from one shared epoch matrix."""

    def __init__(self, X, y, mean, std):
        if X.ndim != 2 or y.shape != (len(X),):
            raise ValueError('Expected X (N, samples) and y (N,)')
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise ValueError('Normalization statistics must be finite; std must be positive')
        self.X, self.y = X, y.astype(np.int64, copy=False)
        self.mean, self.std = float(mean), float(std)

    def __len__(self):
        return len(self.y)

    def batch(self, starts, seq_len):
        index = np.asarray(starts, np.int64)[:, None] + np.arange(seq_len)[None, :]
        x = (self.X[index] - np.float32(self.mean)) / np.float32(self.std)
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), torch.from_numpy(self.y[index])


def augment_batch(x, generator):
    """Seeded amplitude noise, gain and circular shift, one draw per sequence."""
    batch, _, samples = x.shape
    draw = lambda *shape: torch.rand(*shape, generator=generator)
    noise = (draw(batch, 1, 1) < 0.5).float() * draw(batch, 1, 1) * 0.02
    x = x + torch.randn(x.shape, generator=generator) * noise
    gain = torch.where(draw(batch, 1, 1) < 0.3, 0.9 + 0.2 * draw(batch, 1, 1), torch.ones(batch, 1, 1))
    x = x * gain
    shift = torch.where(draw(batch) < 0.3, torch.randint(-100, 101, (batch,), generator=generator),
                        torch.zeros(batch, dtype=torch.int64))
    positions = (torch.arange(samples)[None, :] - shift[:, None]) % samples
    return x.gather(-1, positions[:, None, :].expand_as(x))


def iter_batches(bank, starts, seq_len, batch_size, order=None, augment=False, generator=None):
    """Yield ``(x, y)`` batches; ``order`` (a permutation of starts) must be supplied to shuffle."""
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    starts = np.asarray(starts)[order] if order is not None else np.asarray(starts)
    for begin in range(0, len(starts), batch_size):
        x, y = bank.batch(starts[begin:begin + batch_size], seq_len)
        if augment:
            x = augment_batch(x, generator)
        yield x, y
