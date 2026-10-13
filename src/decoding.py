"""Offline Viterbi decoding with inspectable, configurable transition priors."""
import numpy as np

STAGES = ['W', 'N1', 'N2', 'N3', 'REM']


def default_transition():
    """Soft heuristic prior; rare transitions remain possible. Not clinical rules."""
    return np.array([
        [.92, .045, .015, .005, .015],
        [.10, .65, .21, .01, .03],
        [.025, .04, .85, .055, .03],
        [.015, .01, .08, .89, .005],
        [.05, .035, .04, .005, .87],
    ], dtype=np.float64)


def estimate_transition(labels, sequence_ids, smoothing=1.0):
    """Count only adjacent valid labels within the same contiguous training run."""
    labels = np.asarray(labels)
    sequence_ids = np.asarray(sequence_ids)
    if labels.ndim != 1 or sequence_ids.shape != labels.shape or smoothing <= 0:
        raise ValueError('Expected aligned 1D labels and sequence IDs, with positive smoothing')
    counts = np.full((5, 5), float(smoothing))
    valid = (sequence_ids[1:] == sequence_ids[:-1]) & (labels[1:] >= 0) & (labels[:-1] >= 0)
    if np.any(labels >= 5):
        raise ValueError('Stage labels must be below 5')
    np.add.at(counts, (labels[:-1][valid], labels[1:][valid]), 1)
    return counts / counts.sum(axis=1, keepdims=True)


def viterbi_decode(logits, transition=None, strength=1.0, initial=None):
    """Return the maximum-score path. A zero transition probability forbids an edge."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 5 or not np.isfinite(logits).all():
        raise ValueError('Expected finite (epochs, 5) logits')
    if not np.isfinite(strength) or strength < 0:
        raise ValueError('Transition strength must be finite and nonnegative')
    if not len(logits):
        return np.empty(0, dtype=np.int64)
    if strength == 0:
        return logits.argmax(axis=1).astype(np.int64)
    trans = default_transition() if transition is None else np.asarray(transition, dtype=np.float64)
    if trans.shape != (5, 5) or not np.isfinite(trans).all() or (trans < 0).any() or (trans.sum(axis=1) <= 0).any():
        raise ValueError('Transition matrix must be a nonnegative finite 5 x 5 matrix with nonzero rows')
    trans = trans / trans.sum(axis=1, keepdims=True)
    log_trans = np.full_like(trans, -np.inf)
    np.log(trans, out=log_trans, where=trans > 0)
    prior = np.ones(5) / 5 if initial is None else np.asarray(initial, dtype=np.float64)
    if prior.shape != (5,) or not np.isfinite(prior).all() or (prior < 0).any() or prior.sum() <= 0:
        raise ValueError('Initial probabilities must be five finite nonnegative values')
    log_prior = np.full(5, -np.inf)
    np.log(prior / prior.sum(), out=log_prior, where=prior > 0)
    score = logits[0] + log_prior
    back = np.zeros((len(logits), 5), dtype=np.int64)
    for index in range(1, len(logits)):
        candidates = score[:, None] + strength * log_trans
        back[index] = candidates.argmax(axis=0)
        score = logits[index] + candidates.max(axis=0)
    if not np.isfinite(score).any():
        raise ValueError('No feasible Viterbi path under the supplied constraints')
    path = np.empty(len(logits), dtype=np.int64)
    path[-1] = score.argmax()
    for index in range(len(logits) - 1, 0, -1):
        path[index - 1] = back[index, path[index]]
    return path
