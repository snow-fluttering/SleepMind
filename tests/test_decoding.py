import itertools
import numpy as np
import pytest
from decoding import viterbi_decode, default_transition, estimate_transition


def test_viterbi_matches_exhaustive_path_score():
    logits = np.random.default_rng(7).normal(size=(4, 5))
    transition = default_transition()
    path = viterbi_decode(logits, transition, strength=.7)
    def score(candidate):
        return sum(logits[i, label] for i, label in enumerate(candidate)) + .7 * sum(
            np.log(transition[a, b]) for a, b in zip(candidate, candidate[1:]))
    best = max(itertools.product(range(5), repeat=4), key=score)
    np.testing.assert_array_equal(path, best)


def test_zero_probability_forbids_transition_and_zero_strength_is_argmax():
    logits = np.array([[9, 0, 0, 0, 0], [0, 10, 0, 0, 0]], dtype=float)
    result = viterbi_decode(logits, np.eye(5))
    assert result[0] == result[1]
    np.testing.assert_array_equal(viterbi_decode(logits, strength=0), [0, 1])
    with pytest.raises(ValueError):
        viterbi_decode(np.full((1, 5), np.nan))


def test_transition_estimation_does_not_cross_gaps():
    matrix = estimate_transition(np.array([0, 1, 4, 3]), np.array(['a', 'a', 'b', 'b']))
    assert matrix[0, 1] > matrix[0, 4]
    assert matrix[1, 4] == matrix[1, 0]
