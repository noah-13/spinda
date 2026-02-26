"""Tests for evaluation metrics."""

import numpy as np
import pytest
import math

from nli_toolkits.eval.metrics import (
    validate_and_fix_probs,
    compute_tvd,
    compute_distance_correlation,
    compute_jsd,
    compute_kl_human_to_pred,
    compute_soft_macro_f1,
    compute_soft_micro_f1,

)

""" 
If you want to test only metrics, you can run this file directly with pytest:
pytest tests/test_metrics.py

If you want to test one specific metric, you can use class name or function name:
pytest tests/test_metrics.py::TestProb
"""

class TestProb:
    """Tests for validating input probabilities for metrics."""

    def test_negative_probs_raise(self):
        p = np.array([[0.5, -0.1, 0.6]])

        with pytest.raises(ValueError, match="negative values"):
            validate_and_fix_probs(p)

    def test_not_normalized_probs_raise(self):
        p = np.array([[0.2, 0.2, 0.2]])  # sums to 0.6

        with pytest.raises(ValueError, match="sum to 1"):
            validate_and_fix_probs(p)

    def test_valid_probs_pass(self):
        p = np.array([[0.5, 0.3, 0.2]])
        fixed = validate_and_fix_probs(p)

        assert np.allclose(fixed, p, atol=1e-6)  # valid probabilities should be unchanged
    
    def test_infite_or_non_probs_raise(self):
        p = np.array([[0.5, 0.3, np.inf]])
        q = np.array([[0.5, 0.3, np.nan]])

        with pytest.raises(ValueError, match="inf"):
            validate_and_fix_probs(p)
        with pytest.raises(ValueError, match="NaN"):
            validate_and_fix_probs(q)
    
            
class TestTVD:
    def test_compute_tvd_values(self):
        """Correctness tests for TVD values."""

        # Batch with different rows
        model_probs = np.array([
            [0.5, 0.5, 0.0],  # vs [0,1,0] -> tvd = ||0.5-0| + |0.5-1| + |0-0| = 0.5 (partially overlapping)
            [1.0, 0.0, 0.0],  # vs [0,0,1] -> tvd = ||1-0| + |0-0| + |0-1| = 1.0 (completely different)
            [0.2, 0.3, 0.5],  # same -> tvd = ||0.2-0.2| + |0.3-0.3| + |0.5-0.5| = 0.0 (identical)
        ])
        human_probs = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.2, 0.3, 0.5],
        ])

        tvd = compute_tvd(model_probs, human_probs)
        assert tvd.shape == (3,)
        assert tvd == pytest.approx([0.5, 1.0, 0.0], abs=1e-6)


    def test_compute_invalid_shape_raises(self):
        p = np.array([[0.5, 0.5]])
        q = np.array([[0.5, 0.5], [0.5, 0.5]])

        with pytest.raises(ValueError, match="same shape"):
            compute_tvd(p, q)

class TestKL:
    def test_compute_invalid_shape_raises(self):
        p = np.array([[0.5, 0.5]])
        q = np.array([[0.5, 0.5], [0.5, 0.5]])

        with pytest.raises(ValueError, match="same shape"):
            compute_kl_human_to_pred(p, q)

    def test_kl_values(self):
        human = np.array([
            [0.5, 0.5],   # [0.5, 0.5] vs [0.5, 0.5] -> KL = 0 (identical)
            [0.1, 0.9],   # [0.1, 0.9] vs [0.8, 0.2] -> KL > 0 (different)
            [0.25, 0.75], # [0.25, 0.75] vs [0.5, 0.5] -> KL > 0 (different)
        ])

        pred = np.array([
            [0.5, 0.5],
            [0.8, 0.2],
            [0.5, 0.5],
        ])
        
        epsilon = 1e-12
        kl = compute_kl_human_to_pred(pred, human, epsilon=epsilon)

        assert kl.shape == (3,)
        assert kl[0] == pytest.approx(0.0, abs=1e-6)
        assert 0 < kl[1] < -np.log(epsilon)  # should be positive and less than -log(epsilon)
        assert 0 < kl[2] < -np.log(epsilon)
        assert kl[1] > kl[2]  # the second case should have higher KL than the third since it's more different from human distribution

    def test_kl_zero_case(self):
        human = np.array([[1.0, 0.0]])
        pred = np.array([[0.0, 1.0]])

        epsilon = 1e-12
        kl = compute_kl_human_to_pred(pred, human, epsilon=epsilon)
        assert kl.shape == (1,)
        assert kl[0] == pytest.approx(-np.log(epsilon), abs=1e-6)  # KL should be -log(epsilon) due to clipping, which is about 27.63 for epsilon=1e-12

class TestJSD:
    def test_compute_jsd(self):
        """Test JSD computation."""
        p = np.array([
            [0.7, 0.2, 0.1], # vs [0.6, 0.3, 0.1] -> JSD > 0 (different)
            [0.0, 0.0, 1.0], # vs [0.0, 0.0, 1.0] -> JSD = 1 (completely different)
            [0.2, 0.3, 0.5], # vs [0.2, 0.3, 0.5] -> JSD = 0 (identical)
            ])
        
        q = np.array([
            [0.6, 0.3, 0.1],
            [1.0, 0.0, 0.0],
            [0.2, 0.3, 0.5],
            ])
        
        d = compute_jsd(p, q)
        # JSD should be non-negative and at most 1 for two distributions
        assert d.shape == (3,)

        assert 0 < d[0] < 1         # JSD should be between 0 and 1 for different distributions
        assert d[1] == pytest.approx(1.0, abs=1e-6)
        assert d[2] == pytest.approx(0.0, abs=1e-6)         # JSD should be 0 if distributions are the same


def test_compute_soft_micro_f1():
    """Test soft micro F1 with known value."""
    model_probs = np.array([[0.8, 0.2], [0.1, 0.9]])
    human_probs = np.array([[1.0, 0.0], [0.0, 1.0]])

    # min_sum = 0.8 + 0.0 + 0.0 + 0.9 = 1.7
    # denom = sum(model + human) = 4.0
    # f1 = 2 * 1.7 / 4 = 0.85
    assert np.allclose(compute_soft_micro_f1(model_probs, human_probs), 0.85, atol=1e-6)


def test_compute_soft_micro_f1_zero_denom():
    """Test soft micro F1 zero denominator edge case."""
    model_probs = np.zeros((2, 3))
    human_probs = np.zeros((2, 3))
    assert compute_soft_micro_f1(model_probs, human_probs) == 0.0


def test_compute_soft_macro_f1():
    """Test soft macro F1 with class-wise averaging."""
    model_probs = np.array([[0.8, 0.2], [0.1, 0.9]])
    human_probs = np.array([[1.0, 0.0], [0.0, 1.0]])

    # class 0: 2 * 0.8 / (0.9 + 1.0) = 1.6 / 1.9
    # class 1: 2 * 0.9 / (1.1 + 1.0) = 1.8 / 2.1
    expected = ((1.6 / 1.9) + (1.8 / 2.1)) / 2
    assert np.allclose(compute_soft_macro_f1(model_probs, human_probs), expected, atol=1e-6)


def test_compute_soft_macro_f1_all_zero():
    """Test soft macro F1 when all class denominators are zero."""
    model_probs = np.zeros((3, 2))
    human_probs = np.zeros((3, 2))
    assert compute_soft_macro_f1(model_probs, human_probs) == 0.0


def test_compute_distance_correlation_doc_example():
    """Matches the documented dcor example for distance correlation."""
    a = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]
    )
    b = np.array([[1.0], [0.0], [0.0], [1.0]])

    d = compute_distance_correlation(a, b)
    assert np.allclose(d, 0.5266403, atol=1e-6)


def test_compute_distance_correlation_self_is_one():
    """Distance correlation should be 1.0 against itself."""
    x = np.array([[0.2, 0.8], [0.6, 0.4], [0.9, 0.1], [0.3, 0.7]])
    d = compute_distance_correlation(x, x)
    assert np.allclose(d, 1.0, atol=1e-6)
