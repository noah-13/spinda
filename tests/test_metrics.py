"""Tests for evaluation metrics."""

import numpy as np
import pytest

from nli_toolkits.eval.metrics import (
    compute_distce,
    compute_ece,
    compute_entce,
    compute_rankcs,
)


def test_compute_ece_perfect_calibration():
    """Test ECE with perfectly calibrated predictions."""
    n = 1000
    # Create perfectly calibrated scenario:
    # For each confidence level, accuracy equals confidence
    # We'll use multiple confidence levels to test binning
    num_bins = 10
    bin_size = n // num_bins
    
    predictions = []
    confidences = []
    labels = []
    
    np.random.seed(42)  # For reproducibility
    for i in range(num_bins):
        # Confidence for this bin (0.1 to 1.0)
        conf = (i + 1) / num_bins
        # Create bin_size samples with this confidence
        for j in range(bin_size):
            pred = np.random.randint(0, 3)
            predictions.append(pred)
            confidences.append(conf)
            
            # For perfect calibration: accuracy should equal confidence
            # So if confidence is c, then c fraction should be correct
            # Use random to decide if this prediction is correct
            is_correct = np.random.random() < conf
            if is_correct:
                labels.append(pred)  # Correct prediction
            else:
                # Wrong prediction: choose a different label
                wrong_label = (pred + 1) % 3
                labels.append(wrong_label)
    
    predictions = np.array(predictions)
    confidences = np.array(confidences)
    labels = np.array(labels)
    
    ece = compute_ece(predictions, confidences, labels, num_bins=num_bins)
    
    # Should be close to 0 for perfect calibration
    assert ece >= 0
    assert ece < 0.1  # Allow some numerical error due to randomness


def test_compute_ece_miscalibrated():
    """Test ECE with miscalibrated predictions."""
    n = 1000
    predictions = np.random.randint(0, 3, n)
    confidences = np.ones(n) * 0.9  # High confidence
    labels = np.random.randint(0, 3, n)  # Random labels (low accuracy)
    
    ece = compute_ece(predictions, confidences, labels)
    # Should be high for miscalibration
    assert ece > 0.5


def test_compute_entce():
    """Test EntCE computation."""
    n = 10
    num_classes = 3
    
    # Model predictions (uniform)
    model_probs = np.ones((n, num_classes)) / num_classes
    
    # Human distribution (also uniform)
    human_probs = np.ones((n, num_classes)) / num_classes
    
    entce = compute_entce(model_probs, human_probs)
    assert entce.shape == (n,)
    # Should be close to 0 when entropies match
    assert np.allclose(entce, 0.0, atol=1e-6)


def test_compute_rankcs_perfect_match():
    """Test RankCS with perfect ranking match."""
    n = 10
    num_classes = 3
    
    # Same distributions
    model_probs = np.array([[0.7, 0.2, 0.1], [0.5, 0.3, 0.2]] * (n // 2))
    human_probs = model_probs.copy()
    
    rankcs = compute_rankcs(model_probs, human_probs)
    assert rankcs == 1.0


def test_compute_rankcs_no_match():
    """Test RankCS with completely different rankings."""
    n = 10
    num_classes = 3
    
    # Opposite rankings
    model_probs = np.array([[0.7, 0.2, 0.1]] * n)
    human_probs = np.array([[0.1, 0.2, 0.7]] * n)
    
    rankcs = compute_rankcs(model_probs, human_probs)
    assert rankcs == 0.0


def test_compute_distce():
    """Test DistCE computation."""
    n = 10
    
    # Model predictions
    model_probs = np.array([[0.7, 0.2, 0.1]] * n)
    
    # Human distribution (same)
    human_probs = model_probs.copy()
    
    distce = compute_distce(model_probs, human_probs)
    assert distce.shape == (n,)
    # Should be 0 for identical distributions
    assert np.allclose(distce, 0.0, atol=1e-6)


def test_compute_distce_different():
    """Test DistCE with different distributions."""
    
    # Model: [1.0, 0.0, 0.0]
    model_probs = np.array([[1.0, 0.0, 0.0]])
    
    # Human: [0.0, 0.0, 1.0]
    human_probs = np.array([[0.0, 0.0, 1.0]])
    
    distce = compute_distce(model_probs, human_probs)
    # TVD = 0.5 * (|1-0| + |0-0| + |0-1|) = 0.5 * 2 = 1.0
    assert np.allclose(distce, 1.0, atol=1e-6)
