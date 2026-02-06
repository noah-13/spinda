"""
Evaluation metrics for NLI calibration.

Implements metrics from:
"Stop Measuring Calibration When Humans Disagree" (Baan et al., EMNLP 2022)
TVD-based DistCE
"""

import numpy as np
from typing import Optional
from scipy.special import rel_entr
import math

# def compute_ece(
#     predictions: np.ndarray,
#     confidences: np.ndarray,
#     labels: np.ndarray,
#     num_bins: int = 15,
# ) -> float:
#     """
#     Compute Expected Calibration Error (ECE).
    
#     Formula from Guo et al. (2017), adapted for multi-class:
#     ECE = Σ_m |B_m|/N * |acc(B_m) - conf(B_m)|
    
#     Args:
#         predictions: Predicted class labels (shape: [N])
#         confidences: Maximum predicted probabilities (shape: [N])
#         labels: Ground truth labels (shape: [N])
#         num_bins: Number of bins for discretization (default: 15)
        
#     Returns:
#         ECE score (lower is better)
#     """
#     n = len(predictions)
#     if n == 0:
#         return 0.0
    
#     # Create bins
#     bin_boundaries = np.linspace(0, 1, num_bins + 1)
#     bin_lowers = bin_boundaries[:-1]
#     bin_uppers = bin_boundaries[1:]
    
#     ece = 0.0
#     for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
#         # Find predictions in this bin
#         in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
#         if bin_lower == 0.0:
#             # Include the lower boundary
#             in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        
#         prop_in_bin = in_bin.mean()
#         if prop_in_bin > 0:
#             # Accuracy in this bin
#             accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
#             # Average confidence in this bin
#             avg_confidence_in_bin = confidences[in_bin].mean()
#             ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
#     return float(ece)


# def compute_entce(
#     model_probs: np.ndarray,
#     human_probs: np.ndarray,
# ) -> np.ndarray:
#     """
#     Compute Human Entropy Calibration Error (EntCE) per instance.
    
#     Formula: EntCE(x) = H(f(x)) - H(π̄(x))
#     where H(p) = -Σ p_i * log(p_i) is the entropy.
    
#     Args:
#         model_probs: Model predicted probabilities (shape: [N, num_classes])
#         human_probs: Human annotation distribution (shape: [N, num_classes])
        
#     Returns:
#         Array of EntCE values per instance (shape: [N])
#     """
#     # Add small epsilon to avoid log(0)
#     eps = 1e-10
    
#     # Compute entropy for model predictions
#     model_entropy = -np.sum(model_probs * np.log(model_probs + eps), axis=1)
    
#     # Compute entropy for human distributions
#     human_entropy = -np.sum(human_probs * np.log(human_probs + eps), axis=1)
    
#     # EntCE = model_entropy - human_entropy
#     entce = model_entropy - human_entropy
    
#     return entce


# def compute_rankcs(
#     model_probs: np.ndarray,
#     human_probs: np.ndarray,
# ) -> float:
#     """
#     Compute Human Ranking Calibration Score (RankCS).
    
#     Formula: RankCS = 1/N * Σ_n [argsort(f(x_n)) == argsort(π̄(x_n))]
    
#     Measures whether the model's class ranking matches human ranking.
    
#     Args:
#         model_probs: Model predicted probabilities (shape: [N, num_classes])
#         human_probs: Human annotation distribution (shape: [N, num_classes])
        
#     Returns:
#         RankCS score (higher is better, range: [0, 1])
#     """
#     n = model_probs.shape[0]
#     if n == 0:
#         return 0.0
    
#     matches = 0
#     for i in range(n):
#         model_ranking = np.argsort(model_probs[i])[::-1]  # Descending order
#         human_ranking = np.argsort(human_probs[i])[::-1]  # Descending order
        
#         if np.array_equal(model_ranking, human_ranking):
#             matches += 1
    
#     return matches / n


def compute_distce(
    model_probs: np.ndarray,
    human_probs: np.ndarray,
) -> np.ndarray:
    """
    Compute Human Distribution Calibration Error (DistCE) per instance.
    
    Formula: DistCE(x) = TVD(f(x), π̄(x))
    where TVD (Total Variation Distance) = 0.5 * ||p - q||_1
    
    Args:
        model_probs: Model predicted probabilities (shape: [N, num_classes])
        human_probs: Human annotation distribution (shape: [N, num_classes])
        
    Returns:
        Array of DistCE values per instance (shape: [N])
    """
    # TVD = 0.5 * L1 norm
    l1_norm = np.abs(model_probs - human_probs).sum(axis=1)
    distce = 0.5 * l1_norm
    
    return distce

def tvd(model_probs: np.ndarray, human_probs: np.ndarray, mean_per: Optional[str] = None):
    """
    Original TVD computation from Baan et al. (2022), allowing for multiple sub-samples and groups.
    Computes TVD scores allowing for multiple sub-samples and groups (=classifiers).

    p: classifiers [G, 1, N, C]
    q: MLE given (sub-samples of) annotations [1, S, N, C]

    returns:
        tvd: [G, S, N] (mean_per=None), [G, S] (mean_per=sample), [G, N] (mean_per=instance)
    """
    assert model_probs.max() <= 1.0 and model_probs.min() >= 0
    assert human_probs.max() <= 1.0 and human_probs.min() >= 0

    tvds = np.sum(np.abs(model_probs - human_probs), axis=-1) / 2
    if mean_per is not None:
        if mean_per == "instance":
            tvds = tvds.mean(1)
        elif mean_per == "sample":
            tvds = tvds.mean(2)
    return tvds



def compute_kl(P, Q, epsilon=1e-10):
    """
    Kullback–Leibler divergence KL(P || Q).

    Notes
    -----
    - Both P and Q will be clipped by `epsilon` for numerical stability.
    - Both distributions are re-normalized to ensure sum to 1.
    - Natural logarithm (ln) is used internally.
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    # numerical stability
    P = np.clip(P, epsilon, 1.0)
    Q = np.clip(Q, epsilon, 1.0)

    # normalize
    P = P / np.sum(P)
    Q = Q / np.sum(Q)

    return np.sum(rel_entr(P, Q))  # ln-based KL


def compute_jsd(
    p,
    q,
    base=2,
    epsilon=1e-10,
):
    """
    Jensen–Shannon distance between two probability distributions.

    This function returns the *Jensen–Shannon distance*, i.e.
    the square root of the Jensen–Shannon divergence:

        JSDist(p, q) = sqrt( ( KL(p || m) + KL(q || m) ) / 2 )

    where m = (p + q) / 2.

    Parameters
    ----------
    p, q : array-like
        Input probability distributions.
    base : float, optional (default=2)
        Logarithm base used for the divergence.
        - base = 2  → distance is bounded in [0, 1]
        - base = e  → distance is bounded in [0, sqrt(ln 2)]
    epsilon : float
        Small value for clipping to avoid log(0).

    Notes
    -----
    - Internally uses natural logarithm and rescales by ln(base).
    - Distributions are clipped and re-normalized for stability.
    - If you want the Jensen–Shannon *divergence* instead of distance,
      remove the final sqrt.
    """

    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # numerical stability
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)

    # normalize
    p = p / np.sum(p)
    q = q / np.sum(q)

    m = 0.5 * (p + q)

    js_div = 0.5 * (
        compute_kl(p, m, epsilon=epsilon)
        + compute_kl(q, m, epsilon=epsilon)
    )

    # change log base if needed
    if base is not None:
        js_div /= math.log(base)

    # return distance (not divergence)
    return math.sqrt(js_div)


# def entropy(probs: np.ndarray, axis: int = -1) -> np.ndarray:
#     """
#     Compute entropy of probability distributions.
    
#     Args:
#         probs: Probability distributions (shape: [..., num_classes])
#         axis: Axis along which to compute entropy
        
#     Returns:
#         Entropy values (shape: probs.shape without axis dimension)
#     """
#     eps = 1e-10
#     return -np.sum(probs * np.log(probs + eps), axis=axis)
