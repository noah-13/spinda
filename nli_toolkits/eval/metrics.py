"""
Evaluation metrics for NLI calibration.

"""

import numpy as np
from typing import Optional
from scipy.special import rel_entr
import math
import dcor


import numpy as np

def validate_and_fix_probs(
    probs: np.ndarray,
    *,
    name: str = "probs",
    axis: int = 1,
    atol: float = 1e-6,
) -> np.ndarray:
    """
    Validate probability matrix and fix small floating-point deviations.

    - Requires 2D array [N, num_classes]
    - Non-negative
    - Finite
    - Row sums close to 1 (within atol)
    - Automatically renormalizes small deviations
    """
    probs = np.asarray(probs, dtype=np.float64)

    if probs.ndim != 2:
        raise ValueError(f"{name} must be 2D array [N, num_classes]")

    if not np.isfinite(probs).all():
        raise ValueError(f"{name} contains NaN or inf")

    if np.any(probs < 0):
        raise ValueError(f"{name} contains negative values")

    row_sums = probs.sum(axis=axis, keepdims=True)

    if not np.allclose(row_sums, 1.0, atol=atol):
        raise ValueError(
            f"Each row of {name} must sum to 1 within tolerance {atol}"
        )

    # Fix tiny floating point drift
    probs = probs / row_sums

    return probs

"""
Implements metrics from:
"Stop Measuring Calibration When Humans Disagree" (Baan et al., EMNLP 2022)
- DistCE (Distribution Calibration Error) = TVD(model_probs, human_probs)
"""

def compute_tvd(
    model_probs: np.ndarray,
    human_probs: np.ndarray,
) -> np.ndarray:
    """
    Compute the instance-level Total Variation Distance (TVD) between
    model predictive distributions and human annotation distributions.

    In Baan et al. (2022), this quantity is referred to as
    Distribution Calibration Error (DistCE), defined as:

        DistCE(x) = TVD(f(x), π̄(x))

    where:
        - f(x) is the model predicted probability distribution
        - π̄(x) is the empirical human label distribution

    TVD is defined as:

        TVD(p, q) = 0.5 * ||p - q||_1
                  = 0.5 * sum_c |p_c - q_c|

    Properties:
        - TVD ∈ [0, 1]
        - TVD = 0 iff the two distributions are identical
        - TVD = 1 indicates maximal discrepancy

    Args:
        model_probs: np.ndarray of shape [N, C]
            Model predicted class probability distributions.
        human_probs: np.ndarray of shape [N, C]
            Human annotation distributions per instance.

    Returns:
        np.ndarray of shape [N]
            Per-instance TVD (DistCE) values.
    """
    model_probs = validate_and_fix_probs(model_probs, name="model_probs")
    human_probs = validate_and_fix_probs(human_probs, name="human_probs")
    
    if model_probs.shape != human_probs.shape:
        raise ValueError("model_probs and human_probs must have the same shape")
    
    # ----- TVD computation -----
    # TVD = 0.5 * L1 norm
    l1_norm = np.abs(model_probs - human_probs).sum(axis=1)
    tvd = 0.5 * l1_norm
    
    return tvd

# def tvd(model_probs: np.ndarray, human_probs: np.ndarray, mean_per: Optional[str] = None):
#     """
#     Original TVD computation from Baan et al. (2022), allowing for multiple sub-samples and groups.
#     Computes TVD scores allowing for multiple sub-samples and groups (=classifiers).

#     p: classifiers [G, 1, N, C]
#     q: MLE given (sub-samples of) annotations [1, S, N, C]

#     returns:
#         tvd: [G, S, N] (mean_per=None), [G, S] (mean_per=sample), [G, N] (mean_per=instance)
#     """
#     assert model_probs.max() <= 1.0 and model_probs.min() >= 0
#     assert human_probs.max() <= 1.0 and human_probs.min() >= 0

#     tvds = np.sum(np.abs(model_probs - human_probs), axis=-1) / 2
#     if mean_per is not None:
#         if mean_per == "instance":
#             tvds = tvds.mean(1)
#         elif mean_per == "sample":
#             tvds = tvds.mean(2)
#     return tvds

# # Implementation from Beiduo's seeing the small through the big https://arxiv.org/abs/2406.17600

# def kl_divergence(P, Q, epsilon=1e-10):
#     """
#     Calculate the Kullback-Leibler divergence between two probability distributions.

#     Parameters:
#     P (array-like): The first probability distribution.
#     Q (array-like): The second probability distribution.
#     epsilon (float): A small value to avoid division by zero.

#     Returns:
#     float: The KL divergence value.
#     """
#     # Convert P and Q to numpy arrays
#     P = np.asarray(P, dtype=np.float64)
#     Q = np.asarray(Q, dtype=np.float64)
    
#     # Add epsilon to avoid zero probabilities
#     P = np.clip(P, epsilon, 1)
#     Q = np.clip(Q, epsilon, 1)
    
#     # Normalize the distributions to ensure they sum to 1
#     P = P / np.sum(P)
#     Q = Q / np.sum(Q)
    
#     # Calculate KL divergence
#     kl_divergence_value = np.sum(rel_entr(P, Q))
    
#     return kl_divergence_value


# def jensen_shannon(p, q):
#     # calculate JSD
#     p = np.asarray(p, dtype=np.float64)
#     q = np.asarray(q, dtype=np.float64)
#     m = (p + q) / 2
#     return math.sqrt((kl_divergence(p, m) + kl_divergence(q, m)) / 2)

def compute_kl_human_to_pred(
    pred_probs: np.ndarray,
    human_probs: np.ndarray,
    *,
    epsilon: float = 1e-12,
    atol: float = 1e-6,
) -> np.ndarray:
    """
    Compute per-sample Kullback–Leibler divergence: KL(human || pred), log is the natural logarithm (ln).

    Definitions
    ----------
    Let p be the estimated human label distribution for a sample (a categorical distribution),
    and q be the model's predicted probability distribution (e.g., softmax output).

        p = (p_1, ..., p_K),  p_i >= 0,  sum_i p_i = 1
        q = (q_1, ..., q_K),  q_i >= 0,  sum_i q_i = 1

    The KL divergence from p to q is:

        KL(p || q) = sum_{i=1..K} p_i * log( p_i / q_i )

    Interpretation
    ------------------------
    KL(p || q) measures how well q approximates p, from the perspective of p:

    - Larger KL means q assigns probability mass differently than humans do.
    - It is *not symmetric*: KL(p || q) != KL(q || p) generally.
    - In theory, KL ∈ [0, inf] 
    - KL can be infinite if q_i == 0 for some i where p_i > 0.
    - KL(p || q) = 0  iff  p == q (exact match)
    - Here clipping q to [epsilon, 1] ensures numerical stability and prevents infinite KL values, so KL is actually bounded by -log(epsilon), for epsilon=1e-12, it is about 27.63.

    Parameters
    ----------
    pred_probs : np.ndarray
        Model probabilities q, shape [N, K].
    human_probs : np.ndarray
        Human probabilities p, shape [N, K].
    epsilon : float
        Small value to avoid division by zero.
    atol : float
        Allowed tolerance for row sums being close to 1.

    Returns
    -------
    np.ndarray
        Per-sample KL values, shape [N].
    """
    if pred_probs.shape != human_probs.shape:
        raise ValueError("model_probs and human_probs must have the same shape")

    p = validate_and_fix_probs(human_probs, name="human_probs", atol=atol)
    q = validate_and_fix_probs(pred_probs, name="pred_probs", atol=atol)

    # ----- Numerical stability for KL computation -----
    q = np.clip(q, epsilon, 1.0)

    # Normalize the distributions to ensure they sum to 1
    q = q / np.sum(q, axis=1, keepdims=True)

    return np.sum(rel_entr(p, q), axis=1)

def compute_jsd(
    p: np.ndarray,
    q: np.ndarray,
    *,
    base: float = 2.0,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """
    Per-sample Jensen–Shannon distance.

    JSD is defined as:

    JSD(p, q) = sqrt( 0.5 * KL(p || m) + 0.5 * KL(q || m) )
    where m = 0.5 * (p + q)

    JSD is a symmetric and smoothed version of KL divergence, and is always finite and bounded.

    Properties:
    ----------
    - JSD is in the range [0, sqrt(log_b(2))] where b is the log base.
    - JSD(p, q) = 0 iff p == q
    - JSD(p, q) = sqrt(log_b(2)) indicates maximal divergence, which occurs when p and q are completely disjoint distributions (e.g., p=[1,0], q=[0,1]).
    - So only if base=2, result is in [0, 1].
    - JSD is symmetric: JSD(p, q) = JSD(q, p)

    Parameters
    ----------
    p : np.ndarray  
        First probability distribution (e.g., human), shape [N, K].
    q : np.ndarray
        Second probability distribution (e.g., model), shape [N, K].
    base : float
        Logarithm base for normalization. If base=2, JSD is in [0, 1].
    epsilon : float
        Small value to avoid zero probabilities.

    Returns
    -------
    np.ndarray
        Per-sample JSD values, shape [N].
    """
    m = 0.5 * (p + q)

    left = compute_kl_human_to_pred(m, p, epsilon=epsilon)   # KL(p||m) in ln
    right = compute_kl_human_to_pred(m, q, epsilon=epsilon)  # KL(q||m) in ln
    jsd = 0.5 * (left + right)              # JSD in ln

    if base is not None:
        jsd = jsd / np.log(base)

    return np.sqrt(jsd)

"""
Kemal's soft f1 https://arxiv.org/abs/2502.01891
"""

def compute_soft_micro_f1(model_probs, human_probs):
    min_sum = np.minimum(model_probs, human_probs).sum()
    denom = (model_probs + human_probs).sum()
    if denom == 0:
        return 0.0
    return 2.0 * min_sum / denom

def compute_soft_macro_f1(model_probs, human_probs):
    min_sum = np.minimum(model_probs, human_probs).sum(axis=0)
    pred_sum = model_probs.sum(axis=0)
    true_sum = human_probs.sum(axis=0)
    denom = pred_sum + true_sum

    mask = denom > 0
    if not np.any(mask):
        return 0.0

    f1 = 2.0 * min_sum[mask] / denom[mask]
    return float(f1.mean())


"""
Beiduo's global corre from seeing the small through the big https://arxiv.org/abs/2406.17600
"""
def compute_distance_correlation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    exponent: float = 1.0,
) -> float:
    """
    Biased estimator of distance correlation between two random vectors.

    Inputs are expected as [N, D], where rows are instances and columns are
    variables.
    """

    return float(dcor.distance_correlation(x, y, exponent=exponent))
