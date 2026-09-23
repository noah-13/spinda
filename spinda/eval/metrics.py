"""
Evaluation metrics for HLV.
"""

import numpy as np
from scipy.special import rel_entr
import dcor

def validate_and_fix_probs(
    probs: np.ndarray,
    *,
    name: str = "probs",
    axis: int = 1,
    atol: float = 1e-6,
    require_normalized: bool = True,
) -> np.ndarray:
    """
    Validate probability matrix and fix small floating-point deviations.

    - Requires 2D array [N, num_classes]
    - Non-negative
    - Finite
    - When ``require_normalized`` is true, row sums must be close to 1
      (within ``atol``), and small floating-point deviations are normalized.
      Set it to false for independent multi-label probabilities.
    """
    probs = np.asarray(probs, dtype=np.float64)

    if probs.ndim != 2:
        raise ValueError(f"{name} must be 2D array [N, num_classes]")

    if not np.isfinite(probs).all():
        raise ValueError(f"{name} contains NaN or inf")

    if np.any(probs < 0):
        raise ValueError(f"{name} contains negative values")

    if require_normalized:
        row_sums = probs.sum(axis=axis, keepdims=True)
        if not np.allclose(row_sums, 1.0, atol=atol):
            raise ValueError(
                f"Each row of {name} must sum to 1 within tolerance {atol}"
            )
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

def compute_kl(
    p: np.ndarray,
    q: np.ndarray,
    *,
    epsilon: float = 1e-12,
    atol: float = 1e-6,
) -> np.ndarray:
    """
    Compute per-sample Kullback–Leibler divergence: KL(p || q), log is the natural logarithm (ln).

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
    p : np.ndarray
        First probabilities p, shape [N, K].
    q : np.ndarray
        Second probabilities q, shape [N, K].
    epsilon : float
        Small value to avoid division by zero.
    atol : float
        Allowed tolerance for row sums being close to 1.

    Returns
    -------
    np.ndarray
        Per-sample KL values, shape [N].
    """
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")

    p = validate_and_fix_probs(p, name="p", atol=atol)
    q = validate_and_fix_probs(q, name="q", atol=atol)
    # ----- Numerical stability for KL computation -----
    q = np.clip(q, epsilon, 1.0)

    # Normalize the distributions to ensure they sum to 1
    q = q / np.sum(q, axis=1, keepdims=True)

    return np.sum(rel_entr(p, q), axis=1)

def compute_jsd(
    pred_probs: np.ndarray,
    human_probs: np.ndarray,
    *,
    base: float = 2.0,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """
    Compute per-sample Jensen–Shannon divergence (JSD).

    For distributions ``p`` and ``q``, with ``m = 0.5 * (p + q)``, JSD is:

        JSD(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m)

    This is a symmetric, smoothed version of KL divergence that is always
    finite and bounded. Jensen–Shannon *distance* is the square root of this
    value: ``JS_distance(p, q) = sqrt(JSD(p, q))``. This function deliberately
    returns the divergence, not the distance.

    Properties:
    ----------
    - JSD is in the range [0, log_b(2)] where b is the log base.
    - JSD(p, q) = 0 iff p == q
    - JSD(p, q) = log_b(2) indicates maximal divergence, which occurs when p
      and q have disjoint support (e.g., p=[1, 0], q=[0, 1]).
    - With base=2, JSD is in [0, 1].
    - JSD is symmetric: JSD(p, q) = JSD(q, p)

    Args:
        pred_probs: np.ndarray of shape [N, C]
            Model predicted class probability distributions.
        human_probs: np.ndarray of shape [N, C]
            Human annotation distributions per instance.
        base : float
            Logarithm base for normalization. If base=2, JSD is in [0, 1];
            pass None to retain natural-log units.
        epsilon : float
            Small value to avoid zero probabilities.

    Returns:
        Per-sample Jensen–Shannon divergence values, shape [N].
    """
    m = 0.5 * (pred_probs + human_probs)

    left = compute_kl(pred_probs, m, epsilon=epsilon)   # KL(pred_probs||m) in ln
    right = compute_kl(human_probs, m, epsilon=epsilon)  # KL(human_probs||m) in ln
    jsd = 0.5 * (left + right)              # JSD in ln

    if base is not None:
        jsd = jsd / np.log(base)

    return jsd


def compute_pojsd(
    pred_probs: np.ndarray,
    human_probs: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Compute per-sample positively oriented JSD (PO-JSD) scores.

    PO-JSD is the positively oriented form of Jensen--Shannon divergence, not
    a probability. For each sample it is ``1 - JSD(p, q)`` using base-2
    logarithms and no square root; the dataset-level PO-JSD is the mean of
    these values. It is a similarity score in ``[0, 1]`` where higher is
    better, unlike :func:`compute_jsd`, which returns Jensen--Shannon
    divergence where lower is better.
    """
    jsd = compute_jsd(
        pred_probs,
        human_probs,
        base=2,
        epsilon=epsilon,
    )
    return 1.0 - jsd

"""
Kemal's soft f1 https://arxiv.org/abs/2502.01891
"""

# def soft_micro_f1(y_true: Arr, y_pred: Arr) -> float:
#     "original implementation from Kemal's paper"
#     return 2 * np.where(y_true < y_pred, y_true, y_pred).sum() / (y_true + y_pred).sum()



def compute_soft_micro_f1(
    pred_probs: np.ndarray,
    human_probs: np.ndarray,
    *,
    atol: float = 1e-6,
    ):
    """
    Compute soft micro F1 score between human and model label distributions.

    Reported in Kemal's paper as "Our meta-evaluation shows
    that our proposed soft micro F1 score is one of the best metrics for HLV data."

    Definitions
    ----------
    Let p be the human label distribution and q be the model prediction.

        p = (p_ik),  q = (q_ik)
        where i = 1..N (samples), k = 1..K (classes)
        p_ik >= 0,  q_ik >= 0

    Soft micro F1 is defined as:

        SoftMicroF1(p, q) =
            2 * sum_{i=1..N} sum_{k=1..K} min(p_ik, q_ik)
            ------------------------------------------------
              sum_{i=1..N} sum_{k=1..K} (p_ik + q_ik)

    Interpretation
    ------------------------
    - Measures total overlap between predicted and human label distributions.
    - Symmetric: SoftMicroF1(p, q) = SoftMicroF1(q, p).
    - Bounded in [0, 1].
    - Equals 1 iff p == q for all samples (e.g. [[0.5, 0.5], [0.0, 1.0]] vs [[0.5, 0.5], [0.0, 1.0]])
    - Equals 0 iff there is no overlap at all. (e.g., [[1, 0], [0, 1]] vs [[0, 1], [1, 0]])

    Applicability
    ------------------------
    Soft micro F1 applies to both:

    (1) Multiclass (single-label) classification:
        Each row satisfies the unity constraint:
            sum_k p_ik = 1
            sum_k q_ik = 1

    (2) Multilabel classification:
        No unity constraint; rows need not sum to 1.

    Special Case (Unity Constraint)
    --------------------------------
    In single-label (unity-constrained) classification:

        sum_k p_ik = sum_k q_ik = 1

    Then:

        SoftMicroF1 = SoftAccuracy

    Moreover, in this case:

        SoftMicroF1 = 1 − TVD

    Parameters
    ----------
    pred_probs : np.ndarray
        Model probabilities q, shape [N, K].
    human_probs : np.ndarray
        Human probabilities p, shape [N, K].

    Returns
    -------
    float
        Soft micro F1 score in [0, 1].
    """
    pred_probs = validate_and_fix_probs(pred_probs, name="pred_probs", atol=atol, require_normalized=False)
    human_probs = validate_and_fix_probs(human_probs, name="human_probs", atol=atol, require_normalized=False)

    if pred_probs.shape != human_probs.shape:
        raise ValueError("pred_probs and human_probs must have the same shape")

    min_sum = np.minimum(pred_probs, human_probs).sum()
    denom = (pred_probs + human_probs).sum()
    if pred_probs.shape != human_probs.shape:
        raise ValueError("pred_probs and human_probs must have the same shape")
    return float(2.0 * min_sum / denom) if denom else 1.0


def compute_soft_macro_f1(
    pred_probs: np.ndarray,
    human_probs: np.ndarray,
    *,
    atol: float = 1e-6,
    ):
    """
    Compute soft macro F1 score between human and model label distributions.

    Definitions
    ----------
    For each class k:

        SoftF1_k =
            2 * sum_{i=1..N} min(p_ik, q_ik)
            --------------------------------
              sum_{i=1..N} (p_ik + q_ik)

    The macro variant averages over classes:

        SoftMacroF1(p, q) =
            (1 / K) * sum_{k=1..K} SoftF1_k

    Interpretation
    ------------------------
    Soft macro F1 computes per-class distributional overlap and
    averages them equally across classes.

    - It generalizes standard macro F1 to soft labels.
    - Each class contributes equally, regardless of frequency.
    - Bounded in [0, 1].
    - Equals 1 iff p == q for all samples and classes. (e.g. [[0.5, 0.5], [0.0, 1.0]] vs [[0.5, 0.5], [0.0, 1.0]])
    - Equals 0 iff there is no overlap for any class. (e.g., [[1, 0], [0, 1]] vs [[0, 1], [1, 0]])
    - Designed to address class imbalance

    Parameters
    ----------
        pred_probs : np.ndarray
            Model probabilities q, shape [N, K].
        human_probs : np.ndarray
            Human probabilities p, shape [N, K].
    
    Returns
    -------
    float
        Soft macro F1 score in [0, 1].
    """
    pred_probs = validate_and_fix_probs(pred_probs, name="pred_probs", atol=atol, require_normalized=False)
    human_probs = validate_and_fix_probs(human_probs, name="human_probs", atol=atol, require_normalized=False)

    # sums over examples i, per class k
    if pred_probs.shape != human_probs.shape:
        raise ValueError("pred_probs and human_probs must have the same shape")
    min_sum = np.minimum(pred_probs, human_probs).sum(axis=0)   # shape: (K,)
    denom   = (pred_probs + human_probs).sum(axis=0)            # shape: (K,)

    f1_per_class = np.divide(2.0 * min_sum, denom, out=np.ones_like(denom, dtype=float), where=denom != 0)

    return float(f1_per_class.mean())


def _validate_multilabel_probs(pred_probs, human_probs):
    pred_probs = validate_and_fix_probs(pred_probs, name="pred_probs", require_normalized=False)
    human_probs = validate_and_fix_probs(human_probs, name="human_probs", require_normalized=False)
    if pred_probs.shape != human_probs.shape:
        raise ValueError("pred_probs and human_probs must have the same shape")
    if np.any(pred_probs > 1) or np.any(human_probs > 1):
        raise ValueError("Multilabel probabilities must be in [0, 1]")
    return pred_probs, human_probs


def compute_multilabel_pojsd(pred_probs, human_probs, *, epsilon=1e-12):
    """Mean PO-JSD after treating every instance--label value as Bernoulli."""
    pred_probs, human_probs = _validate_multilabel_probs(pred_probs, human_probs)
    pred = np.stack((pred_probs, 1.0 - pred_probs), axis=-1).reshape(-1, 2)
    human = np.stack((human_probs, 1.0 - human_probs), axis=-1).reshape(-1, 2)
    return float(np.mean(compute_pojsd(pred, human, epsilon=epsilon)))


def compute_multilabel_entropy_correlation(pred_probs, human_probs, *, base=2.0):
    """Macro-average per-label Pearson correlation of Bernoulli entropies."""
    if base <= 0 or base == 1:
        raise ValueError("base must be positive and different from 1")
    pred_probs, human_probs = _validate_multilabel_probs(pred_probs, human_probs)
    def entropy(probs):
        complement = 1.0 - probs
        result = np.zeros_like(probs)
        positive = probs > 0
        result[positive] -= probs[positive] * np.log(probs[positive])
        positive_complement = complement > 0
        result[positive_complement] -= complement[positive_complement] * np.log(complement[positive_complement])
        return result / np.log(base)
    pred_entropy, human_entropy = entropy(pred_probs), entropy(human_probs)
    correlations = []
    for label_idx in range(pred_probs.shape[1]):
        pred_label, human_label = pred_entropy[:, label_idx], human_entropy[:, label_idx]
        if len(pred_label) < 2 or np.isclose(np.std(pred_label), 0.0) or np.isclose(np.std(human_label), 0.0):
            correlations.append(0.0)
        else:
            correlations.append(float(np.corrcoef(pred_label, human_label)[0, 1]))
    return float(np.mean(np.clip(correlations, -1.0, 1.0)))




"""
Beiduo's global corre from seeing the small through the big https://arxiv.org/abs/2406.17600
"""
def compute_distance_correlation(
    pred_probs: np.ndarray,
    human_probs: np.ndarray,
    *,
    exponent: float = 1.0,
    atol: float = 1e-6,
) -> float:
    """
    Compute (biased) distance correlation between two sets of vectors.

    Definitions
    ----------
    Let X = {x_i}_{i=1..N} and Y = {y_i}_{i=1..N} be two collections
    of D-dimensional vectors.

    Distance correlation is defined as:

        dCor^2(X, Y) =
            dCov^2(X, Y)
            ------------------------------
            sqrt(dVar^2(X) * dVar^2(Y))

    where:

        dCov^2(X, Y)  = distance covariance
        dVar^2(X)     = distance variance of X
        dVar^2(Y)     = distance variance of Y

    Intuition
    ------------------------
    Distance correlation measures the similarity of *global structure*
    between two datasets.

    In HLV distribution comparison settings (e.g., MJD vs HJD):

        - Each row is a soft label vector for one instance.
        - The metric evaluates whether the geometry of the entire
          point cloud is preserved.

    Properties
    ------------------------
    - Bounded in [0, 1].
    - Equals 0 iff X and Y are statistically independent.
    - Equals 1 iff Y is a deterministic function of X (perfect structural alignment).
    - Sensitive to global configuration rather than per-sample overlap.
    - Invariant to translation and orthogonal transformation.
    - Symmetric: dCor(X, Y) = dCor(Y, X)

    Parameters
    ----------
    pred_probs : np.ndarray
        Model probabilities q, shape [N, K].
    human_probs : np.ndarray
        Human probabilities p, shape [N, K].
    exponent : float, optional
        Power of Euclidean distance used in computation (default=1.0).
    atol : float, optional
        Allowed tolerance for row sums being close to 1.
    Returns
    -------
    float
        Distance correlation in [0, 1].
    """
    p = validate_and_fix_probs(pred_probs , name="pred_probs", atol=atol)
    q = validate_and_fix_probs(human_probs, name="human_probs", atol=atol)

    return float(dcor.distance_correlation(p, q, exponent=exponent))


"""
Entropy correlation for human label variation (HLV) evaluation.

Sources:
- Uma et al. (2020): entropy correlation for comparing model and human
  disagreement/uncertainty.
- Kurniawan et al. (2026): included in systematic HLV metric comparison.

Unlike ``compute_distance_correlation``, this metric computes Pearson
correlation over per-instance Shannon entropy values. It measures alignment
in uncertainty, not full class-distribution geometry.
"""
def compute_entropy_correlation(
    pred_probs: np.ndarray,
    human_probs: np.ndarray,
    *,
    base: float = 2.0,
    atol: float = 1e-6,
) -> float:
    """Compute Pearson correlation between model and human entropies.

    A high score means the model identifies the same examples as more or less
    uncertain than humans do. It does not establish class-level distributional
    agreement, so report it together with PO-JSD or TVD.

    Returns ``0.0`` when correlation is undefined (fewer than two examples or
    either entropy vector has zero variance), keeping evaluation output finite.
    """
    pred_probs = validate_and_fix_probs(pred_probs, name="pred_probs", atol=atol)
    human_probs = validate_and_fix_probs(human_probs, name="human_probs", atol=atol)
    if pred_probs.shape != human_probs.shape:
        raise ValueError("pred_probs and human_probs must have the same shape")
    if base <= 0 or base == 1:
        raise ValueError("base must be positive and different from 1")

    def _entropy(probs: np.ndarray) -> np.ndarray:
        log_probs = np.zeros_like(probs)
        positive = probs > 0
        log_probs[positive] = np.log(probs[positive])
        return -np.sum(probs * log_probs, axis=1) / np.log(base)

    pred_entropy = _entropy(pred_probs)
    human_entropy = _entropy(human_probs)
    if (
        len(pred_entropy) < 2
        or np.isclose(np.std(pred_entropy), 0.0)
        or np.isclose(np.std(human_entropy), 0.0)
    ):
        return 0.0
    return float(np.corrcoef(pred_entropy, human_entropy)[0, 1])


def compute_euclidean_distance(
    pred_probs: np.ndarray,
    human_probs: np.ndarray,
    *,
    atol: float = 1e-6,
) -> np.ndarray:
    """
    Compute per-sample Euclidean distance (L2 distance) between distributions.

    Definitions
    ----------
    Let p be the target human label distribution (soft label)[cite: 16],
    and q be the model's predicted probability distribution[cite: 26].

        L2(p, q) = sqrt( sum_{i=1..K} (p_i - q_i)^2 ) [cite: 43]

    Intuitive Properties & Examples
    -------------------------------
    1. Fair Penalization: Unlike Cross Entropy, L2 treats errors of the same 
       magnitude equally. 
       - Example: If target p = [0.5, 0.5].
       - Prediction q1 = [0.7, 0.3] (0.2 error) and q2 = [0.3, 0.7] (0.2 error) 
         will receive the exact same penalty.
    
    2. Zero on Perfect Match: If the model predicts the human distribution 
       exactly (p == q), the score is always 0.0, regardless of how much 
       disagreement (entropy) is in the label.
    
    3. Scale Sensitivity: If the gap between two distributions doubles, 
       the L2 distance doubles.
       - Example: L2([0.2, 0.8], [0.1, 0.9]) is exactly half of 
         L2([0.4, 1.6], [0.2, 1.8]).


    Parameters
    ----------
    pred_probs : np.ndarray
        Model probabilities q, shape [N, K].
    human_probs : np.ndarray
        Human probabilities p, shape [N, K].
    atol : float
        Allowed tolerance for row sums being close to 1.

    Returns
    -------
    np.ndarray
        Per-sample L2 values, shape [N].
    """
    if pred_probs.shape != human_probs.shape:
        raise ValueError("pred_probs and human_probs must have the same shape")

    # Assuming validate_and_fix_probs is defined elsewhere as in your snippet
    p = validate_and_fix_probs(human_probs, name="human_probs", atol=atol)
    q = validate_and_fix_probs(pred_probs, name="pred_probs", atol=atol)

    return np.sqrt(np.sum(np.square(p - q), axis=1))


def compute_cross_entropy(
    p: np.ndarray,
    q: np.ndarray,
    *,
    epsilon: float = 1e-12,
    atol: float = 1e-6,
) -> np.ndarray:
    """
    Compute per-sample Cross Entropy: H(p, q) = - sum(p * log(q)).

    Definitions
    ----------
    H(p, q) measures the difference between target distribution p and 
    predicted distribution q. 
    H(p, q) = KL(p || q) + Entropy(p).

    Intuitive Paradoxes & Examples (Why it can be "unfair")
    ------------------------------------------------------
    1. Unfair Penalization on Perfect Match: Two models can both be 
       "perfect" but get different scores.
       - Example: Item A has total human agreement p=[1.0, 0.0]. 
         Model 1 predicts [1.0, 0.0], Score = 0.0.
       - Item B has high human disagreement p=[0.5, 0.5]. 
         Model 2 predicts [0.5, 0.5], Score = 0.69.
       - Model 2 is penalized more simply because the humans disagreed.

    2. Asymmetry: Swapping the "Target" and "Prediction" changes the score.
       - Example: H(Target=[0.8, 0.2], Pred=[0.5, 0.5]) = 0.69.
       - H(Target=[0.5, 0.5], Pred=[0.8, 0.2]) = 0.97.

    3. Boundary Sensitivity: CE punishes "confident but wrong" predictions 
       extremely harshly compared to distance-based metrics.

    Parameters
    ----------
    p : np.ndarray
        First probabilities p, shape [N, K].
    q : np.ndarray
        Second probabilities q, shape [N, K].
    epsilon : float
        Small value to avoid log(0).
    atol : float
        Allowed tolerance for row sums.

    Returns
    -------
    np.ndarray
        Per-sample Cross Entropy values, shape [N].
    """
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")

    p = validate_and_fix_probs(p, name="p", atol=atol)
    q = validate_and_fix_probs(q, name="q", atol=atol)
    # Numerical stability: prevents log(0) = -inf
    q_stable = np.clip(q, epsilon, 1.0)
    
    # Re-normalize after clipping to ensure distribution sums to 1
    q_stable = q_stable / np.sum(q_stable, axis=1, keepdims=True)

    return -np.sum(p * np.log(q_stable), axis=1)
