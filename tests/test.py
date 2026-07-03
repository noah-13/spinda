from hlv_toolkits.eval import compute_kl_human_to_pred
import numpy as np

human_q = np.array([[0.5, 0.3, 0.2], [0.0, 0.0, 1.0]])
pred_p = np.array([[0.4, 0.4, 0.2], [1.0, 0.0, 0.0]])

kl = compute_kl_human_to_pred(pred_p, human_q)
print("KL divergence (human || pred):", kl)