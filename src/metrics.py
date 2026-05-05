import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def calculate_tpr_at_fpr(true_labels, scores, target_fpr=0.05):
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    idx = np.argmin(np.abs(fpr - target_fpr))
    tpr_at_target_fpr = tpr[idx]
    return tpr_at_target_fpr
