# effnet_eval/metrics.py

import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    average_precision_score,
    roc_curve,
    roc_auc_score,
)
from Scripts.StatisticsAndResults.effnet_eval.config import *


def calculate_prauc_ci(y_true, y_scores, n_bootstraps=N_BOOTSTRAPS, alpha=0.95):
    """
    Compute bootstrap confidence interval for PR AUC.
    
    Parameters:
        y_true (array): Ground truth binary labels.
        y_scores (array): Predicted probabilities.
        n_bootstraps (int): Number of bootstrap samples.
        alpha (float): Confidence level (default 95%).

    Returns:
        tuple: (lower_bound, upper_bound) of PR AUC.
    """
    bootstrapped_scores = []
    rng = np.random.RandomState(42)

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_scores), len(y_scores))
        if len(np.unique(y_true[indices])) < 2:
            continue

        try:
            if PRAUC_MACRO_AVERAGE:
                score = average_precision_score(y_true[indices], y_scores[indices], average='macro')
            else:
                p, r, _ = precision_recall_curve(y_true[indices], y_scores[indices])
                score = auc(r, p)
            bootstrapped_scores.append(score)
        except:
            continue

    if not bootstrapped_scores:
        return np.nan, np.nan

    percentiles = np.percentile(
        bootstrapped_scores, [(1 - alpha) / 2 * 100, (1 + alpha) / 2 * 100]
    )
    return tuple(percentiles)


def calculate_roc_auc_ci(y_true, y_scores, n_bootstraps=N_BOOTSTRAPS, alpha=0.95):
    """
    Compute bootstrap confidence interval for ROC AUC.
    
    Parameters:
        y_true (array): Ground truth binary labels.
        y_scores (array): Predicted probabilities.
        n_bootstraps (int): Number of bootstrap samples.
        alpha (float): Confidence level (default 95%).

    Returns:
        tuple: (lower_bound, upper_bound) of ROC AUC.
    """
    bootstrapped_scores = []
    rng = np.random.RandomState(42)

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_scores), len(y_scores))
        if len(np.unique(y_true[indices])) < 2:
            continue

        try:
            score = roc_auc_score(y_true[indices], y_scores[indices])
            bootstrapped_scores.append(score)
        except:
            continue

    if not bootstrapped_scores:
        return np.nan, np.nan

    percentiles = np.percentile(
        bootstrapped_scores, [(1 - alpha) / 2 * 100, (1 + alpha) / 2 * 100]
    )
    return tuple(percentiles)


def calculate_pr_curve(y_true, y_scores):
    """
    Compute the Precision-Recall curve and AUC.

    Returns:
        tuple: precision, recall, thresholds, auc_value
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    auc_val = auc(recall, precision)
    return precision, recall, thresholds, auc_val


def calculate_roc_curve(y_true, y_scores):
    """
    Compute the ROC curve and AUC.

    Returns:
        tuple: fpr, tpr, thresholds, auc_value
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_val = roc_auc_score(y_true, y_scores)
    return fpr, tpr, thresholds, auc_val
