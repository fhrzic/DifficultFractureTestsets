# effnet_eval/plotting.py

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

from Scripts.StatisticsAndResults.effnet_eval.metrics import *
from Scripts.StatisticsAndResults.effnet_eval.config import *

def plot_roc_auc_curves(df, efficientnets, output_dir):
    """
    Generate ROC AUC plots with bootstrapped confidence intervals.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'efficientnet', 'ground_truth', 'confidence', and 'testset_name'.
        efficientnets (list): List of EfficientNet model names to evaluate.
        output_dir (str): Directory to save the plots and result tables.

    Returns:
        pd.DataFrame: Summary of ROC AUC scores and confidence intervals.
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    auc_data = []

    for idx, model in enumerate(tqdm(efficientnets, desc="ROC AUC")):
        if idx >= 9:
            break
        ax = axes[idx]
        model_label = efficientnet_dict.get(model, model)
        df_model = df[df['efficientnet'] == model]
        if df_model.empty:
            ax.axis('off')
            continue

        for proj in df_model['testset_name'].unique():
            df_proj = df_model[df_model['testset_name'] == proj]
            if df_proj.empty:
                continue

            y_true = df_proj['ground_truth'].values
            y_scores = df_proj['confidence'].astype(float).values

            fpr, tpr, _ = bootstrap_roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            lower, upper = calculate_roc_auc_ci(y_true, y_scores)

            label = projections_dict.get(proj, proj)
            color = colors.get(proj, 'black')
            ax.plot(fpr, tpr, label=f"{label} (AUC {roc_auc:.3f})", color=color, linewidth=2)
            ax.fill_between(fpr, tpr - 0.05, tpr + 0.05, color=color, alpha=0.2)

            auc_data.append({
                "EfficientNet": model_label,
                "Testset": label,
                "AUC": roc_auc,
                "95% CI Lower": lower,
                "95% CI Upper": upper
            })

        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax.set_title(f"ROC - {model_label}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_auc_effnet_proj.png"))
    plt.savefig(os.path.join(output_dir, "roc_auc_effnet_proj.pdf"))
    plt.close()

    return pd.DataFrame(auc_data)

def plot_precision_recall_curves(df, efficientnets, output_dir):
    """
    Generate Precision-Recall plots with bootstrapped confidence intervals.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'efficientnet', 'ground_truth', 'confidence', and 'testset_name'.
        efficientnets (list): List of EfficientNet model names to evaluate.
        output_dir (str): Directory to save the plots and result tables.

    Returns:
        pd.DataFrame: Summary of PR AUC scores and confidence intervals.
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    auc_data = []

    for idx, model in enumerate(tqdm(efficientnets, desc="PR AUC")):
        if idx >= 9:
            break
        ax = axes[idx]
        model_label = efficientnet_dict.get(model, model)
        df_model = df[df['efficientnet'] == model]
        if df_model.empty:
            ax.axis('off')
            continue

        for proj in df_model['testset_name'].unique():
            df_proj = df_model[df_model['testset_name'] == proj]
            if df_proj.empty:
                continue

            y_true = df_proj['ground_truth'].values
            y_scores = df_proj['confidence'].astype(float).values

            if PRAUC_MACRO_AVERAGE:
                recall, precision, lower, upper = bootstrap_macro_pr_curve(y_true, y_scores)
                auc_val = auc(recall, precision)
                ci_lower, ci_upper = calculate_ci_auc_of_macro_pr_curve(y_true, y_scores)
            else:
                recall, precision, lower, upper = bootstrap_pr_curve(y_true, y_scores)
                auc_val = auc(recall, precision)
                ci_lower, ci_upper = calculate_prauc_ci(y_true, y_scores)

            label = projections_dict.get(proj, proj)
            color = colors.get(proj, 'black')

            ax.plot(recall, precision, label=f"{label} (AUC {auc_val:.3f})", color=color, linewidth=2)
            ax.fill_between(recall, lower, upper, color=color, alpha=0.2)

            auc_data.append({
                "EfficientNet": model_label,
                "Testset": label,
                "PRAUC": auc_val,
                "95% CI Lower PRAUC": ci_lower,
                "95% CI Upper PRAUC": ci_upper
            })

        ax.set_title(f"PR Curve - {model_label}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_recall_effnet_proj.png"))
    plt.savefig(os.path.join(output_dir, "precision_recall_effnet_proj.pdf"))
    plt.close()

    return pd.DataFrame(auc_data)
