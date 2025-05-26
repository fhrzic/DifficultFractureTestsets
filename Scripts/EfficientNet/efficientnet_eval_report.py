import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path
import argparse
import os

# Set global plot settings
plt.rcParams['font.family'] = 'Times New Roman'
FONT_SIZE = 14

def generate_report(xlsx_path, ground_truth_path, pdf_dir, output_xlsx_dir):
    """
    Generate classification metrics and confusion matrices for EfficientNet model predictions.

    Parameters:
        xlsx_path (Path): Path to the Excel file with predictions.
        ground_truth_path (Path): Path to the Excel file containing ground truth (if needed).
        pdf_dir (Path): Directory to save confusion matrix PDFs.
        output_xlsx_dir (Path): Directory to save final results Excel file.
    """
    os.makedirs(pdf_dir, exist_ok=True)
    df_results = pd.read_excel(xlsx_path)

    true_class_col = 'ground_truth'
    pred_class_col = 'pred_class'

    # Add confidence score
    df_results['confidence'] = df_results.apply(
        lambda row: row['probability'] if row[pred_class_col] == 1 else 1 - row['probability'], axis=1
    )

    # Merge ground truth if missing
    if true_class_col not in df_results.columns:
        df_gt = pd.read_excel(ground_truth_path)
        if true_class_col in df_gt.columns:
            df_gt[true_class_col] = df_gt[true_class_col].fillna(0)
            df_results = df_results.merge(df_gt[['file_stem', true_class_col]], on='file_stem', how='left')
        else:
            raise ValueError(f"Ground truth file is missing column: {true_class_col}")

    unique_models = df_results['efficientnet'].unique()
    results_df = pd.DataFrame()
    confusion_matrices = {}

    for testset in df_results['testset_name'].unique():
        subset = df_results[df_results['testset_name'] == testset]
        for model in unique_models:
            model_df = subset[subset['efficientnet'] == model]
            y_true, y_pred = model_df[true_class_col], model_df[pred_class_col]

            try:
                report = classification_report(y_true, y_pred, output_dict=True, zero_division=1)
                acc = accuracy_score(y_true, y_pred)
                cm = confusion_matrix(y_true, y_pred, normalize='true')
                confusion_matrices[f'{testset}_{model}'] = cm

                row = {
                    'Testset': testset,
                    'EfficientNet': model,
                    'Precision (Weighted)': report['weighted avg']['precision'],
                    'Recall (Weighted)': report['weighted avg']['recall'],
                    'F1-Score (Weighted)': report['weighted avg']['f1-score'],
                    'Precision (Macro)': report['macro avg']['precision'],
                    'Recall (Macro)': report['macro avg']['recall'],
                    'F1-Score (Macro)': report['macro avg']['f1-score'],
                    'Accuracy': acc
                }
                results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)

                # Plot individual confusion matrix
                fig, ax = plt.subplots()
                cax = ax.matshow(cm, cmap='Blues', vmin=0, vmax=1)
                for (i, j), val in np.ndenumerate(cm):
                    color = 'white' if val > 0.6 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color)
                ax.set_title(f'{model.upper()} ({testset})')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                plt.colorbar(cax)
                plt.tight_layout()
                plt.savefig(pdf_dir / f'{testset}_{model}_confusion_matrix.pdf')
                plt.close()
            except Exception as e:
                print(f"Error with model {model} on testset {testset}: {e}")

        # Plot all confusion matrices in a single PDF grid
        keys = sorted([k for k in confusion_matrices if k.startswith(testset)])
        fig, axes = plt.subplots(3, 3, figsize=(9, 9))
        axes = axes.flatten()
        for idx, key in enumerate(keys):
            cm = confusion_matrices[key]
            ax = axes[idx]
            ax.matshow(cm, cmap='Blues', vmin=0, vmax=1)
            for (i, j), val in np.ndenumerate(cm):
                color = 'white' if val > 0.55 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=FONT_SIZE)
            ax.set_title(key.split('_')[-1], fontsize=FONT_SIZE+2, fontweight='bold')
            ax.set_xlabel('Predicted', fontsize=FONT_SIZE)
            ax.set_ylabel('True', fontsize=FONT_SIZE)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])

        for idx in range(len(keys), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(pdf_dir / f'{testset}_confusion_matrices.pdf')
        plt.close()

    # Save results to Excel
    output_xlsx_file = output_xlsx_dir / 'classification_results.xlsx'
    results_df.to_excel(output_xlsx_file, index=False)

    # Create and save pivot table
    pivot_table = results_df.pivot_table(
        index='EfficientNet',
        columns='Testset',
        values=['Precision (Weighted)', 'Recall (Weighted)', 'Accuracy',
                'F1-Score (Weighted)', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)'],
        aggfunc='mean')

    with pd.ExcelWriter(output_xlsx_file, engine='openpyxl', mode='a') as writer:
        pivot_table.to_excel(writer, sheet_name='Pivot Table')

    print(f"Saved classification results and pivot table to {output_xlsx_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate classification report and confusion matrices for EfficientNet predictions.")
    parser.add_argument('--xlsx_path', required=True, help='Path to predictions Excel file')
    parser.add_argument('--ground_truth_path', required=True, help='Path to ground truth Excel file')
    parser.add_argument('--pdf_dir', required=True, help='Directory to save PDF plots')
    parser.add_argument('--output_xlsx_dir', required=True, help='Directory to save final Excel summary')
    args = parser.parse_args()

    generate_report(
        Path(args.xlsx_path),
        Path(args.ground_truth_path),
        Path(args.pdf_dir),
        Path(args.output_xlsx_dir)
    )

if __name__ == '__main__':
    main()