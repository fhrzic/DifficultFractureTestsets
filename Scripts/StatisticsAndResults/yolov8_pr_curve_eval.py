import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import precision_recall_curve, auc
from PIL import Image
import argparse

# Set font style
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
FONT_SIZE = 23

# Define utility functions
def compute_iou(box1, box2):
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def load_preds(file):
    """Load prediction JSON file."""
    with open(file) as json_data:
        return json.load(json_data)

def convert_to_coco(bbox, width, height):
    """Convert YOLO bbox format to COCO format."""
    x_center, y_center, bbox_width, bbox_height = bbox
    x = (x_center - bbox_width / 2) * width
    y = (y_center - bbox_height / 2) * height
    w = bbox_width * width
    h = bbox_height * height
    return [x, y, w, h]

def get_iou(pred, data_root):
    """Calculate maximum IoU between prediction and ground truth boxes."""
    max_iou = 0.0
    img_filename = pred["image_id"]
    img_path = os.path.join(data_root, "images", "test", f"{img_filename}.png")
    lab_path = os.path.join(data_root, "labels", "test", f"{img_filename}.txt")
    with Image.open(img_path) as img:
        width, height = img.size
    with open(lab_path) as f:
        for line in f:
            parts = list(map(float, line.split(" ")))
            if int(pred["category_id"]) != int(parts[0] + 1):
                continue
            gt_bb = convert_to_coco(parts[1:], width, height)
            iou = compute_iou(pred["bbox"], gt_bb)
            max_iou = max(max_iou, iou)
    return max_iou

def get_pr_curve(preds, data_root, threshold=0.5):
    """Generate binary labels and confidence scores for PR curve calculation."""
    y_true = [get_iou(pred, data_root) >= threshold for pred in preds]
    confs = [pred["score"] for pred in preds]
    return y_true, confs

def calculate_auc_ci(y_true, y_scores, n_bootstraps=1000, alpha=0.95):
    """Calculate confidence interval for PR AUC using bootstrapping."""
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    rng = np.random.RandomState(42)
    bootstrapped_scores = []
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_scores), len(y_scores))
        if len(np.unique(y_true[indices])) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true[indices], y_scores[indices])
        bootstrapped_scores.append(auc(recall, precision))
    if not bootstrapped_scores:
        return None, None
    sorted_scores = np.sort(bootstrapped_scores)
    lower_idx = int((1.0 - alpha) / 2.0 * len(sorted_scores))
    upper_idx = int((1.0 + alpha) / 2.0 * len(sorted_scores))
    return sorted_scores[lower_idx], sorted_scores[upper_idx]

def bootstrap_pr_curve(y_true, y_scores, n_bootstraps=128, alpha=0.95):
    """Generate PR curve confidence bands using bootstrapping."""
    rng = np.random.RandomState(42)
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    curves = []
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_scores), len(y_scores))
        if len(np.unique(y_true[indices])) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true[indices], y_scores[indices])
        curve = np.interp(np.linspace(0, 1, 1000), recall[::-1], precision[::-1])
        curves.append(curve)
    mean = np.mean(curves, axis=0)
    lower = np.percentile(curves, (1 - alpha) / 2 * 100, axis=0)
    upper = np.percentile(curves, (1 + alpha) / 2 * 100, axis=0)
    return np.linspace(0, 1, 1000), mean, lower, upper

def main(args):
    model_names = ["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x"]
    model_paths = [
        ["786_diff_yolov8n", "786_matched_yolov8n"],
        ["786_diff_yolov8s", "786_matched_yolov8s"],
        ["786_diff_yolov8m", "786_matched_yolov8m"],
        ["786_diff_yolov8l", "786_matched_yolov8l"],
        ["786_diff_yolov8x", "786_matched_yolov8x"]
    ]
    data_paths = args.data_paths
    output_folder = Path(args.output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)
    metrics_output = output_folder / "metrics_output.xlsx"
    file_formats = ["pdf", "png"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    metrics = []

    for i, (name, paths) in enumerate(zip(model_names, model_paths)):
        for j, (path, label, color) in enumerate(zip(paths, ["difficult", "matched"], ["tomato", "forestgreen"])):
            preds = load_preds(os.path.join(path, "predictions.json"))
            y_true, conf = get_pr_curve(preds, data_paths[j])
            sorted_idx = np.argsort(-np.array(conf))
            y_true = np.array(y_true)[sorted_idx]
            conf = np.array(conf)[sorted_idx]
            precision, recall, _ = precision_recall_curve(y_true, conf)
            pr_auc = auc(recall, precision)
            lower_ci, upper_ci = calculate_auc_ci(y_true, conf)
            mean_recall, mean_precision, lower, upper = bootstrap_pr_curve(y_true, conf)
            f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
            best_idx = np.argmax(f1_scores)

            axes[i].plot(recall, precision, label=f"{label} (AUC={pr_auc:.3f})", linewidth=3, color=color)
            axes[i].fill_between(mean_recall, lower, upper, color=color, alpha=0.15)
            metrics.append({
                "Model": name,
                "Dataset": label,
                "AUC": pr_auc,
                "F1": f1_scores[best_idx],
                "Precision": precision[best_idx],
                "Recall": recall[best_idx]
            })

        axes[i].set_title(name)
        axes[i].set_xlim(0, 1)
        axes[i].set_ylim(0, 1)
        axes[i].set_xlabel("Recall")
        axes[i].set_ylabel("Precision")
        axes[i].legend()

    axes[-1].set_title("Combined PR Curves")
    for i, (name, paths) in enumerate(zip(model_names, model_paths)):
        for j, path in enumerate(paths):
            preds = load_preds(os.path.join(path, "predictions.json"))
            y_true, conf = get_pr_curve(preds, data_paths[j])
            sorted_idx = np.argsort(-np.array(conf))
            y_true = np.array(y_true)[sorted_idx]
            conf = np.array(conf)[sorted_idx]
            precision, recall, _ = precision_recall_curve(y_true, conf)
            label = f"{name[-1:]} {'matched' if j==1 else 'difficult'}"
            color = ["darkred", "darkgreen"][j]
            axes[-1].plot(recall, precision, label=label, linewidth=2.5, color=color)

    axes[-1].set_xlim(0, 1)
    axes[-1].set_ylim(0, 1)
    axes[-1].set_xlabel("Recall")
    axes[-1].set_ylabel("Precision")
    axes[-1].legend()
    plt.tight_layout()

    for ext in file_formats:
        plt.savefig(output_folder / f"pr_curve_summary.{ext}", format=ext)

    pd.DataFrame(metrics).to_excel(metrics_output, index=False)
    print(f"Saved results to {metrics_output} and PR curves to {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PR curves for YOLOv8 models and generate metrics.")
    parser.add_argument('--data_paths', nargs=2, required=True, help='Paths to ground truth data: [difficult_path matched_path]')
    parser.add_argument('--output_dir', required=True, help='Directory to save outputs (plots and metrics)')
    args = parser.parse_args()
    main(args)
