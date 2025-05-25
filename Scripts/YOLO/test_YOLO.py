from ultralytics import YOLO
from pathlib import Path
import argparse

def run_evaluations(models, gpus, test_sets, output_dir):
    """
    Evaluates a list of YOLOv8 models on multiple test sets.

    Parameters:
        models (list): List of [weights_path, model_name] pairs.
        gpus (list): List of GPU IDs to use.
        test_sets (list): List of [dataset_yaml, tag] pairs.
        output_dir (Path): Directory to save evaluation results.
    """
    for model_path, model_name in models:
        model = YOLO(model_path)
        for dataset_yaml, tag in test_sets:
            model.val(
                data=dataset_yaml,
                split='test',
                device=gpus,
                project=output_dir,
                name=f'{tag}_{model_name}',
                plots=True,
                save_json=True
            )

def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple YOLOv8 models on test datasets.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save evaluation results.")
    parser.add_argument("--gpus", nargs='+', type=int, default=[0], help="List of GPU device IDs to use.")
    args = parser.parse_args()

    # Define models to evaluate
    models = [
        [Path('/mnt/extended/DEVELOPMENT/DIFFICULT_FX/allregions/runs/detect/yolov8n.pt_640/weights/best.pt'), "yolov8n"],
        [Path('/mnt/extended/DEVELOPMENT/DIFFICULT_FX/allregions/runs/detect/yolov8s.pt_640/weights/best.pt'), "yolov8s"],
        [Path('/mnt/extended/DEVELOPMENT/DIFFICULT_FX/allregions/runs/detect/yolov8m.pt_640/weights/best.pt'), "yolov8m"],
        [Path('/mnt/extended/DEVELOPMENT/DIFFICULT_FX/allregions/runs/detect/yolov8l.pt_640/weights/best.pt'), "yolov8l"],
        [Path('/mnt/extended/DEVELOPMENT/DIFFICULT_FX/allregions/runs/detect/yolov8x.pt_640/weights/best.pt'), "yolov8x"]
    ]

    # Define test datasets
    test_sets = [
        [Path('/mnt/extended/DEVELOPMENT/DIFFICULT_FX/_TESTSETS_FINAL/difficult1017_allregions/difficult/meta.yaml'), '1017_diff'],
        [Path('/mnt/extended/DEVELOPMENT/DIFFICULT_FX/_TESTSETS_FINAL/difficult1017_allregions/random/meta.yaml'), '1017_random']
    ]

    run_evaluations(models, args.gpus, test_sets, args.output_dir)

if __name__ == "__main__":
    main()