import os
import shutil
import random
from pathlib import Path
import yaml
import argparse
from ultralytics import YOLO

def train_yolo(temp_dir, data_yaml, model_weights, epochs, imgsz, batch, device, workers):
    """
    Trains a YOLO model using the specified configuration and exports it to ONNX.

    Parameters:
        temp_dir (Path): Directory to store training outputs.
        data_yaml (Path): Path to the YAML dataset config.
        model_weights (Path): Path to the pretrained YOLO weights (e.g. yolo8x.pt).
        epochs (int): Number of training epochs.
        imgsz (int): Image size to use during training.
        batch (int): Batch size.
        device (list[int]): GPU device IDs (e.g. [0, 1]).
        workers (int): Number of data loading workers.
    """
    temp_dir = Path(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    os.chdir(temp_dir)

    model = YOLO(str(model_weights))
    results = model.train(
        data=str(data_yaml),
        project=str(temp_dir),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers
    )

    model.export(format="onnx")


def main():
    parser = argparse.ArgumentParser(description="Train a YOLO model and export it to ONNX format.")
    parser.add_argument("--temp_dir", type=Path, required=True, help="Temporary directory for training output.")
    parser.add_argument("--data_yaml", type=Path, required=True, help="Path to data YAML file.")
    parser.add_argument("--model_weights", type=Path, default="yolo8x.pt", help="Path to YOLO pretrained weights.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training.")
    parser.add_argument("--batch", type=int, default=32, help="Batch size.")
    parser.add_argument("--device", nargs="+", type=int, default=[0], help="List of GPU device IDs (e.g. 0 1).")
    parser.add_argument("--workers", type=int, default=2, help="Number of data loader workers.")
    args = parser.parse_args()

    train_yolo(
        temp_dir=args.temp_dir,
        data_yaml=args.data_yaml,
        model_weights=args.model_weights,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers
    )

if __name__ == "__main__":
    main()