"""
Convert CVAT 1.1 annotations in ZIP files to YOLO format labels.
Draws sample visualizations for verification.

Author: [Your Name]
Date: [Update Date]
"""

import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import random
import argparse
import cv2


def convert_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    """
    Convert bounding box coordinates from absolute format to YOLO format.

    Parameters:
        xmin, ymin, xmax, ymax (float): Absolute bounding box coordinates.
        img_w, img_h (float): Image dimensions.

    Returns:
        Tuple[float, float, float, float]: Normalized (x_center, y_center, width, height).
    """
    x_center = (xmin + xmax) / 2.0 / img_w
    y_center = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return x_center, y_center, w, h


def find_image_file(img_name, images_dir):
    """
    Locate the actual image file by name (regardless of extension) within the given directory.

    Parameters:
        img_name (str): Image name (may include extension or not).
        images_dir (Path): Directory where images are stored.

    Returns:
        Path: Path to the image file if found.

    Raises:
        FileNotFoundError: If image is not found.
    """
    stem = Path(img_name).stem
    for ext in ['.png', '.jpg', '.jpeg']:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    candidates = list(images_dir.glob(f"{stem}.*"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No image found for {img_name} in directory {images_dir}")


def process_cvat_zips(zips_dir, images_dir, output_dir, target_label="fracture", sample_size=10):
    """
    Convert CVAT annotation ZIP files to YOLO text format.

    Parameters:
        zips_dir (Path): Directory containing ZIP files with annotations.
        images_dir (Path): Directory containing original images.
        output_dir (Path): Output directory for YOLO labels and verification images.
        target_label (str): Label to extract from CVAT XML.
        sample_size (int): Number of samples to draw for visualization.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    yolo_records = []

    for zip_file in zips_dir.glob("*.zip"):
        with zipfile.ZipFile(zip_file, "r") as zf:
            if "annotations.xml" not in zf.namelist():
                continue
            with zf.open("annotations.xml") as xml_file:
                tree = ET.parse(xml_file)
                root = tree.getroot()

                for image_node in root.findall("image"):
                    img_name = image_node.get("name", "")
                    if "_ELB" not in img_name:
                        continue

                    img_w = float(image_node.get("width", "1") or 1)
                    img_h = float(image_node.get("height", "1") or 1)

                    boxes_data = []
                    for box in image_node.findall("box"):
                        if box.get("label") == target_label:
                            xtl = float(box.get("xtl", "0"))
                            ytl = float(box.get("ytl", "0"))
                            xbr = float(box.get("xbr", "0"))
                            ybr = float(box.get("ybr", "0"))
                            x_c, y_c, w, h = convert_box_to_yolo(xtl, ytl, xbr, ybr, img_w, img_h)
                            boxes_data.append(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

                    if boxes_data:
                        stem = Path(img_name).stem
                        out_txt_path = output_dir / f"{stem}.txt"
                        with open(out_txt_path, "w") as txt_f:
                            txt_f.write("\n".join(boxes_data))
                        yolo_records.append((img_name, boxes_data))

    verify_dir = output_dir / "verify"
    verify_dir.mkdir(exist_ok=True)
    sample = random.sample(yolo_records, min(sample_size, len(yolo_records)))

    for img_name, boxes_data in sample:
        img_file = find_image_file(img_name, images_dir)
        img = cv2.imread(str(img_file))
        if img is None:
            raise ValueError(f"Failed to load image: {img_file}")
        h, w = img.shape[:2]

        for line in boxes_data:
            _, x_c, y_c, ww, hh = line.split()
            x_c, y_c, ww, hh = map(float, (x_c, y_c, ww, hh))
            xmin = int((x_c - ww / 2) * w)
            ymin = int((y_c - hh / 2) * h)
            xmax = int((x_c + ww / 2) * w)
            ymax = int((y_c + hh / 2) * h)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        out_jpg_path = verify_dir / f"{Path(img_name).stem}_verify.png"
        cv2.imwrite(str(out_jpg_path), img)


def main():
    parser = argparse.ArgumentParser(
        description="Convert CVAT XML annotations (in .zip) to YOLO format with optional sample visualization."
    )
    parser.add_argument("--zips_dir", required=True, type=Path, help="Directory with zipped CVAT annotation files")
    parser.add_argument("--images_dir", required=True, type=Path, help="Directory with original images")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory to store YOLO labels and visualizations")
    parser.add_argument("--target_label", default="fracture", help="Label name to extract and convert (default: fracture)")
    parser.add_argument("--sample_size", type=int, default=10, help="Number of images to visualize with drawn boxes (default: 10)")
    args = parser.parse_args()

    process_cvat_zips(
        zips_dir=args.zips_dir,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        target_label=args.target_label,
        sample_size=args.sample_size
    )


if __name__ == "__main__":
    main()
