import pandas as pd
from pathlib import Path
import shutil
import tempfile
import cv2
import numpy as np
from fastai.vision.all import *
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
import re
import argparse

def apply_gradcam(learn, img_path, layer_name):
    """
    Applies Grad-CAM visualization on a given image using a specified model layer.

    Parameters:
        learn (Learner): FastAI learner object.
        img_path (Path): Path to the image.
        layer_name (str): Name of the layer to apply Grad-CAM to.

    Returns:
        heatmap_colored: Colorized heatmap overlay.
        img: Original image.
        superimposed_img: Heatmap superimposed on the original image.
    """
    img = PILImage.create(img_path)
    img_tensor = learn.dls.test_dl([img]).one_batch()[0]
    learn.model.eval()

    def forward_hook(module, input, output):
        learn.activations = output
    def backward_hook(module, grad_in, grad_out):
        learn.gradients = grad_out[0]

    layer = dict([*learn.model.named_modules()])[layer_name]
    handle_forward = layer.register_forward_hook(forward_hook)
    handle_backward = layer.register_backward_hook(backward_hook)

    preds = learn.model(img_tensor)
    preds[0, torch.argmax(preds)].backward()

    pooled_gradients = torch.mean(learn.gradients, dim=[0, 2, 3])
    activations = learn.activations[0]
    for i in range(len(pooled_gradients)):
        activations[i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=0).detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    handle_forward.remove()
    handle_backward.remove()

    img = cv2.imread(str(img_path))
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    superimposed_img = heatmap_colored * 0.33 + img

    return heatmap_colored, img, superimposed_img

def main():
    parser = argparse.ArgumentParser(description="Apply Grad-CAM to test images using a trained EfficientNet model.")
    parser.add_argument('--model_path', required=True, help='Path to the .pkl model file')
    parser.add_argument('--input_dir', required=True, help='Directory containing input image data')
    parser.add_argument('--excel_file', required=True, help='Excel file with image stem metadata')
    parser.add_argument('--output_dir', required=True, help='Directory to save Grad-CAM outputs')
    args = parser.parse_args()

    model_path = Path(args.model_path)
    inputdir = Path(args.input_dir)
    excel_path = Path(args.excel_file)
    columns = ['file_stem', 'folder_name']
    output_dir = Path(args.output_dir)

    df = pd.read_excel(excel_path)
    filtered_df = df[df[columns[1]] == 'test']
    temp_dir = Path(tempfile.mkdtemp())

    for file_stem in filtered_df[columns[0]]:
        for img_path in inputdir.rglob(f'{file_stem}.*'):
            if img_path.suffix.lower() in ['.tif', '.png', '.jpg', '.jpeg']:
                shutil.copy(img_path, temp_dir / img_path.name)

    learn = load_learner(model_path)

    last_layer_name = None
    max_block_num = -1
    for name, module in learn.model.named_modules():
        match = re.match(r'_blocks\.(\d+)', name)
        if match:
            block_num = int(match.group(1))
            if block_num > max_block_num:
                max_block_num = block_num
                last_layer_name = name

    print(f'Last layer name: {last_layer_name}')
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(temp_dir.glob('*'))
    total_images = len([img_path for img_path in image_paths if img_path.suffix.lower() in ['.tif', '.png', '.jpg', '.jpeg']])

    for img_path in tqdm(image_paths, desc="Processing images", unit="image", total=total_images):
        if img_path.suffix.lower() in ['.tif', '.png', '.jpg', '.jpeg']:
            heatmap_img, original_img, superimposed_img = apply_gradcam(learn, img_path, last_layer_name)

            heatmap_output_path = output_dir / f"{img_path.stem}_heatmap{img_path.suffix}"
            cv2.imwrite(str(heatmap_output_path), heatmap_img)

            original_output_path = output_dir / f"{img_path.stem}_original{img_path.suffix}"
            cv2.imwrite(str(original_output_path), original_img)

            superimposed_output_path = output_dir / f"{img_path.stem}_superimposed{img_path.suffix}"
            cv2.imwrite(str(superimposed_output_path), superimposed_img)

    print(f'Saved images to {output_dir}.')

if __name__ == '__main__':
    main()
