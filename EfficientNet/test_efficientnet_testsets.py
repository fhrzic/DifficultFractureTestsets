import argparse
import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from fastai.vision.all import *
from skimage import exposure


def resizetosquare(img, size, interpolation):
    """
    Resize an image to a square by padding it, then resize.
    """
    h, w = img.shape[:2]
    c = None if len(img.shape) < 3 else img.shape[2]
    dif = max(h, w)
    x_pos = (dif - w) // 2
    y_pos = (dif - h) // 2

    if c is None:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos + h, x_pos:x_pos + w] = img
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos + h, x_pos:x_pos + w, :] = img

    return cv2.resize(mask, (size, size), interpolation)

def main(args):
    """
    Evaluate multiple EfficientNet models on multiple test sets and export results to an Excel file.
    """
    test_sets = [[Path(p), n] for p, n in zip(args.test_dirs[::2], args.test_dirs[1::2])]
    models = [[Path(p), n, s] for p, n, s in zip(args.model_paths[::3], args.model_paths[1::3], args.model_paths[2::3])]

    xlsx_output = Path(args.xlsx_output)
    equalize = args.equalize
    extract_info = args.extract_info
    file_types = ('.png', '.jpg', '.jpeg')
    cuda_device = args.cuda_device

    file_stem_slices = {
        'patient_hash': [0, 64],
        'region': [82, 85],
        'laterality': [86, 87],
        'projection': [87, 88],
        'gender': [89, 90],
        'age': [91, 95]
    }

    data_dict = {}

    for test_set in test_sets:
        with tempfile.TemporaryDirectory() as temp_dir:
            for glob_path in test_set[0].rglob('*'):
                if glob_path.suffix.lower() in file_types:
                    shutil.copy(glob_path, temp_dir)

            for model_path, model_name, image_pixels in models:
                learn = load_learner(model_path)
                defaults.device = torch.device(f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu')

                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        file_path = Path(root, file)
                        if file_path.suffix.lower() not in file_types:
                            continue
                        try:
                            img = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
                            img = resizetosquare(img, int(image_pixels), cv2.INTER_AREA)
                            img = (img / 255.0).astype(np.float64) if img.dtype == 'uint8' else img

                            if equalize:
                                img = exposure.rescale_intensity(img, in_range=(np.percentile(img, 0.05), np.percentile(img, 99.95)))
                                img = exposure.equalize_adapthist(img)

                            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                            if len(img.shape) == 2:
                                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                            if len(img.shape) == 4:
                                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

                            unique_key = f"{model_name}_{test_set[1]}_{file_path.stem}"

                            try:
                                pred_class, pred_idx, outputs = learn.predict(img)
                                probability = round(outputs[pred_idx].item(), 9)
                                data_dict[unique_key] = {
                                    'model': model_name,
                                    'testset_name': test_set[1],
                                    'file_stem': file_path.stem,
                                    'file_path': file_path.parent,
                                    'folder_name': file_path.parent.name,
                                    'pred_class': pred_class,
                                    'probability': probability
                                }
                            except Exception as ex:
                                print(f"Prediction error: {ex}")
                                data_dict[unique_key] = {
                                    'model': model_name,
                                    'testset_name': test_set[1],
                                    'file_stem': file_path.stem,
                                    'file_path': file_path.parent,
                                    'folder_name': file_path.parent.name,
                                    'pred_class': 'ERROR',
                                    'probability': 'ERROR'
                                }

                        except Exception as ex:
                            print(f"Image processing error: {ex}")

    df = pd.DataFrame.from_dict(data_dict, orient='index')
    df.reset_index(drop=True, inplace=True)

    if extract_info:
        df_sliced = pd.DataFrame({'file_stem': df['file_stem']})
        for key, (start, end) in file_stem_slices.items():
            df_sliced[key] = df['file_stem'].str.slice(start, end)
        df_sliced['age'] = pd.to_numeric(df_sliced['age'].str.replace('-', '.', regex=False))
        df_sliced['projection'] = pd.to_numeric(df_sliced['projection'], errors='coerce').notnull().astype(int)
        df = pd.concat([df, df_sliced.drop(columns='file_stem')], axis=1)

    try:
        writer = pd.ExcelWriter(xlsx_output, engine="xlsxwriter")
        df.to_excel(writer, sheet_name="Sheet1", index=False)
        workbook = writer.book
        worksheet = writer.sheets["Sheet1"]
        (max_row, max_col) = df.shape
        column_settings = [{"header": col} for col in df.columns]
        worksheet.add_table(0, 0, max_row, max_col - 1, {"columns": column_settings})
        worksheet.set_column(0, max_col - 1, 12)
        writer.close()
        print(f'Excel file saved: {xlsx_output}')
    except Exception as ex:
        print(f"Error writing Excel: {ex}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate EfficientNet models on test sets and export results to Excel.")
    parser.add_argument('--test_dirs', nargs='+', required=True, help='List of test set directories and names, e.g., path1 name1 path2 name2')
    parser.add_argument('--model_paths', nargs='+', required=True, help='List of model .pkl paths, names, and input sizes: path name size')
    parser.add_argument('--xlsx_output', required=True, help='Path to the output Excel file')
    parser.add_argument('--cuda_device', type=int, default=0, help='CUDA device index (default: 0)')
    parser.add_argument('--equalize', action='store_true', help='Apply histogram equalization to images')
    parser.add_argument('--extract_info', action='store_true', help='Extract metadata from filenames')
    args = parser.parse_args()
    main(args)
