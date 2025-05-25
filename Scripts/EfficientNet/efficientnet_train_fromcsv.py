import torch
from fastai.vision.all import *
from efficientnet_pytorch import EfficientNet
from pathlib import Path
import timm
import os
import csv
import pandas as pd
import argparse

def find_delimiter(file):
    """
    Detects the delimiter used in a CSV file.

    Parameters:
        file (Path): Path to the CSV file.

    Returns:
        str: Detected delimiter character.
    """
    sniffer = csv.Sniffer()
    with open(file) as fp:
        delimiter = sniffer.sniff(fp.read(4096)).delimiter
    return delimiter

def main(args):
    """
    Main function to train EfficientNet models on image data specified in an Excel sheet.
    Converts Excel to CSV, creates data loaders, applies augmentation, and trains models.
    """
    img_path = Path(args.img_path)
    output_path = Path(args.output_path)
    cuda_device = args.cuda_devices
    model_names = args.model_names
    epochs = args.epochs
    validation_percentage = args.valid_pct
    xlsx_path = [Path(args.xlsx_file), args.sheet_name, args.file_stem_col, args.file_suffix, args.set_col, args.label_col]

    df = pd.read_excel(xlsx_path[0], sheet_name=xlsx_path[1])
    df[xlsx_path[2]] = df[xlsx_path[2]].astype(str) + str(xlsx_path[3])
    filtered_df = df[df[xlsx_path[4]].isin(['train', 'val'])].dropna()
    filtered_df = filtered_df[[xlsx_path[2], xlsx_path[5]]]
    output_csv_path = xlsx_path[0].with_suffix('.csv')
    filtered_df.to_csv(output_csv_path, index=False)
    print(f'Data filtered and saved to {output_csv_path}')

    efficientnet_parameters = {
        'efficientnet-b0': [224, 192, 8],
        'efficientnet-b1': [240, 96, 8],
        'efficientnet-b2': [260, 72, 8],
        'efficientnet-b3': [300, 64, 8],
        'efficientnet-b4': [380, 24, 8],
        'efficientnet-b5': [456, 12, 4],
        'efficientnet-b6': [528, 8, 4],
        'efficientnet-b7': [600, 4, 4]
    }

    csv_delim = find_delimiter(output_csv_path)

    for item in model_names:
        try:
            torch.cuda.empty_cache()
            save_folder = output_path / str(item)
            save_pkl = save_folder / (str(item)+'.pkl')
            save_pth = save_folder / (str(item)+'.pth')

            effnet_input, effnet_batch, effnet_workers = efficientnet_parameters[item]

            data = ImageDataLoaders.from_csv(
                path=img_path,
                csv_fname=output_csv_path,
                delimiter=csv_delim,
                folder='',
                header='infer',
                fn_col=0,
                label_col=1,
                num_workers=effnet_workers,
                bs=effnet_batch,
                valid_pct=validation_percentage,
                item_tfms=[Resize(effnet_input)],
                batch_tfms=[*aug_transforms()]
            )

            model = EfficientNet.from_pretrained(item, num_classes=data.c)
            learn = Learner(data, model, metrics=[accuracy])

            if torch.cuda.device_count() > 1 and len(cuda_device) > 1:
                learn.to_distributed(0)
            if len(cuda_device) == 1:
                torch.cuda.set_device(cuda_device[0])

            learn.fine_tune(epochs)

            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            torch.save(model.state_dict(), save_pth)
            learn.export(save_pkl)

            print(f'Training and saving successful for {item}')
        except Exception as e:
            print(f'Error with model {item}: {e}')
            continue

    if output_csv_path.exists():
        output_csv_path.unlink()
        print(f"Temporary CSV {output_csv_path} removed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet models on image data from Excel annotations.")
    parser.add_argument('--img_path', required=True, help='Path to the image directory')
    parser.add_argument('--output_path', required=True, help='Directory to save model outputs')
    parser.add_argument('--cuda_devices', nargs='+', type=int, default=[0], help='List of CUDA device IDs to use')
    parser.add_argument('--model_names', nargs='+', required=True, help='List of EfficientNet model variants to train')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--valid_pct', type=float, default=0.1, help='Validation split percentage')
    parser.add_argument('--xlsx_file', required=True, help='Path to Excel file with image annotations')
    parser.add_argument('--sheet_name', required=True, help='Sheet name in the Excel file')
    parser.add_argument('--file_stem_col', required=True, help='Column with file stem names')
    parser.add_argument('--file_suffix', required=True, help='File extension to append to stem names')
    parser.add_argument('--set_col', required=True, help='Column indicating train/val split')
    parser.add_argument('--label_col', required=True, help='Column with class labels')

    args = parser.parse_args()
    main(args)
