# FracturesHardCases

Repository providing code for the study performed in paper <b>"Artificial Intelligence Test Set Performance in Difficult Cases Matched with Routine Cases of Pediatric Appendicular Skeleton Fractures"</b>.

### Repository organisation

Repository is divided in several moduls where each module is impleneted in its own script inside designated directory. The idea is that code can be used separately based on ones needs. Here is the overview of repository composition and available scripts:

```
FracturesHardCases:
    --> DataFiltering:
        --> CaseMatching.py
    --> YOLO:
        --> CVAT_to_YOLO.py
        --> train_YOLO.py
        --> test_YOLO.py
    --> EfficientNet:
        --> train_efficientnet_fromcsv.py
        -->
        -->
```

## CaseMatching.py

Script which can be found in <<span style="color:green">DataFiltering</span> and is internal script to filter necssary files into valid cases. It simply merges two major dataframes <b>df1</b> and <b>df2</b> into one based on the keys and similarity score. This scrip is created to solve compatibility issues.

In order to run the script, it is possible to do so from comand line interfece by prividing the <b>".json"</b> config file with the following content:
<details>
<ul>
<li><b>spreadsheet</b> --  Full path to the Excel or CSV file containing the dataset.</li>
<li><b>sheet</b> -- Name of the sheet to be read (used only for Excel files).</li>
<li><b>columns</b> -- List of columns to load from the spreadsheet.</li>
<li><b>csv_delimiter</b> -- Delimiter used if the file is in CSV format (default is comma).</li>
<li><b>match_column</b> -- Column used as the identifier for matching (must exist in both filtered DataFrames).</li>
<li><b>match_paramerers</b> -- List of parameters (column names) used to compute similarity between rows.</li>
<li><b>match_threshold</b> -- Minimum similarity score (0–1) required to consider two rows as a match.</li>
<li><b>filters_df1</b> -- Filters to apply to the first copy of the DataFrame (e.g., keep only rows where test300 == 1.0).</li>
<li><b>filters_df2</b> -- Filters to apply to the second copy of the DataFrame (e.g., keep only rows where rel_pathol == 0.0 and include == 1.0).
</li>
<li><b>outputfile</b> -- Full path to the Excel file where the matched results will be saved.</li>
<li><b>verbose</b> -- Whether to print the resulting matched DataFrame to the console (true = print, false = silent).
}</li>
</ul>
</details>
<hr>
Based on the provided config file the following functions enabels mergining, filtering and creation of the new spreadsheet with cases to use in the study:
<details>
<ul>
<hr>
<li><b>main</b> -- The core function of the script. Executes the full matching process:
    <ol>
    <li>Loads the dataset.</li>
    <li>Applies filtering rules to produce two DataFrames.</li>
    <li>Computes similarity scores between rows.</li>
    <li>Selects best matches based on a threshold.</li>
    <li>Saves the final results to an Excel file.</li>
    </ol>
    <b>Parameters</b>:
        <ol>
        <li><b>config_path</b> -- Path to the JSON configuration file specifying all operational parameters.</li>
        </ol>
    <b>Side effects</b>:
    <ol>
    <li>Writes the result to an output Excel file.</li>
    <li>Prints results to console if verbose is enabled.</li>
    </ol>
</li>
<hr>
<li><b>read_spreadsheet_to_df</b> -- Reads data from an Excel or CSV file into a pandas DataFrame, selecting only the specified columns.
<b>Parameters</b>:
    <ol>
    <li><b>xlsx</b>-- Path to the input file (either .xlsx or .csv).</li>
    <li><b>sheet</b>-- Sheet name to read (only used if Excel file).</li>
    <li><b>cols</b>-- List of column names to import.</li>
    <li><b>csv_delimiter</b> -- Delimiter character for CSV files (default is comma).</li>
    </ol>
<b>Returns</b>:
    <ol>
    <li>A DataFrame with the specified columns and normalized column names (lowercase).</li>
    </ol>
</li>
<hr>
<li><b>similarity</b> -- Computes a similarity score between two values. Supports both strings and numbers.

<b>Parameters</b>:
    <ol>
    <li><b>param1, param2</b> -- The two values to compare (either string or numeric).</li>
    </ol>
<b>Returns</b>:
    <ol>
    <li>A float score between 0 and 1, where 1 indicates a perfect match.</li>
    </ol>
<hr>
</li>

<li><b>similarity_score</b> -- Calculates the average similarity score across multiple columns (features) for two rows.
<b>Parameters</b>:
    <ol>
    <li><b>row1, row2</b> -- pandas Series objects representing rows to compare.</li>
    <li><b>parameters</b> -- List of column names to be used in the similarity computation.</li>
    </ol>

<b>Returns</b>:
    <ol>
    <li>A float representing the mean similarity across all selected parameters.</li>
    </ol>
<hr>
</li>
</ul>
</details>
<hr>

## YOLO

YOLO directory contains all scripts necessary for YOLO method training along with its dedicated data preprocessing and testing. YOLOv8 was used in this paper because at the given time when the experiments were conducted, this version was the most recent/stable one. In following paragraphs every script regarding YOLO will be presented:

### CVAT_to_YOLO.py

This tool converts annotations from CVAT 1.1 XML format (inside .zip archives) to <b>YOLO-compatible</b> .txt label files. It also optionally draws bounding boxes on a random sample of images for quick visual verification.

The arguments to run the script are as follows:
<details>
<ul>
<li><b>zips_dir</b> -- Directory containing .zip files exported from CVAT.</li>
<li><b>images_dir</b> -- Directory containing the corresponding original images</li>
<li><b>output_dir</b> -- Directory where YOLO .txt files and verification images will be saved.</li>
<li><b>target_label</b> -- Label name to extract from CVAT annotations (default: fracture).</li>
<li><b>sample_size</b> -- Number of random images to generate with drawn bounding boxes (default: 10).</li>
</ul>
</details>

<hr>

Implemented functions inside the script are as follows:
<details>
<ul>
<li><b>main</b> -- Parses command-line arguments using argparse, then calls process_cvat_zips() with the provided input/output paths and parameters.
</li>
<hr>
<li><b>process_cvat_zips(zips_dir, images_dir, output_dir, target_label="fracture", sample_size=10)</b> -- The core function which solves the following tasks:
<ol>
<li>Parses all <i>.zip</i> files in zips_dir that contain <i>annotations.xml.</i></li>
<li>Extracts bounding boxes with <i>label == target_label.</i></li>
<li>Converts them to <i>YOLO format</i> and writes <i>.txt</i> files to <b>output_dir.</b></li>
<li>Optionally draws bounding boxes on a random subset of images (<i>sample_size</i>) and saves them for verification.</li>
</ol>

<b>Parameters</b>:
<ol>
<li><b>zips_dir</b> -- Directory containing ZIP files from CVAT.</li>
<li><b>images_dir</b> -- Directory containing original image files.</li>
<li><b>output_dir</b> -- Where to save the YOLO labels and sample images.</li>
<li><b>target_label</b> -- The class label to convert (default: <i>"fracture"</i>).</li>
<li><b>sample_size</b> -- Number of random images to use for box visualization.</li>
</ol>
<hr>
<li><b>find_image_file(img_name, images_dir)</b> -- Finds the corresponding image file in the given directory <i>images_dir</i>, using the base name (<i>stem</i>) of the image filename (ignores the extension).

<b>Returns</b>:
<ol>
<li>A <b>Path</b> to the image file if found.</li>
</ol> 
<hr>
<li><b>convert_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)</b> -- Converts bounding box coordinates from absolute pixel values <i>(xmin, ymin, xmax, ymax)</i> to YOLO format: <i>(x_center, y_center, width, height) — all normalized to [0, 1] using the image's width and height.</i>

<b>Returns</b>:
<ol>
<li>A tuple of four floats (YOLO-format coordinates).</li>
</ol>
</li>
<hr>
</li>
</ul>
</details>
<hr>

### train_YOLO.py
This script trains a YOLO model using the Ultralytics YOLOv8 framework. Specifically it:

<ul>
<li>Sets up a temporary training directory.</li>
<li>Changes working directory so YOLO can write output there.</li>
<li>Loads a pretrained YOLO model (yolo8x.pt).</li>
<li>Trains the model on a dataset defined by a meta.yaml file.</li>
<li>Exports the trained model to ONNX format for deployment.</li>
</ul>

<b>Parameters</b>:
<details>
<ol>
  <li><b>temp_dir</b> -- Path to a temporary directory where training outputs (runs, weights, logs) will be saved. <b>(Required)</b></li>
  <li><b>data_yaml</b> -- Path to the YOLO dataset YAML file (e.g., <b>meta.yaml</b>) that defines your training/validation sets and class names. <b>(Required)</b></li>
  <li><b>model_weights</b> -- Path to the pretrained YOLO model weights to start from (default: <b>yolo8x.pt</b>).</li>
  <li><b>epochs</b> -- Number of training epochs (default: <b>100</b>).</li>
  <li><b>imgsz</b> -- Input image size for training (default: <b>640</b>).</li>
  <li><b>batch</b> -- Batch size for training (default: <b>32</b>).</li>
  <li><b>device</b> -- List of GPU device IDs to use (e.g., <b>0</b>, <b>0 1</b>) (default: <b>[0]</b>).</li>
  <li><b>workers</b> -- Number of worker processes for data loading (default: <b>2</b>).</li>
</ol>
</details>
<hr>

### test_YOLO.py

This script evaluates several pretrained YOLOv8 models <b>(n, s, m, l, x variants)</b> on two different test sets using the <i>Ultralytics YOLOv8 framework</i>. It:
<ol>
<li>Loads the model weights for each variant.</li>
<li>Runs .val() on two test sets (a "difficult" and a "random" one).</li>
<li>Saves evaluation metrics, plots, and COCO-style JSON outputs to a specified results directory.</li>
</ol>
    
<b>Parameters</b>:
<details>
<ol>
  <li><b>output_dir</b> -- Directory where evaluation results, plots, and COCO-style JSON files will be saved. <b>(Required)</b>.</li>
  <li><b>gpus</b> -- List of GPU device IDs to use for inference (e.g., <b>0</b> or <b>0 1</b>). Default: <b>[0]</b>.</li>
</ol>
</details>
<hr>

## EfficientNet

EfficientNet directory hold all necessary scripts for EfficientNet training and evaluation. 

### train_efficientnet_fromcsv.py

This script trains one or more <b>EfficientNet models</b> using image labels and metadata provided in an Excel spreadsheet. It converts the Excel file to a CSV, applies data augmentations, and trains each model using the <b>FastAI</b> and <b>efficientnet-pytorch</b> libraries. Multi-GPU training is supported through <b>Distributed Data Parallel (DDP)</b>.

The arguments to run the script are as follows:
<details> 
<ul> 
<li><b>img_path</b> -- Path to the directory containing input images.</li> 
<li><b>output_path</b> -- Directory where trained model outputs (.pth and .pkl) will be saved.</li> 
<li><b>cuda_devices</b> -- List of CUDA device IDs to use (e.g. 0 1).</li> 
<li><b>model_names</b> -- List of EfficientNet model variants to train (e.g. efficientnet-b4 efficientnet-b5).</li> <li><b>epochs</b> -- Number of training epochs (default: 100).</li> 
<li><b>valid_pct</b> -- Validation split percentage (default: 0.1).</li> 
<li><b>xlsx_file</b> -- Path to the Excel file containing metadata and labels.</li> 
<li><b>sheet_name</b> -- Sheet name in the Excel file (e.g. Sheet1).</li> 
<li><b>file_stem_col</b> -- Column name that contains image filename stems.</li> 
<li><b>file_suffix</b> -- File extension to append to each image stem (e.g. .png).</li> 
<li><b>set_col</b> -- Column name indicating the train/val split (e.g. train_val).</li> 
<li><b>label_col</b> -- Column name containing class labels.</li> 
</ul> 
</details> <hr>

Implemented functions inside the script are as follows:
<details> 
<ul> 
<li><b>main(args)</b> -- Parses CLI arguments and performs the entire pipeline: 
<ol> 
<li>Reads and filters Excel metadata.</li> 
<li>Converts metadata to CSV format.</li> 
li>Iterates through a list of EfficientNet models and trains each using FastAI.</li> 
<li>Saves both the model weights (.pth) and FastAI learner (.pkl).</li> 
<li>Removes temporary CSV after training.</li> 
</ol>

<b>Parameters</b>: All values are read from the command line using <code>argparse</code>.
<hr> <li><b>find_delimiter(file)</b> -- Detects the delimiter used in a CSV file (e.g. comma, tab).

<b>Returns</b>:
<ol> 
<li>A <b>str</b> representing the detected delimiter.</li> 
</ol> 
</li> 
</ul> 
</details> <hr>

    
### efficientnet_test_eval.py

Certainly! Here's a `README` section for the `efficientnet_test_eval.py` script, written in the same format as your `CVAT_to_YOLO.py` description:

---

### efficientnet_test_eval.py

This tool evaluates multiple <b>EfficientNet models</b> on defined test sets. It reads model predictions from `.pkl` files, processes the input images (optionally equalizing them), runs inference, and saves the results to a well-formatted <b>Excel (.xlsx)</b> file. Optional metadata can also be extracted directly from filenames.

The arguments to run the script are as follows:

<details>
<ul>
<li><b>test_dirs</b> -- A flat list of test set directory and name pairs. For example: <br>
<code>--test_dirs /path/to/testset1 difficult /path/to/testset2 random</code></li>

<li><b>model_paths</b> -- A flat list of model info triples: path to <b>.pkl</b>, model name, and input size (e.g., 224, 380). For example: <br>
<code>--model_paths /path/to/model_b0.pkl efficientnet-b0 224</code></li>

<li><b>xlsx_output</b> -- Path to the Excel file where results will be saved.</li>
<li><b>cuda_device</b> -- Index of the CUDA device to use (default: 0).</li>
<li><b>equalize</b> -- Optional flag to apply histogram equalization to each input image.</li>
<li><b>extract_info</b> -- Optional flag to extract structured metadata from file names (e.g. age, gender, etc.).</li>
</ul>
</details>

<hr>

Implemented functions inside the script are as follows:

<details>
<ul>

<li><b>main(args)</b> — The main function that:
<ol>
<li>Parses CLI arguments using <code>argparse</code>.</li>
<li>Iterates through test sets and models, loading images and running predictions.</li>
<li>Formats results into a Pandas DataFrame.</li>
<li>Optionally extracts metadata from filenames.</li>
<li>Exports final results to an Excel spreadsheet.</li>
</ol>
</li>

<hr>

<li><b>resizetosquare(img, size, interpolation)</b> — Pads and resizes an image to be square.

<b>Parameters:</b>

<ol>
<li><b>img</b> — NumPy image array.</li>
<li><b>size</b> — Target image size in pixels.</li>
<li><b>interpolation</b> — OpenCV interpolation method (e.g., <i>cv2.INTER_AREA</i>).</li>
</ol>

<b>Returns:</b>

<ol>
<li>A square NumPy image array of the specified size.</li>
</ol>
</li>

</ul>
</details>

<hr>

### gradcam_efficientnet.py

This tool applies <b>Grad-CAM</b> visualizations to test images using a trained <b>EfficientNet</b> model saved as a FastAI <code>.pkl</code> file. Based on an Excel file containing file stems and metadata, the script locates and processes matching images. For each image, it produces and saves:
<ul>
    <li>A raw Grad-CAM <b>heatmap</b>,</li>
    <li>The <b>original</b> image,</li>
    <li>A <b>superimposed</b> visualization (heatmap + original).</li>
</ul>

The arguments to run the script are as follows:
<details> 
<ul> 
<li><b>model_path</b> -- Path to the trained model (.pkl file).</li> 
<li><b>input_dir</b> -- Directory containing test images.</li> 
<li><b>excel_file</b> -- Excel file containing image filename stems and a split column (e.g. train/val/test).</li> 
<li><b>output_dir</b> -- Directory where Grad-CAM visualizations will be saved.</li> 
</ul> 
</details> 
<hr>

Implemented functions inside the script are as follows:
<details> 
<ul> 
<li><b>main(args)</b> — The main function that: 
<ol> 
<li>Parses CLI arguments using <b>argparse</b>.</li> 
<li>Reads the Excel file and filters for relevant test image file stems.</li> 
<li>Copies all matching image files to a temporary directory for processing.</li> 
<li>Loads a trained EfficientNet model.</li> 
<li>Finds the last model block used for Grad-CAM visualization.</li> 
<li>Applies Grad-CAM to each image and saves the following outputs: 
<ul> 
<li><i>_heatmap</i> -- raw Grad-CAM heatmap</li> 
<li><i>_original</i> -- original image</li> 
<li><i>_superimposed</i> -- heatmap overlayed on the original image</li> 
</ul> 
</li> 
</ol> 
</li> 
<hr> 
<li><b>apply_gradcam(learn, img_path, layer_name)</b> -- Applies Grad-CAM to a single image.

<b>Parameters:</b>
<ol> 
<li><b>learn</b> -- FastAI <b>Learner</b> object with a trained model.</li> 
<li><b>img_path</b> -- Path to the image file.</li> 
<li><b>layer_name</b> -- Name of the model layer used for Grad-CAM visualization.</li> </ol>

<b>Returns:</b>
<ol> 
<li><b>heatmap_colored</b> -- The colorized heatmap as a NumPy array.</li> 
<li><b>img</b> -- The original image loaded via OpenCV.</li> 
<li><b>superimposed_img</b> -- Heatmap blended with the original image.</li> 
</ol> </li> 
<hr>
</ul> 
</details> 
<hr>



### Efficientnet_eval_report.py

This tool evaluates predictions from multiple <b>EfficientNet</b> models on classification test sets. It computes classification metrics, generates <b>confusion matrices</b>, and exports results into a detailed <b>Excel summary</b>. Additionally, it produces:

<ul> 
<li>PDF visualizations of individual confusion matrices for each model and test set</li> 
<li>A combined grid of all confusion matrices</li> 
<li>An Excel-based <b>pivot table</b> summarizing model performance</li> 
</ul>

The arguments to run the script are as follows:
<details> 
<ul> 
<li><b>xlsx_path</b> -- Path to the Excel file containing prediction results and metadata.</li> 
<li><b>ground_truth_path</b> -- Path to a second Excel file with ground-truth labels (used if <b>ground_truth</b> column is missing from <b>xlsx_path</b>).</li> <li><b>pdf_dir</b> -- Directory where confusion matrix PDF visualizations will be saved.</li> 
<li><b>output_xlsx_dir</b> -- Directory where the results Excel file and pivot table will be saved.</li> 
</ul> 
</details> 
<hr>

Implemented functions inside the script are as follows:
<details> 
<ul> 
<li><b>generate_report(xlsx_path, ground_truth_path, pdf_dir, output_xlsx_dir)</b> -- The main function that: 
<ol> 
<li>Loads predictions and optionally merges ground-truth labels if missing.</li> 
<li>Calculates metrics using <b>classification_report</b> and <b>accuracy_score</b>.</li> 
<li>Saves per-model confusion matrices as individual PDFs.</li> 
<li>Generates a combined grid of confusion matrices (up to 9 per page).</li> 
<li>Writes a full metrics table to <b>classification_results.xlsx</b>.</li> 
<li>Creates a pivot table summarizing model performance by test set.</li> 
</ol> 
</li> 
<hr> 
<li><b>classification_report</b> / <b>confusion_matrix</b> (from <b>sklearn.metrics</b>) -- Used to: 
<ol> <li>Compute macro- and weighted-averaged precision, recall, F1-scores</li>
<li>Generate normalized confusion matrices for visualization</li> 
</ol> 
</li> 
</ul> 
</details> 
<hr>