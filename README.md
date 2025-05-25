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
        --> efficientnet_train_fromcsv.py
        --> gradcam_efficientnet.py
        --> efficientnet_eval_report.py
        --> efficientnet_test_eval.py
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

### efficientnet_train_fromcsv.py

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


### efficientnet_eval_report.py

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

## StatisticsAndResults

In this direcrtory there are scripts used to evaluate and generate images for statistics.

### yolov8_pr_curve_eval.py

This tool evaluates <b>YOLOv8</b> predictions on binary detection tasks using <b>Precision-Recall (PR) curves</b>. It calculates per-model <b>Average Precision (AP)</b>, <b>F1-scores</b>, and <b>confidence intervals</b> via bootstrapping. Results are visualized and saved to disk in multiple formats.

The script generates:
<ul> 
<li>PDF and PNG PR curve visualizations per model (for both <b>matched</b> and <b>difficult</b> test sets)</li> 
<li>A combined PR plot showing all model curves</li> 
<li>An Excel file summarizing AUC, F1, Precision, and Recall</li> 
</ul>

The arguments to run the script are as follows:
<details> 
<ul> 
<li><b>data_paths</b> -- A list of two paths: <br> <b>[ /path/to/difficult, /path/to/matched ]</b><br> Each path must follow the YOLO directory format with subfolders: <b>images/test/</b> and <b>labels/test/</b>. </li> 
<li><b>output_dir</b> -- Directory where PR curve plots and the Excel metrics summary will be saved.</li> 
</ul> 
</details> 
<hr>

Implemented functions inside the script are as follows:
<details> 
<ul> 
<li><b>main(args)</b> -- The main function that: 
<ol> 
<li>Loads predictions for all 5 YOLOv8 model variants (<i>n, s, m, l, x</i>).</li> 
<li>Computes PR curves for both <b>difficult</b> and <b>matched</b> test sets.</li> 
<li>Bootstraps the PR curves and computes 95% confidence bands.</li> 
<li>Generates per-model PR plots and a combined PR plot.</li> 
<li>Exports all metrics to an Excel file (<i>metrics_output.xlsx</i>).</li> 
</ol> 
</li> 
<hr> 
<li><b>get_pr_curve(preds, data_root, threshold=0.5)</b> -- Computes binary labels and confidence scores: 
<ol> 
<li>Compares predicted boxes with ground-truth labels using IoU ≥ 0.5.</li> 
<li>Returns binary vector of detections and associated confidence scores.</li> 
</ol> 
</li> 
<hr> 
<li><b>bootstrap_pr_curve(y_true, y_scores)</b> -- Estimates confidence intervals on PR curve: 
<ol> 
<li>Resamples input scores using bootstrap sampling.</li> 
<li>Interpolates PR points to create smooth curves with shaded error regions.</li> 
</ol> 
</li> 
<hr> 
<li><b>calculate_auc_ci(y_true, y_scores)</b> -- Bootstraps AUC estimation: 
<ol> 
<li>Calculates mean AUC and lower/upper confidence bounds (default 95%).</li> 
</ol> 
</li> 
</ul> 
</details> 
<hr>

### Scripts in effnet_eval dir
The scripts in this directory enabels creating visualisations and calculation of the metrices in regard to evaluation of efficient net.

#### main.py
This tool evaluates EfficientNet model variants on binary classification tasks by computing ROC AUC and Precision-Recall (PR) AUC curves, including confidence intervals estimated via bootstrapping. It supports both per-model visualization and a combined summary, and exports all results to Excel.

The script generates:
<ul> <li>ROC AUC and PR AUC plots (PDF/PNG) for each EfficientNet variant.</li> <li>Optional 3x3 combined subplot of all models’ curves.</li> <li>An Excel file with average AUCs, F1-scores, precision, and recall, including bootstrap confidence bounds.</li> </ul>

The arguments to run the script are as follows:
<details> 
<ul> 
<li><b>--input_excel</b> -- Path to the Excel file containing model predictions and ground truth labels.</li> 
<li><b>--output_dir</b> -- Directory where plots and Excel summary will be saved.</li> 
<li><b>--macro_pr_auc</b> -- (Optional) If set, uses macro-averaged PR AUC instead of micro.</li> 
<li><b>--combined_subplot</b> -- (Optional) If set, creates a 3×3 subplot of all EfficientNet variants.</li> 
<li><b>--n_bootstraps</b> -- (Optional) Number of bootstrap samples for confidence interval calculation. Default: 256.</li> 
</ul> 
</details> 
<hr>

Implemented functions inside the script are as follows:
<details> 
<ul> 
<li><b>main()</b> -- The main function that: 
<ol> 
<li>Parses CLI arguments and prepares input/output paths.</li> 
<li>Loads prediction results and extracts EfficientNet variants.</li> 
<li>Calls plotting functions to compute and visualize ROC and PR curves per model.</li> 
<li>Bootstraps the AUC metrics for confidence interval estimation.</li> 
<li>Exports ROC and PR results to an Excel file with multiple sheets.</li> 
</ol> </li> 
<hr> 
<li><b>plot_roc_auc_curves(models, part)</b> -- (Imported)<br> Generates ROC AUC curves and confidence intervals for each model. Returns a DataFrame with per-model AUC statistics.</li> 
<hr> 
<li><b>plot_precision_recall_curves(models, part)</b> -- (Imported)<br> Generates PR AUC curves and confidence intervals for each model. Returns a DataFrame with per-model AP/F1 metrics.</li> 
</ul> 
</details> 
<hr>


#### config.py

This configuration module defines global constants and lookup mappings used throughout the EfficientNet evaluation pipeline. It supports consistent labeling, plotting, region classification, and bootstrap evaluation for ROC and PR AUC metrics.

The module provides:
<ul> 
<li>Global evaluation flags (e.g., macro-averaged PR AUC, number of bootstrap iterations).</li> 
<li>Label mappings for anatomical regions and projections.</li> 
<li>Model name aliases for simplified visualization (e.g., “B0”, “B1”, ...).</li> 
<li>Color codes for matched vs. difficult test subsets in plots.</li> 
</ul>

The defined configuration variables are as follows:
<details> 
<ul> 
<li><b>PRAUC_MACRO_AVERAGE</b> -- If <b>True</b>, uses macro-averaged PR AUC (averages over classes). Otherwise, micro-averaged.</li> 
<li><b>COMBINATION_SUBPLOT_MEAN</b> -- If <b>True</b>, generates a 3x3 subplot combining all models in a single figure.</li> 
<li><b>N_BOOTSTRAPS</b> -- Number of bootstrap iterations to estimate AUC confidence intervals. Default: <b>256</b>.</li> 
</ul> 
</details> 
<hr>

Mappings and dictionaries:
<details> 
<ul> 
<li><b>regions_dict</b> -- Maps region abbreviations to full anatomical names and integer IDs used in datasets. <pre> "ANK": ["Ankle", 15], "KNE": ["Knee", 13], ... </pre> </li> 
<hr> 
<li><b>projections_dict</b> -- Maps internal dataset projection types to human-readable names: <pre> "difficult" → "difficult" "matched" → "easy" </pre> </li> <hr> <li><b>efficientnet_dict</b> -- Maps full EfficientNet model names to shorthand labels for visualization: <pre> "efficientnet-b0" → "B0", "efficientnet-b7" → "B7" </pre> </li> <hr> <li><b>colors</b> -- Defines color mapping for test set types in visualizations: <pre> "matched" → forestgreen "difficult" → tomato </pre> </li> 
</ul> 
</details>
<hr>

#### effnet_eval/metrics.py

This module provides core evaluation utilities for binary classification performance analysis using EfficientNet model predictions. It supports generation of ROC and Precision-Recall (PR) curves, as well as bootstrapped confidence intervals for both AUC metrics.

The module is used throughout the evaluation pipeline to generate statistically robust metrics and visualize model performance with confidence bounds.

The script provides:
<ul> 
<li>Bootstrapped confidence intervals for ROC AUC and PR AUC scores.</li> 
<li>Flexible support for macro- or micro-averaged PR AUC (based on configuration).</li> 
<li>Helper functions to generate standard ROC and PR curves.</li> 
</ul>

The following global settings are imported from <b>config.py</b>:
<details> 
<ul> 
<li><b>N_BOOTSTRAPS</b> -- Number of bootstrap samples (default: <b>256</b>).</li> 
<li><b>PRAUC_MACRO_AVERAGE</b> -- If set, uses macro-averaged PR AUC; otherwise, computes AUC from PR curve points.</li> 
</ul> 
</details> 
<hr>

Implemented functions inside the script are as follows:
<details> 
<ul> 
<li><b>calculate_prauc_ci(y_true, y_scores, n_bootstraps, alpha)</b> -- 
<ol> 
<li>Bootstraps precision-recall AUC using resampled scores.</li> 
<li>Returns a tuple: lower and upper confidence bounds.</li> 
<li>Respects <b>PRAUC_MACRO_AVERAGE</b> flag.</li> 
</ol> </li> 
<hr> 
<li><b>calculate_roc_auc_ci(y_true, y_scores, n_bootstraps, alpha)</b> -- 
<ol> 
<li>Bootstraps ROC AUC using standard resampling.</li> 
<li>Returns a tuple: lower and upper confidence bounds.</li> 
</ol> 
</li> 
<hr> 
<li><b>calculate_pr_curve(y_true, y_scores)</b> -- <ol> 
<li>Computes the full precision-recall curve.</li> 
<li>Returns: <b>precision</b>, <b>recall</b>, <b>thresholds</b>, and <b>auc</b>.</li> </ol> </li> <hr> <li><b>calculate_roc_curve(y_true, y_scores)</b> -- <ol> <li>Computes the full ROC curve and AUC.</li> 
<li>Returns: <b>fpr</b>, <b>tpr</b>, <b>thresholds</b>, and <b>auc</b>.</li> </ol> </li> 
</ul> 
</details>
<hr>

#### plotting.py

This module provides all visualization utilities for EfficientNet model evaluation. It generates ROC AUC and Precision-Recall (PR) AUC plots for each model variant across different test sets (e.g., matched, difficult), including bootstrapped confidence intervals. The resulting plots are saved as PNG and PDF files, and a summary table of AUC values with confidence bounds is returned.

The script generates:
<ul> 
<li>ROC AUC plots per model (with shaded confidence bands)</li> 
<li>Precision-Recall plots per model (with shaded confidence bands)</li> 
<li>PDF and PNG visualizations in a 3×3 grid layout</li> 
<li>Summary <b>DataFrame</b> of AUC scores and confidence intervals (for export to Excel)</li> 
</ul>

It uses global settings and dictionaries from the configuration module:

<details> 
<ul> 
<li><b>PRAUC_MACRO_AVERAGE</b> -- If set, uses macro-averaged PR AUC and bootstraps over full curve.</li> 
<li><b>COMBINATION_SUBPLOT_MEAN</b> -- (Imported, unused in this module).</li> 
<li><b>colors</b> -- Used for coloring *matched* and *difficult* test sets.</li> 
<li><b>efficientnet_dict</b> -- Short model names (e.g., “B0” for “efficientnet-b0”).</li> 
<li><b>projections_dict</b> -- Maps test set names for labeling (e.g., “difficult” → “difficult”).</li> 
</ul> 
</details>
<hr>

Implemented functions inside the script:
<details> 
<ul> 
<li><b>plot_roc_auc_curves(df, efficientnets, output_dir)</b> -- <ol> 
<li>Iterates over EfficientNet model variants.</li> 
<li>Filters the DataFrame by model and test set name.</li> 
<li>Computes ROC curves with bootstrapped intervals.</li> 
<li>Generates a 3×3 subplot grid of ROC curves.</li> 
<li>Saves both PDF and PNG plots to <b>output_dir</b>.</li> 
<li>Returns a <b>pandas.DataFrame</b> with AUC and 95% confidence bounds.</li> 
</ol> </li> 
<hr> 
<li><b>plot_precision_recall_curves(df, efficientnets, output_dir)</b> -- <ol> 
<li>Similar to <b>plot_roc_auc_curves</b>, but for PR curves.</li> 
<li>Uses either macro or micro-averaged bootstrapping (configurable).</li> 
<li>Draws shaded confidence bands for precision-recall curve points.</li> 
<li>Returns a <b>pandas.DataFrame</b> with PR AUC and confidence intervals.</li>
</ol> 
</li> 
</ul> 
</details>
<hr>