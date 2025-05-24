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
        -->
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
<ol>
  <li><b>output_dir</b> -- Directory where evaluation results, plots, and COCO-style JSON files will be saved. <b>(Required)</b>.</li>
  <li><b>gpus</b> -- List of GPU device IDs to use for inference (e.g., <b>0</b> or <b>0 1</b>). Default: <b>[0]</b>.</li>
</ol>
<hr>



    

    

    

