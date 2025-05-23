# FracturesHardCases

Repository providing code for the study performed in paper <b>"Artificial Intelligence Test Set Performance in Difficult Cases Matched with Routine Cases of Pediatric Appendicular Skeleton Fractures"</b>.

### Repository organisation

Repository is divided in several moduls where each module is impleneted in its own script inside designated directory. The idea is that code can be used separately based on ones needs. Here is the overview of repository composition and available scripts:

```
FracturesHardCases:
    --> DataFiltering:
        --> CaseMatching.py
    -->
    -->
```

## CaseMatching.py

Script which can be found in <<span style="color:green">DataFiltering</span> and is internal script to filter necssary files into valid cases. It simply merges two major dataframes <b>df1</b> and <b>df2</b> into one based on the keys and similarity score. This scrip is created to solve compatibility issues.

In order to run the script, it is possible to do so from comand line interfece by prividing the <b>".json"</b> config file with the following content:

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
<li><b>read_spreadsheet_to_df</b> --< Reads data from an Excel or CSV file into a pandas DataFrame, selecting only the specified columns.
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
