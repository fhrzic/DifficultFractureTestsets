# Match folders of files based on a set of parameters
# Updated with CLI support and polished documentation
# 2024-03-05

import os
import pathlib
import pandas
import sys
import json
import argparse
from rapidfuzz import fuzz

def read_spreadsheet_to_df(xlsx, sheet, cols, csv_delimiter=','):
    """
    Reads data from an Excel or CSV spreadsheet into a pandas DataFrame.

    Parameters:
        xlsx (Path): Path to the spreadsheet.
        sheet (str): Sheet name for Excel files.
        cols (list): Columns to load.
        csv_delimiter (str): Delimiter used for CSV files.

    Returns:
        pandas.DataFrame: Loaded and normalized DataFrame.
    """
    try:
        if str(xlsx).endswith('.xlsx'):
            df = pandas.read_excel(xlsx, sheet_name=sheet, usecols=cols)
        elif str(xlsx).endswith('.csv'):
            df = pandas.read_csv(xlsx, dtype=str, sep=csv_delimiter,
                                 skipinitialspace=True, usecols=cols)
        else:
            raise ValueError("Unsupported file format. Only .xlsx and .csv are supported.")
        df.columns = map(str.lower, df.columns)
        return df
    except Exception as ex:
        print('ERROR: Failed to load spreadsheet. Check path, sheet name, and columns.')
        print(ex)
        sys.exit(1)

def similarity(param1, param2):
    """
    Computes a similarity score between two parameters.

    For strings: fuzzy matching ratio.
    For numbers: normalized relative difference.

    Parameters:
        param1, param2: Comparable values (str or numeric).

    Returns:
        float: Similarity score between 0 and 1.
    """
    try:
        if isinstance(param1, str) and isinstance(param2, str):
            return fuzz.ratio(param1, param2) / 100
        elif isinstance(param1, (int, float)) and isinstance(param2, (int, float)):
            max_val = max(abs(param1), abs(param2), 1)
            return 1 - abs(param1 - param2) / max_val
    except:
        pass
    return 0.0

def similarity_score(row1, row2, parameters):
    """
    Computes average similarity across multiple parameters for two DataFrame rows.

    Parameters:
        row1, row2 (Series): Rows to compare.
        parameters (list): List of column names to compare.

    Returns:
        float: Mean similarity score.
    """
    return sum(similarity(row1[param], row2[param]) for param in parameters) / len(parameters)

def main(config_path):
    """
    Main execution function.

    Loads data, filters rows, performs matching based on similarity of selected parameters,
    and writes matched results to an Excel file.

    Parameters:
        config_path (str): Path to JSON configuration file.
    """
    # Load configuration
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    # Parse config paths and parameters
    spreadsheet = pathlib.Path(cfg["spreadsheet"])
    sheet = cfg["sheet"]
    columns = cfg["columns"]
    csv_delimiter = cfg.get("csv_delimiter", ',')
    match_column = cfg["match_column"]
    match_params = [p.lower() for p in cfg["match_parameters"]]
    match_threshold = cfg["match_threshold"]
    filters_df1 = {k.lower(): v for k, v in cfg["filters_df1"].items()}
    filters_df2 = {k.lower(): v for k, v in cfg["filters_df2"].items()}
    outputfile = pathlib.Path(cfg["outputfile"])
    verbose = cfg.get("verbose", True)

    # Read and filter data
    df1 = read_spreadsheet_to_df(spreadsheet, sheet, columns, csv_delimiter)
    df2 = read_spreadsheet_to_df(spreadsheet, sheet, columns, csv_delimiter)

    for k, v in filters_df1.items():
        df1 = df1[df1[k] == v]
        df2 = df2[df2[k] != v]

    for k, v in filters_df2.items():
        df2 = df2[df2[k] == v]

    matched = []
    for _, row1 in df1.iterrows():
        best_score = 0
        best_match = None
        best_idx = None

        for idx2, row2 in df2.iterrows():
            score = similarity_score(row1, row2, match_params)
            if score > best_score:
                best_score = score
                if score >= match_threshold:
                    best_match = row2[match_column]
                    best_idx = idx2

        matched.append((row1[match_column], best_match, best_score))
        if best_idx is not None:
            df2 = df2.drop(index=best_idx)

    for _, row1 in df1.iterrows():
        if row1[match_column] not in [m[0] for m in matched]:
            matched.append((row1[match_column], None, None))

    result_df = pandas.DataFrame(matched, columns=[match_column+'_df1', match_column+'_df2', 'similarity_score'])

    outputfile.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_excel(outputfile, index=False)

    if verbose:
        print(result_df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Match spreadsheet rows based on parameter similarity.')
    parser.add_argument('--config', required=True, help='Path to JSON configuration file.')
    args = parser.parse_args()
    main(args.config)


"""
Example for json config file.
{
    "spreadsheet": "KNEE_dataset_file_total.xlsx",
    "sheet": "dataset_file",
    "columns": ["filestem", "INTERNAL_ID", "Projection_data", "REL_PATHOL", "TEST200", "TEST300", "Age1", "Laterality", "INCLUDE"],
    "csv_delimiter": ",",
    "match_column": "filestem",
    "match_parameters": ["age1", "projection_data"],
    "match_threshold": 0.95,
    "filters_df1": {"test300": 1.0},
    "filters_df2": {"rel_pathol": 0.0, "include": 1.0},
    "outputfile": "M:/test_matching_4.xlsx",
    "verbose": true
}
"""