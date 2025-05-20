# Match folders of files based on a set of parameters, 2024-03-05

import os
import pathlib
import pandas
import sys
from rapidfuzz import fuzz

# =========================== CONFIGURATION SETTINGS ===========================
spreadsheet = pathlib.Path(
    'M:/Documents/Beruf/Wissenschaft/Laufend/2024 - Predict knee MRI - Stranger/KNEE_dataset_file_total.xlsx'
)  # Path to the CSV or XLSX file

xlsx_sheetname = 'dataset_file'  # Name of the sheet in Excel file

# Columns to load from spreadsheet (case-insensitive match later)
spreadsheet_columns = ['filestem', 'INTERNAL_ID', 'Projection_data', 'REL_PATHOL', 
                       'TEST200', 'TEST300', 'Age1', 'Laterality', 'INCLUDE']

csv_delimiter = ','  # Only used if reading CSV instead of XLSX

match_column = 'filestem'  # Column used as the identifier for matching
match_parameters = ['age1', 'projection_data']  # Parameters to match (lowercase)
match_threshold = 0.95  # Similarity threshold [0–1]

# Filters to apply on the two dataframes
filters_df1 = {
    'test300': 1.0
}

filters_df2 = {
    'rel_pathol': 0.0,
    'include': 1.0
}

outputdir = pathlib.Path('M:')
outputfile = outputdir / 'test_matching_4.xlsx'  # Output Excel file
verbose_output = True  # Whether to include detailed data
copy_files = False  # If True, copy matched files (currently not used)

# =========================== FUNCTION DEFINITIONS ===========================

def read_spreadsheet_to_df(xlsx, sheet, cols):
    """
    Reads data from an Excel or CSV spreadsheet into a pandas DataFrame.

    Parameters:
        xlsx (Path): Path to the spreadsheet.
        sheet (str): Sheet name if Excel file.
        cols (list): Columns to load.

    Returns:
        pandas.DataFrame: The loaded and cleaned DataFrame.
    """
    try:
        if str(xlsx).endswith('.xlsx'):
            spreadsheet_df = pandas.read_excel(xlsx, sheet_name=sheet, usecols=cols)
        elif str(xlsx).endswith('.csv'):
            spreadsheet_df = pandas.read_csv(
                os.path.normpath(xlsx), dtype=str, sep=csv_delimiter,
                skipinitialspace=True, usecols=cols
            )
        else:
            raise ValueError("Unsupported file format.")
        
        # Normalize column names to lowercase
        spreadsheet_df.columns = map(str.lower, spreadsheet_df.columns)
        return spreadsheet_df
    except Exception as ex:
        print('ERROR:', 'Problem reading dataset file into Pandas. Check sheet name and column names.')
        print(ex)
        sys.exit(1)

def similarity(param1, param2):
    """
    Computes similarity between two values (string or numeric).

    For strings: fuzzy string ratio using rapidfuzz.
    For numbers: similarity based on relative difference.

    Parameters:
        param1: First value (string or number).
        param2: Second value (string or number).

    Returns:
        float: Similarity score between 0 and 1.
    """
    x = 0
    try:
        if isinstance(param1, str) and isinstance(param2, str):
            x = fuzz.ratio(param1, param2) / 100
        elif isinstance(param1, (int, float)) and isinstance(param2, (int, float)):
            max_val = max(abs(param1), abs(param2), 1)  # Avoid division by zero
            x = 1 - abs(param1 - param2) / max_val
        return x
    except:
        return x

# =========================== MAIN SCRIPT ===========================

def main():
    """
    Main function to match files in two filtered datasets based on similarity
    of selected parameters. Results are saved to an Excel file.
    """
    # Load dataset twice (will be filtered separately)
    df1 = read_spreadsheet_to_df(spreadsheet, xlsx_sheetname, spreadsheet_columns)
    df2 = read_spreadsheet_to_df(spreadsheet, xlsx_sheetname, spreadsheet_columns)

    # Apply filters to df1
    for k, v in filters_df1.items():
        df1 = df1.drop(df1[df1[k] != v].index)
        df2 = df2.drop(df2[df2[k] == v].index)

    # Apply filters to df2
    for k, v in filters_df2.items():
        df2 = df2.drop(df2[df2[k] != v].index)

    print(df1, df1.count())
    print(df2, df2.count())

    def similarity_score(row1, row2):
        """
        Calculates average similarity score across match_parameters for two rows.

        Parameters:
            row1 (Series): Row from df1.
            row2 (Series): Row from df2.

        Returns:
            float: Average similarity score.
        """
        similarity_list = []
        for item in match_parameters:
            similarity_list.append(similarity(row1[item], row2[item]))
        return sum(similarity_list) / len(similarity_list)

    matched_filenames = []

    # Iterate over rows in df1 and find best matching row in df2
    for index1, row1 in df1.iterrows():
        best_score = 0
        best_match = None
        best_index = None

        for index2, row2 in df2.iterrows():
            score = similarity_score(row1, row2)
            if score > best_score:
                best_score = score
                if score >= match_threshold:
                    best_match = row2[match_column]
                    best_index = index2

        matched_filenames.append((row1[match_column], best_match, best_score))

        # Remove matched row from df2 to avoid duplicate matching
        if best_index is not None:
            df2.drop(index=best_index, inplace=True)

    # Add unmatched entries in df1 with None match
    for index1, row1 in df1.iterrows():
        if row1[match_column] not in [x[0] for x in matched_filenames]:
            matched_filenames.append((row1[match_column], None, None))

    result_df = pandas.DataFrame(matched_filenames, 
                                 columns=[match_column+'_df1', match_column+'_df2', 'similarity_score'])

    print(result_df, result_df.count())

    # Save results to Excel
    try:
        outputdir.mkdir(parents=True, exist_ok=True)
        result_df.to_excel(outputfile, index=False)
    except Exception as ex:
        print('Error writing EXCEL file.', ex)

if __name__ == '__main__':
    main()
