# FracturesHardCases

Repository providing code for the study performed in paper "Artificial Intelligence Test Set Performance in Difficult Cases Matched with Routine Cases of Pediatric Appendicular Skeleton Fractures"

### Repository organisation

Repository is divided in several moduls where each module is impleneted in its own script inside designated directory. The idea is that code can be used separately based on ones needs. Here is the overview of repository composition and available scripts:

```
FracturesHardCases:
    -->
    -->
    -->
```


Summary of Workflow:

    Read Excel twice into df1 and df2.

    Filter both dataframes with pre-defined column-value conditions.

    Compute similarity scores between each row in df1 and all rows in df2 using:

        Fuzzy string matching (for strings).

        Relative numeric difference (for numbers).

    Pair each row from df1 with the best match from df2 if similarity ≥ threshold.

    Remove matched entries from df2 to prevent duplicates.

    Save results in a new Excel file with original and matched filenames plus similarity score.