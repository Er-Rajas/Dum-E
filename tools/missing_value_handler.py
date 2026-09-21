import pandas as pd
import numpy as np
import itertools


AVAILABLE_METHODS = {
    "ffill",
    "bfill",
    "mean",
    "median",
    "mode",
    "drop",
    "increment" 
}


def generate_alphabet_seq(n: int):
    """Helper function to generate an Excel-like alphabet sequence (A, B... Z, AA, AB...)."""
    result = []
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for length in itertools.count(1):
        for p in itertools.product(chars, repeat=length):
            result.append("".join(p))
            if len(result) == n:
                return  result


def missing_value_handler(
    df: pd.DataFrame,
    column: str,
    method: str
) -> pd.DataFrame:
    """
    Handle missing values in one DataFrame column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    column : str
        Column on which the operation should be performed.

    method : str
        Missing-value handling method.

        Available:
        - ffill
        - bfill
        - mean
        - median
        - mode
        - drop
        - increment

    Returns
    -------
    pd.DataFrame
        Cleaned copy of the DataFrame.

    Raises
    ------
    ValueError
        If the requested method is not supported.

    KeyError
        If the requested column does not exist.
    """

    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' does not exist in DataFrame."
        )

    method = method.strip().lower()

    if method not in AVAILABLE_METHODS:
        raise ValueError(
            f"'{method}' is not a Dum-E capability. "
            f"Available methods: {sorted(AVAILABLE_METHODS)}"
        )

    df = df.copy()

    # Nothing to do
    if not df[column].isna().any():
        return df

    # --------------------------------
    # Forward fill
    # --------------------------------
    if method == "ffill":
        df[column] = df[column].ffill()

    # --------------------------------
    # Backward fill
    # --------------------------------
    elif method == "bfill":
        df[column] = df[column].bfill()

    # --------------------------------
    # Mean
    # --------------------------------
    elif method == "mean":

        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TypeError(
                f"Mean imputation requires a numeric column. "
                f"'{column}' is not numeric."
            )

        mean_value = df[column].mean()

        if pd.isna(mean_value):
            raise ValueError(
                f"Cannot calculate mean for '{column}'. "
                f"The column contains no valid numeric values."
            )

        df[column] = df[column].fillna(mean_value)

    # --------------------------------
    # Median
    # --------------------------------
    elif method == "median":

        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TypeError(
                f"Median imputation requires a numeric column. "
                f"'{column}' is not numeric."
            )

        median_value = df[column].median()

        if pd.isna(median_value):
            raise ValueError(
                f"Cannot calculate median for '{column}'. "
                f"The column contains no valid numeric values."
            )

        df[column] = df[column].fillna(median_value)

    # --------------------------------
    # Mode
    # --------------------------------
    elif method == "mode":

        mode_values = df[column].mode(dropna=True)

        if mode_values.empty:
            raise ValueError(
                f"Cannot calculate mode for '{column}'. "
                f"The column contains no valid values."
            )

        # If multiple modes exist, use the first one.
        df[column] = df[column].fillna(mode_values.iloc[0])

    # --------------------------------
    # Drop
    # --------------------------------
    elif method == "drop":

        df = df.dropna(subset=[column])
    
        # --------------------------------
    # Increment (Numeric and Alphabetical)
    # --------------------------------
    elif method == "increment":
        missing_mask = df[column].isna()
        num_missing = missing_mask.sum()

        # Handle numeric increments
        if pd.api.types.is_numeric_dtype(df[column]):
            max_val = df[column].max()
            start_val = 1 if pd.isna(max_val) else int(max_val) + 1
            df.loc[missing_mask, column] = range(
                start_val, start_val + num_missing
            )

        # Handle text/alphabetical increments
        elif pd.api.types.is_string_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            
            # 1. Clean valid entries to find the alphabetical baseline
            valid_entries = df[column].dropna().astype(str).str.strip()
            
            # Detect case preference based on the first valid entry (default to lowercase if empty)
            is_lowercase = valid_entries.iloc[0].islower() if not valid_entries.empty else True
            
            # Helper to convert a string (like 'a', 'z', 'aa') to an integer index
            def alpha_to_num(s):
                num = 0
                for char in s.upper():
                    if 'A' <= char <= 'Z':
                        num = num * 26 + (ord(char) - ord('A') + 1)
                return num

            # Helper to convert an integer index back to a string
            def num_to_alpha(n, lowercase=True):
                result = []
                while n > 0:
                    n, remainder = divmod(n - 1, 26)
                    result.append(chr((ord('a') if lowercase else ord('A')) + remainder))
                return "".join(reversed(result))

            # 2. Determine starting point index
            if not valid_entries.empty:
                # Find the maximum alphabetical value present
                max_alpha = max(valid_entries, key=alpha_to_num)
                start_idx = alpha_to_num(max_alpha) + 1
            else:
                start_idx = 1

            # 3. Generate sequential continuation strings
            generated_letters = [
                num_to_alpha(i, lowercase=is_lowercase) 
                for i in range(start_idx, start_idx + num_missing)
            ]
            
            df.loc[missing_mask, column] = generated_letters

        else:
            raise TypeError(
                f"Increment method is only supported for numeric or string columns. "
                f"'{column}' is {df[column].dtype}."
            )

    return df