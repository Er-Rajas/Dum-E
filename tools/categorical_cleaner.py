import pandas as pd


def _clean_categorical_value(value):
    """
    Normalize a categorical/string value.

    Operations:
    - Strip leading/trailing whitespace
    - Convert alphabetic text to lowercase

    Non-string and missing values are preserved.
    """

    if pd.isna(value):
        return value

    if not isinstance(value, str):
        return value

    return value.strip().lower()


def categorical_cleaner(
    df: pd.DataFrame,
    columns: list[str] | None = None
) -> pd.DataFrame:
    """
    Normalize categorical/string columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    columns : list[str] | None
        Columns to process.
        If None, all columns with object/string dtype
        are processed.

    Returns
    -------
    pd.DataFrame
        Cleaned copy of the DataFrame.
    """

    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()

    for col in columns:

        if col not in df.columns:
            continue

        df[col] = df[col].apply(
            _clean_categorical_value
        )

    return df