import pandas as pd


def _clean_datetime_value(value):
    """
    Convert a single value to pandas datetime.

    Invalid/unparseable values are preserved rather than
    silently converted to NaT.
    """

    # Preserve missing values
    if pd.isna(value):
        return value

    # Already datetime
    if isinstance(value, (pd.Timestamp,)):
        return value

    try:
        parsed = pd.to_datetime(
            value,
            errors="raise"
        )

        return parsed

    except (ValueError, TypeError, OverflowError):
        # Preserve the original value.
        return value


def datetime_cleaner(
    df: pd.DataFrame,
    columns: list[str] | None = None
) -> pd.DataFrame:
    """
    Convert date/time columns into pandas datetime values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    columns : list[str] | None
        Columns to process.
        If None, all columns are processed.

    Returns
    -------
    pd.DataFrame
        Cleaned copy of the DataFrame.
    """

    df = df.copy()

    if columns is None:
        columns = list(df.columns)

    for col in columns:

        if col not in df.columns:
            continue

        df[col] = df[col].apply(_clean_datetime_value)

    return df