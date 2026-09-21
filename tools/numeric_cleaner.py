import re
import pandas as pd


# Common currency symbols.
# This is intentionally limited to formatting symbols.
CURRENCY_SYMBOLS = "₹$€£¥"


def _clean_numeric_value(value):
    """
    Convert common numeric string representations into numbers.

    Examples
    --------
    "42,000"   -> 42000
    "₹42,000"  -> 42000
    "$42.50"   -> 42.50
    " 50000 "  -> 50000

    Values that cannot safely be interpreted as numbers are
    returned unchanged.
    """

    # Preserve actual missing values
    if pd.isna(value):
        return value

    # Already numeric
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value

    # We only process strings
    if not isinstance(value, str):
        return value

    value = value.strip()

    if not value:
        return value

    # Remove currency symbols
    cleaned = re.sub(
        rf"[{re.escape(CURRENCY_SYMBOLS)}]",
        "",
        value
    )

    # Remove thousands separators
    cleaned = cleaned.replace(",", "")

    # Remove surrounding whitespace
    cleaned = cleaned.strip()

    # Validate numeric format
    #
    # Accepted:
    # 50000
    # -50000
    # 50000.50
    # -50000.50
    # .50
    # 50.
    numeric_pattern = r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$"

    if not re.fullmatch(numeric_pattern, cleaned):
        # Unknown value → DO NOT destroy it
        return value

    try:
        number = float(cleaned)

        # Keep integer-looking values as integers
        if number.is_integer() and "." not in cleaned:
            return int(number)

        return number

    except (ValueError, OverflowError):
        return value


def numeric_cleaner(
    df: pd.DataFrame,
    columns: list[str] | None = None
) -> pd.DataFrame:
    """
    Clean numeric representations in a DataFrame.

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

        df[col] = df[col].apply(_clean_numeric_value)

    return df