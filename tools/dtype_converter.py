import pandas as pd
import numpy as np


SUPPORTED_DTYPES = {
    "int",
    "integer",
    "float",
    "numeric",
    "string",
    "text",
    "object",
    "category",
    "categorical",
    "datetime",
    "datetime64",
    "id",
}


def _convert_column(
    series: pd.Series,
    dtype: str
) -> pd.Series:
    """
    Convert one Series to the requested semantic dtype.

    Conversion is strict. Values that cannot be safely
    converted raise an error instead of becoming NaN.
    """

    dtype = dtype.strip().lower()

    # --------------------------------
    # Numeric / integer
    # --------------------------------

    if dtype in {"int", "integer"}:

        numeric = pd.to_numeric(
            series,
            errors="coerce"
        )

        invalid_mask = (
            numeric.isna()
            & series.notna()
        )

        if invalid_mask.any():

            invalid_values = series[invalid_mask].unique().tolist()

            raise ValueError(
                f"Column '{series.name}' contains values "
                f"that cannot be converted to integer: "
                f"{invalid_values}"
            )

        # Check that values are actually integers
        non_integer = (
            numeric.dropna() % 1 != 0
        )

        if non_integer.any():

            invalid_values = numeric.dropna()[
                non_integer
            ].unique().tolist()

            raise ValueError(
                f"Column '{series.name}' contains non-integer "
                f"values: {invalid_values}"
            )

        # Nullable integer so NaN can remain
        return numeric.astype("Int64")

    # --------------------------------
    # Float / numeric
    # --------------------------------

    if dtype in {"float", "numeric"}:

        numeric = pd.to_numeric(
            series,
            errors="coerce"
        )

        invalid_mask = (
            numeric.isna()
            & series.notna()
        )

        if invalid_mask.any():

            invalid_values = series[invalid_mask].unique().tolist()

            raise ValueError(
                f"Column '{series.name}' contains values "
                f"that cannot be converted to numeric: "
                f"{invalid_values}"
            )

        return numeric.astype(float)

    # --------------------------------
    # String / text
    # --------------------------------

    if dtype in {"string", "text"}:
        return series.astype("string")

    # --------------------------------
    # Object
    # --------------------------------

    if dtype == "object":
        return series.astype("object")

    # --------------------------------
    # Category
    # --------------------------------

    if dtype in {"category", "categorical"}:
        return series.astype("category")

    # --------------------------------
    # Datetime
    # --------------------------------

    if dtype in {"datetime", "datetime64"}:

        parsed = pd.to_datetime(
            series,
            errors="coerce"
        )

        invalid_mask = (
            parsed.isna()
            & series.notna()
        )

        if invalid_mask.any():

            invalid_values = series[invalid_mask].unique().tolist()

            raise ValueError(
                f"Column '{series.name}' contains values "
                f"that cannot be converted to datetime: "
                f"{invalid_values}"
            )

        return parsed

    # --------------------------------
    # ID
    # --------------------------------

    if dtype == "id":
        # IDs should generally remain strings.
        # This prevents IDs such as E001, 00123, etc.
        # from being interpreted as numeric data.
        return series.astype("string")

    raise ValueError(
        f"Unsupported dtype '{dtype}'. "
        f"Available types: {sorted(SUPPORTED_DTYPES)}"
    )


def dtype_converter(
    df: pd.DataFrame,
    expected_types: dict
) -> pd.DataFrame:
    """
    Convert DataFrame columns according to a semantic
    dtype mapping.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    expected_types : dict
        Mapping of column name -> expected dtype.

        Example:
        {
            "age": "integer",
            "salary": "float",
            "joining_date": "datetime",
            "gender": "category",
            "employee_id": "id"
        }

    Returns
    -------
    pd.DataFrame
        DataFrame with converted dtypes.
    """

    df = df.copy()

    for column, dtype in expected_types.items():

        if column not in df.columns:
            raise KeyError(
                f"Column '{column}' specified in expected_types "
                f"does not exist in DataFrame."
            )

        df[column] = _convert_column(
            df[column],
            dtype
        )

    return df