import pandas as pd
import numpy as np


BAD_VALUES = {
    "nan",
    "null",
    "na",
    "n/a",
    "none",
    "empty",
    "error",
    "unknown",
    "unspecified",
    "unavailable",
    "unassigned",
    "-",
    "garbage"
}


def bad_values(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    for col in df.columns:

        if df[col].dtype == "object" or df[col].dtypes == "str":

            # 1. Normalize strings
            df[col] = df[col].apply(
                lambda x: x.strip().lower()
                if isinstance(x, str)
                else x
            )

            # 2. Convert known missing representations to NaN
            df[col] = df[col].apply(
                lambda x: np.nan
                if isinstance(x, str) and x in BAD_VALUES
                else x
            )

    return df