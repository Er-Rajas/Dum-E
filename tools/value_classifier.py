import pandas as pd
from word2number import w2n
import re


def classify_values(df: pd.DataFrame, expected_types: dict):

    classification_report = {}

    for col in df.columns:

        expected_type = expected_types.get(col)

        classification_report[col] = {
            "expected_type": expected_type,
            "numeric": [],
            "word_numeric": [],
            "formatted_numeric": [],
            "datetime": [],
            "missing": [],
            "invalid": [],
            "valid": []
        }

        for index, value in df[col].items():

            # -------------------------
            # 1. Missing
            # -------------------------
            if pd.isna(value):
                classification_report[col]["missing"].append(index)
                continue

            # -------------------------
            # 2. Numeric column
            # -------------------------
            if expected_type == "numeric":

                # Actual int / float
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    classification_report[col]["numeric"].append(index)
                    continue

                if isinstance(value, str):

                    value = value.strip()

                    # Numeric string
                    try:
                        float(value)

                        # Prevent "nan" / "inf"
                        if value.lower() not in {
                            "nan",
                            "+nan",
                            "-nan",
                            "inf",
                            "+inf",
                            "-inf",
                            "infinity",
                            "+infinity",
                            "-infinity"
                        }:
                            classification_report[col]["numeric"].append(index)
                            continue

                    except ValueError:
                        pass

                    # Word number
                    try:
                        w2n.word_to_num(value)
                        classification_report[col]["word_numeric"].append(index)
                        continue

                    except (ValueError, TypeError):
                        pass

                    # Formatted numeric
                    cleaned = re.sub(r"[₹$€£,\s]", "", value)

                    if re.fullmatch(r"[-+]?\d*\.?\d+", cleaned):
                        classification_report[col]["formatted_numeric"].append(index)
                        continue

                    # Invalid
                    classification_report[col]["invalid"].append(index)
                    continue

            # -------------------------
            # 3. Datetime column
            # -------------------------
            elif expected_type == "datetime":

                try:
                    pd.to_datetime(value)
                    classification_report[col]["datetime"].append(index)

                except (ValueError, TypeError):
                    classification_report[col]["invalid"].append(index)

                continue

            # -------------------------
            # 4. Categorical / object / text
            # -------------------------
            else:
                classification_report[col]["valid"].append(index)

    return classification_report