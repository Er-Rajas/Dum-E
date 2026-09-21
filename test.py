import pandas as pd
import numpy as np

from agent.agent import DumE


# -----------------------------------------
# Test dataset
# -----------------------------------------

df = pd.DataFrame({
    "Employee_ID": [
        "E001",
        "E002",
        "E003",
        "E004",
        "E005"
    ],

    "Age": [
        25,
        "twenty five",
        30,
        "N/A",
        40
    ],

    "Salary": [
        "₹42,000",
        "50000",
        "ERROR",
        "65000",
        np.nan
    ],

    "Gender": [
        "Female",
        "M",
        "female",
        "Male",
        "F"
    ],

    "Joining_Date": [
        "2025-09-05",
        "2024/01/10",
        "N/A",
        "garbage",
        "2023-12-01"
    ],

    "Department": [
        "Finance",
        "finance",
        "-",
        "IT",
        "Sales"
    ]
})


# -----------------------------------------
# Create Dum-E
# -----------------------------------------

dume = DumE(
    model="qwen3:4b"
)


# -----------------------------------------
# Run Dum-E
# -----------------------------------------

cleaned_df, schema = dume.clean(df)


# -----------------------------------------
# Results
# -----------------------------------------

print("\n" + "=" * 60)
print("ORIGINAL DATA")
print("=" * 60)

print(df)


print("\n" + "=" * 60)
print("DUM-E SCHEMA")
print("=" * 60)

print(schema)


print("\n" + "=" * 60)
print("CLEANED DATA")
print("=" * 60)

print(cleaned_df)


print("\n" + "=" * 60)
print("FINAL DTYPES")
print("=" * 60)

print(cleaned_df.dtypes)