import pandas as pd 
import numpy as np
# from pathlib import Path
# from word2number import w2n
from tools.missing_normalization import bad_values
from tools.value_classifier import classify_values


df = pd.DataFrame({
    "A": [1, 2, "nan", 4, 5],
    "B": [10, 20, "two", "three", "four"],
    "C": ["A", "B", np.nan, "C", "B"],
    "D" : ["₹42,000", "50000", "ERROR", "Twenty Five", np.nan],
    "E" : ["2025-09-05", "2024/01/10", "N/A", "garbage", "2023-12-01"]
})
excepted_types = {
    "A":"numeric",
    "B":"numeric",
    "C":"object",
    "D":"numeric",
    "E":"datetime"
}
# print("=" * 10)
# print("old df")
# print(df)
df = bad_values(df)
print(df)
print(classify_values(df,excepted_types))

# print("=" * 10)
# print("new df")
# print(df)