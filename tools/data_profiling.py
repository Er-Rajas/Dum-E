import pandas as pd

def data_profiling(df : pd.DataFrame):
    return {
        "head" : df.head(5),
        "dtypes" : df.dtypes,
        "cols" : df.columns,
        "shape": df.shape
    }