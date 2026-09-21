import numpy as np 
import pandas as pd 

def strip_symbols(df: pd.DataFrame):
    """
    Strip symbols from categorical data in a pandas DataFrame 
    """
    for column in df.columns :
        if df[column].dtype == 'object' :
            df[column] = df[column].str.replace('[^\w\s]','')
            

def bad_values(df: pd.DataFrame) :
    """
    Normalize categorical data in a pandas DataFrame 
    """
    non_consistent_values = ['None','null','Empty','ERROR','NA','nan']
    for column in df.columns :
        if df[column].dtype == 'object' :
            df[column] = df[column].apply(lambda x : x if x not in non_consistent_values else np.nan)
    return df
