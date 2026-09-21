import pandas as pd 
import numpy as np
from word2number import w2n


def try_prase_num_string(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val,(int,float)):
        return val 
    try:
        return w2n.word_to_num(str(val))
    except ValueError:
        return val
    
def find_text_num_column(df:pd.DataFrame,sample_size=10,threshold = 0.3) -> list :
    bad_num_cols = []
    mixed_cols = df.select_dtypes(include=['object', 'string']).columns

    
    for col in mixed_cols:
        sample_val = df[col].dropna().unique()[:sample_size]
        if len(sample_val) == 0:
            continue
        sucess_count = 0
        for val in sample_val:
            if str(val).isdigit():
                sucess_count +=1
                continue
            try :
                w2n.word_to_num(str(val))
                sucess_count +=1
            except ValueError:
                pass
        if sucess_count/len(sample_val) > threshold:
            bad_num_cols.append(col)
    return bad_num_cols

def word_to_num(df:pd.DataFrame) -> pd.DataFrame:
    cols_to_clean = find_text_num_column(df)
    for col in cols_to_clean:
        df[col] = df[col].apply(try_prase_num_string)
    return df