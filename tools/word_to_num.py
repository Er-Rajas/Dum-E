import pandas as pd 
import numpy as np
from word2number import w2n

def word_to_num(num_cols: list,df:pd.DataFrame):
    """
    conert words from integer columns to number in pandas DataFrame
    """
    for col in num_cols:
        df[col] = df[col].apply(lambda x : w2n.word_to_num(str(x)) if isinstance(x,str) else x)

    