import pandas as pd 
from pathlib import Path
    
def load_csv(file_path : str) -> pd.DataFrame:

    """
    Load data from a csv file and returns a pandas DataFrame
    """
    path =  Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist")
    
    if path.suffix != ".csv":
        raise ValueError(f"File {file_path} is not a csv file")

    df = pd.read_csv(file_path)
    
    if df.empty:
        raise ValueError(f"File {file_path} is empty")
    else :
        print(f"Loaded {len(df)} rows from {file_path}")
    return df

