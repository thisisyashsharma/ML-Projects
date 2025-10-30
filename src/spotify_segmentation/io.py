from pathlib import Path
import pandas as pd

def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    return df

def read_in_chunks(path, chunksize=200_000, downcast=True, usecols=None):
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path, columns=usecols)
        yield _downcast(df) if downcast else df
    else:
        for chunk in pd.read_csv(path, chunksize=chunksize, usecols=usecols):
            yield _downcast(chunk) if downcast else chunk

def save_labels(path, df):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
