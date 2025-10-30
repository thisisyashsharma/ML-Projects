import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_and_select(df: pd.DataFrame, cols, drop_na=True):
    df = df.copy()
    if drop_na:
        df = df.dropna(subset=cols)
    return df[cols]

def make_scaler():
    return StandardScaler()
