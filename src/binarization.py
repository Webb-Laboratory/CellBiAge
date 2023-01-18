import pandas as pd

def binarize_data(df):
    num_cols = list(df.select_dtypes(exclude=['O']).columns)
    df.loc[:,num_cols] = (df.loc[:,num_cols] > 0.0).astype(int)
    return df