import pandas as pd
import numpy as np

# To binarize `adata_df_preprocessed.csv`, which contains data that's already standardized (2000 cols)
def binarize_with_std_data(df):
    num_cols = list(df.select_dtypes(exclude=['O']).columns)
    df[num_cols] = (df[num_cols] > 0.0).astype(int)

    return df

# To binarize `adata_df_origin_1888`, which contains data that's not yet standardized (1888 cols)
# Binarization criterion

def binarize_with_origin_data(df):
    num_cols = list(df.select_dtypes(exclude=['O']).columns)
    # for col in num_cols:
    #     col_median = np.median(df[col])
    #     df[col] = (df[col] > col_median).astype(int)
    
    df[num_cols] = np.where(df[num_cols] <= np.median(df[num_cols]), 0, 1)

    return df

def exclude_cate(df):
    return df.select_dtypes(exclude=['O'])

def onehot_encoding(df):
    cat_ftrs = list(df.select_dtypes(include=['O']).columns)
    df = pd.get_dummies(df, prefix=cat_ftrs)
    return df
