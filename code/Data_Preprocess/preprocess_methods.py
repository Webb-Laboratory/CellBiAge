import pandas as pd

# To binarize `adata_df_preprocessed.csv`
# def binarize(df):
#     df_cate = df.select_dtypes(include=['O'])
#     df = df.select_dtypes(exclude=['O'])


def exclude_cate(df):
    return df.select_dtypes(exclude=['O'])


def onehot_encoding(df):
    cat_ftrs = list(df.select_dtypes(include=['O']).columns)
    df = pd.get_dummies(df, prefix=cat_ftrs)
    return df
