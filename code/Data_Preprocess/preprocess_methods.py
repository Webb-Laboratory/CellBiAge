import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# To binarize `adata_df_preprocessed.csv`
# def binarize(df):
#     df_cate = df.select_dtypes(include=['O'])
#     df = df.select_dtypes(exclude=['O'])


def exclude_cate(df):
    return df.select_dtypes(exclude=['O'])


def onehot_encoding(df):
    cat_ftrs = list(df.select_dtypes(include=['O']).columns)
    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'), cat_ftrs)
    #     ])
    #
    # clf = Pipeline(steps=[("preprocessor", preprocessor)])
    # df = clf.fit(df)

    df = pd.get_dummies(df, prefix=cat_ftrs)
    return df
