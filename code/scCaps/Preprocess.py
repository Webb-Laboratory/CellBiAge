import numpy as np
import pandas as pd


df = pd.read_csv('data/adata_df_2k_grouped.csv', index_col=0) 
df.head()

df_raw = df.copy
df = df.iloc[:, :2000]
np.save('data.npy', df)

df_c = df_raw['target']
np.save('data.npy', df_c)
