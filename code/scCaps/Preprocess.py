import numpy as np
import pandas as pd

data = np.load('data/PBMC_celltype.npy')
print(data.shape)
pd.DataFrame(data).to_csv("PBMC_celltype.csv")

df = pd.read_csv('data/adata_df_2k_grouped.csv', index_col=0) 
df = pd.read_csv('data/adata_df_2k_grouped.csv', index_col=0) 
df2 = pd.read_csv('PBMC_data.csv', index_col=0) 
df.head()

df_raw = df.copy
np.save('data.npy', df)
df = df.iloc[:, :2000]

df_c = df_raw['target'].to_numpy()