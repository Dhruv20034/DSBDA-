import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
df1=pd.read_csv(r'E:\DSBDA\DSBDA Datasets\cal_cities_lat_long.csv')
print(df1)

df1.columns

df1.mean()
df1.loc[:,'Latitude'].mean()
df1.mean(axis=1)[0:4]

df1.median()
df1.loc[:,'Latitude'].median()
df1.median(axis=1)[0:4]

df1.mode()
df1.loc[:,'Latitude'].mode()

df1.min()
df1.loc[:,'Latitude'].min(skipna=False)

df1.max()
df1.loc[:,'Latitude'].max(skipna=False)


df1.std()
df1.loc[:,'Latitude'].std()

df1.groupby(['Latitude'])['Longitude'].mean()
enc=preprocessing.OneHotEncoder()
enc_df=pd.DataFrame(enc.fit_transform(df1[['Latitude']]).toarray())
enc_df
df_encode=df1.join(enc_df)
df_encode
