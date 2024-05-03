import pandas as pd
df1=pd.read_csv(r'E:\DSBDA\DSBDA Datasets\n_movies.csv')
print(df1)
df1.head(n=5)
df1.tail(n=6)
df1.index
df1.shape
df1.columns
df1.dtypes
df1.columns.values
df1.describe(include='all')
df1.title
df1.sort_index(axis=1,ascending=False)
df1.sort_values(by='year')
df1.iloc[5]
df1[0:3]
df1.loc[:,['title','rating']]
df1.iloc[:,:3]
