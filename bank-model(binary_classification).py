#%%
# Import pandas and numpy
import pandas as pd

#Read the dataset
dff=pd.read_csv("default of credit card clients.csv")
df=dff.drop(['ID'],axis=1)
print('Number of samples: {}'.format(df.shape[0]))
print('Number of features: {}'.format(df.shape[1]))
df
#%%
# Print info on white wine
df.info()
#%%
#Missing values
df.isnull().sum()
#%%
#describe the dataset
df.describe()
#%%
