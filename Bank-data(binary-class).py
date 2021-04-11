# Import pandas and numpy
import pandas as pd
import numpy as np
import math

#Read the dataset
dff=pd.read_csv("default of credit card clients.csv")
df=dff.drop(['ID'],axis=1)
print('Number of samples: {}'.format(df.shape[0]))
print('Number of features: {}'.format(df.shape[1]))
df

# Print info on white wine
df.info()

#Missing values
df.isnull().sum()

#describe the dataset
df.describe()

# Import `train_test_split` from `sklearn.model_selection`
from sklearn.model_selection import train_test_split
import numpy as np

# Specify the data 
X=df.iloc[:,0:23]

# Specify the target labels and flatten the array
y=df['default payment next month'].values.ravel()
# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


from sklearn.preprocessing import StandardScaler

# Define the scaler 
scaler = StandardScaler().fit(X_train)

# Scale the train set
X_train = scaler.transform(X_train)

# Scale the test set
X_test = scaler.transform(X_test)

