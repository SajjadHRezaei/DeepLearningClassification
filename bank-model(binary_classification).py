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
# Import `train_test_split` from `sklearn.model_selection`
from sklearn.model_selection import train_test_split

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

X_train.shape

#%%
# Import `Sequential` from `keras.models`
from keras.models import Sequential

# Import `Dense` from `keras.layers`
from keras.layers import Dense

# Initialize the constructor
model = Sequential()

# Add an input layer 
model.add(Dense(12, activation='relu', input_shape=(23,)))

# Add one hidden layer 
model.add(Dense(8, activation='relu'))

# Add an output layer 
model.add(Dense(1, activation='sigmoid'))

model.summary()

# Model output shape
model.output_shape

# Model summary
model.summary()

# Model config
model.get_config()

# List all weight tensors 
model.get_weights()

#%%
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
                   
model.fit(X_train, y_train,validation_split=0.2,epochs=20, batch_size=4, verbose=1)

#%%
score = model.evaluate(X_test, y_test,return_dict=True,verbose=1)

print(score)