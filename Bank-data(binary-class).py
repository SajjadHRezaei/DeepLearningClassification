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

# Import `Sequential` from `keras.models`
from keras.models import Sequential

# Import `Dense` from `keras.layers`
from keras.layers import Dense

# Import `KerasClassifier` from `scikit_learn`
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Import 'Grid and Random search' from 'scikit_learn'
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def FindLayerNodesLinear(n_layers, first_layer_nodes, last_layer_nodes):
    layers = []
    
    nodes_increment = (last_layer_nodes - first_layer_nodes)/ (n_layers-1)
    nodes = first_layer_nodes
    for i in range(1, n_layers+1):
        layers.append(math.ceil(nodes))
        nodes = nodes + nodes_increment
    
    return layers
FindLayerNodesLinear(3, 12, 8)

def createmodel(n_layers, first_layer_nodes, last_layer_nodes, activation_func, loss_func):
    model = Sequential()
    n_nodes = FindLayerNodesLinear(n_layers, first_layer_nodes, last_layer_nodes)
    for i in range(1, n_layers+1):
        if i==1:
            model.add(Dense(first_layer_nodes, input_dim=X_train.shape[1], activation=activation_func))
        else:
            model.add(Dense(n_nodes[i-1], activation=activation_func))
            
    #Finally, the output layer should have a single node in binary classification
    model.add(Dense(1, activation=activation_func))
    model.compile(optimizer='adam', loss=loss_func, metrics = ["accuracy"]) 
    
    return model
model =  KerasClassifier(build_fn=createmodel, verbose = False)

activation_funcs = ['sigmoid', 'relu', 'tanh'] 
loss_funcs = ['binary_crossentropy']
param_grid = dict(n_layers=[2,3,4], first_layer_nodes = [64,23], last_layer_nodes = [12,8],  activation_func = activation_funcs, loss_func = loss_funcs, batch_size = [32], epochs = [20,60])
grid = GridSearchCV(estimator = model, param_grid = param_grid)

grid.fit(X_train,y_train)

print(grid.best_score_)
print(grid.best_params_)
