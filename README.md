** Project workflow **

This research employed a binary variable, default payment (Yes = 1, No = 0), as the response variable. 
This study reviewed the literature and used 23 variables as explanatory variables. 
The aim was to predict if a client will pay his depts 
next month or not For this we loaded the dataset and evaluated its different aspects to get an insight about it. 
We then performed preprocessing by searching for null values and checking for categorical columns (/features). 
We next split the data into test (0.33) and train and normalized the data using standardscaler function of Skitlearn. 
Next we tuned the parameters for deep learning in which we evaluated several parameters such as the number of layers, numer of the nodes/layer, acivation functions, loss functions, and the number of Epochs.
Using Gridsearch function, we detected the best parameters. 
We then performed deep learning using the optimized parameters and reached accuracy of 82 percent.
At the end, we compared different loss values at different epoch numbers, for train and validation datasets by visualizing the data using line plot.