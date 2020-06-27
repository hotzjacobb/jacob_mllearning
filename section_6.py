import pandas as pd
import numpy as np 
import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plot

train_df = pd.read_csv('paris_airbnb_train.csv')
test_df = pd.read_csv('paris_airbnb_test.csv')
# hyperparams = [] # list to hold hyperparams that will be tested for k val.
# mse_values = [] # list to hold error value; index is the the value of the hyperparam - 1
# for hyperparam in range(1, 100): # 1...5
#     hyperparams.append(hyperparam)
# for i in hyperparams:
#     knn = KNeighborsRegressor(n_neighbors=i, algorithm='brute')
#     categories = train_df.columns.tolist()
#     categories = remove('price')
#     knn.fit(train_df[['accommodates', 'bedrooms', 'bathrooms', 'number_of_reviews']], train_df['price'])
#     predictions = knn.predict(test_df[['accommodates', 'bedrooms', 'bathrooms', 'number_of_reviews']])
#     mse = metrics.mean_squared_error(test_df['price'], predictions)
#     mse_values.append(mse)
# plot.scatter(range(1, 70), mse_values)
# plot.show()


# Redoing the above with other hyperparams just to get accustomed to operations
two_columns_names = ['bathrooms', 'accommodates']
two_col_min_mse = -1
two_col_min_ind = -1
for i in range(1, 100):
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(train_df[two_columns_names], train_df['price'])
    predictions = knn.predict(test_df[two_columns_names])
    mse = metrics.mean_squared_error(test_df['price'], predictions)
    if two_col_min_ind == -1:
        two_col_min_ind = 0 # just for first iter.
        two_col_min_mse = mse
    elif mse < two_col_min_mse: # returns last added value to dic.; i.e. the min
            two_col_min_ind = i
            two_col_min_mse = mse
two_col_dic = {i: two_col_min_mse} # the tutorial problem wanted answer as a dic. 
print(two_col_dic)
# if not for familiarizing myself with workflow I would have written a function
# repetetive code
three_columns_names = ['bathrooms', 'accommodates', 'bedrooms']
three_col_min_mse = -1
three_col_min_ind = -1
for i in range(1, 100):
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(train_df[three_columns_names], train_df['price'])
    predictions = knn.predict(test_df[three_columns_names])
    mse = metrics.mean_squared_error(test_df['price'], predictions)
    if three_col_min_ind == -1:
        three_col_min_ind = 0 # just for first iter.
        three_col_min_mse = mse
    elif mse < three_col_min_mse:
        three_col_min_ind = i
        three_col_min_mse = mse
three_col_dic = {three_col_min_ind: three_col_min_mse}
print(three_col_dic)

