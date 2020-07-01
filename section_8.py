import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import sklearn.metrics as metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

car_data = pd.read_csv('imports-85.data', names=['symboling', 'normalized-losses', 
                       'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 
                       'drive-wheels', 'engine-locaion', 'wheel-base', 'length', 'width', 
                       'height', 'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 
                       'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 
                       'peak-rpm', 'city-mpg', 'highway-mpg', 'price'])
cars_numerics = car_data[['normalized-losses', 'wheel-base', 'length', 'width', 'height', 
                          'curb-weight', 'engine-size', 'bore', 'stroke', 'compression-ratio', 
                          'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']]

cars_numerics = cars_numerics.replace(to_replace='?', value=np.nan) # csv has question marks for unknown values
cars_numerics = cars_numerics.dropna(subset=['normalized-losses', 'price']) # throw out row missing either of these cols.
# cast as floats b/c pandas made cols. strings b/c thay had '?' values
cars_numerics = cars_numerics.astype('float')
cars_numerics = cars_numerics.fillna(value={'bore': cars_numerics['bore'].mean(), # replace missing w/ avg.
                                     'stroke': cars_numerics['stroke'].mean()})
# standarize data
price_col = cars_numerics['price']
cars_numerics = (cars_numerics - cars_numerics.mean()) / cars_numerics.std()
cars_numerics['price'] = price_col

def knn_train_test(learn_column, target_column, n_neihgbours, dataset):
    model = KNeighborsRegressor(n_neighbors=n_neihgbours)
    training_data = dataset[0:int(len(dataset)*4/5)] # 80% of data
    testing_data = dataset[int(len(dataset)*4/5):len(dataset)] # 20% of the data
    model.fit(training_data[[learn_column]], training_data[[target_column]])
    predictions = model.predict(testing_data[[learn_column]])
    mean_error = np.sqrt(metrics.mean_squared_error(testing_data[target_column], predictions))
    return mean_error

error_cats = []
for col in cars_numerics.columns:
    error_different_k = []
    for i in range(1, 9, 2):
        error_different_k.append(knn_train_test(col, 'price', i , cars_numerics))
    error_cats.append([col, np.mean(error_different_k)])

for left in range(len(error_cats) - 1):    # selection sort
    curr_min_ind = left
    curr_min  = error_cats[left][1]
    for ind in range(left + 1, len(error_cats)):
        if (error_cats[ind][1] < curr_min):
            curr_min_ind = ind
            curr_min = error_cats[ind][1]
    temp = error_cats[left]
    error_cats[left] = error_cats[curr_min_ind]
    error_cats[curr_min_ind] = temp
print(error_cats)
