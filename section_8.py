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
# cast as types as pandas made cols. strings b/c thay had '?' values
# cars_numerics['normalized-losses'] = cars_numerics['normalized-losses'].astype('int')
# cars_numerics['price'] = cars_numerics['price'].astype('int')
# cars_numerics['horsepower'] = cars_numerics['horsepower'].astype('int')
# cars_numerics['peak-rpm'] = cars_numerics['peak-rpm'].astype('int')
# cars_numerics['bore'] = cars_numerics['bore'].astype('float')
# cars_numerics['stroke'] = cars_numerics['stroke'].astype('float')
cars_numerics = cars_numerics.astype('float')
cars_numerics = cars_numerics.fillna(value={'bore': cars_numerics['bore'].mean(), # replace missing w/ avg.
                                     'stroke': cars_numerics['stroke'].mean()})
# standarize data
cars_numerics = (cars_numerics - cars_numerics.mean()) / cars_numerics.std()
print(cars_numerics)
