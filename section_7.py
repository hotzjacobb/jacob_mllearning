import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import sklearn.metrics as metrics

def price_tofloat(price): # takes price as a string and outputs as a float
    price_to_convert = price.replace('$', '')
    price_to_convert = price_to_convert.replace(',', '')
    try:
        return float(price_to_convert)
    except ValueError:
        exit(-1)



airbnb = pd.read_csv('paris_airbnb.csv')
airbnb['price'] = airbnb['price'].apply(price_tofloat) # change prices to floats
np.random.seed(seed=1)
permutation = np.random.permutation(len(airbnb))
airbnb.reindex(permutation)
split_one = airbnb.iloc[0:4000]
split_two = airbnb.iloc[4000:8000]

# train first split to predict second split
knn = KNeighborsRegressor()
knn.fit(split_one[['accommodates']], split_one[['price']])
first_preds = knn.predict(split_two[['accommodates']]) # preds. for second split prices
first_rmse = np.sqrt(metrics.mean_squared_error(split_two[['price']], first_preds))

# train second split to predict first 
knn.fit(split_two[['accommodates']], split_two[['price']])
second_preds = knn.predict(split_one[['accommodates']]) # preds. for second split prices
second_rmse = np.sqrt(metrics.mean_squared_error(split_one[['price']], second_preds))

mean = np.mean([first_preds, second_preds])
print(mean)




