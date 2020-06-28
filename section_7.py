import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import sklearn.metrics as metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

def price_tofloat(price): # takes price as a string and outputs as a float
    price_to_convert = price.replace('$', '')
    price_to_convert = price_to_convert.replace(',', '')
    try:
        return float(price_to_convert)
    except ValueError:
        exit(-1)


def k_fold(dataset, k):
    list_of_rsme = [] # result
    model = KNeighborsRegressor()
    partition_size = len(dataset) / k
    dataset['fold'] = dataset.index.values / partition_size
    dataset['fold'].astype('int')
    for current_test in range(k):
        training_data = dataset[dataset['fold'] != current_test]
        testing_data = dataset[dataset['fold'] == current_test]
        model.fit(training_data[['accommodates']], training_data[['price']])
        predicted_prices = model.predict(testing_data[['accommodates']])
        rsme = np.sqrt(metrics.mean_squared_error(testing_data['price'], predicted_prices))
        list_of_rsme.append(rsme) 
    return list_of_rsme



airbnb = pd.read_csv('paris_airbnb.csv')
airbnb['price'] = airbnb['price'].apply(price_tofloat) # change prices to floats
np.random.seed(seed=1)

permutation = np.random.permutation(airbnb.index)
airbnb = airbnb.reindex(permutation)

list_of_results = k_fold(airbnb, 5)
avg_rsme = np.mean(list_of_results)

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


# add values to rows for k-fold
airbnb['fold'] = airbnb.index.values / (len(airbnb) / 5)
airbnb['fold'] = airbnb['fold'].astype('int')


folds_model = KNeighborsRegressor()
test_part = airbnb[airbnb['fold'] == 0]
training_part = airbnb[airbnb['fold'] != 0]
folds_model.fit(training_part[['accommodates']], training_part[['price']])
predicted_prices = folds_model.predict(test_part[['accommodates']])
rsme = np.sqrt(metrics.mean_squared_error(test_part['price'], predicted_prices))

# Use library functions to perform k-fold
kf = KFold(shuffle=True, random_state=1)
knn = KNeighborsRegressor()
results = cross_val_score(knn, airbnb[['accommodates']], 
                          airbnb['price'], scoring='neg_mean_squared_error', cv=kf)
results = np.sqrt(np.absolute(results))
result = np.mean(results)
print(results)
print(result)




