import pandas as pd
import numpy as np
from scipy.spatial import distance
import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsRegressor
import sys

airbnb = pd.read_csv('paris_airbnb.csv')
seed = np.random.seed(1)
# get random order
random_order = np.random.permutation(len(airbnb))
# put data in random order
airbnb_random_sort = airbnb.iloc[random_order]
airbnb = airbnb_random_sort
# remove dollar sign and commas
new_price = airbnb['price'].str.replace(',', '')  # remove commas
new_price = new_price.str.replace('$', "")  # remove $
try:
    airbnb['price'] = new_price.astype('float')
except ValueError:
    print("non-numeric price")
    sys.exit(-1)
# remove data not being used for predictions
airbnb = airbnb.drop(labels=['room_type', 'city', 'state', 'latitude', 'longitude', 
'zipcode', 'host_response_rate', 'host_acceptance_rate', 'host_listings_count'], axis=1)
airbnb = airbnb.drop(labels=['cleaning_fee', 'security_deposit'], axis=1)
airbnb = airbnb.dropna() # get rid of nulls
# below we fit all columns to a standard distribution 
# to avoid diffent unist having differet weight
normalized = (airbnb - airbnb.mean()) / airbnb.std()
normalized['price'] = airbnb['price']
first_listing = normalized.iloc[0]
first_list_vec = [first_listing['bedrooms'], first_listing['accommodates']]
fifth_listing = normalized.iloc[4][['accommodates', 'bedrooms']]
fifth_list_vec = [fifth_listing['bedrooms'], fifth_listing['accommodates']]
first_fifth_distance = distance.euclidean(first_list_vec, fifth_list_vec)
# using scipy
knn = KNeighborsRegressor(algorithm="brute")
# seperate data
training_df = normalized.iloc[0:6000]
test_df = normalized.iloc[6000:]
# here we take the columns that we're going to use to make a prediction
training_features = training_df[['accommodates', 'bedrooms']]
# these are the columns that we want to extrapolate
training_target = training_df[['price']]
knn.fit(training_features, training_target)
predictions = knn.predict(test_df[['accommodates', 'bedrooms']])
# print(predictions)
# calculate the mean squared error
two_features_mse = metrics.mean_squared_error(test_df['price'], predictions)
print(two_features_mse)
two_features_rsme = np.sqrt(two_features_mse)
print(two_features_rsme)
# create a model with four categories
kn4 = KNeighborsRegressor(algorithm='brute')
kn4.fit(training_df[['accommodates', 'bedrooms', 'bathrooms', 'number_of_reviews']], training_df['price'])
four_predictions = kn4.predict(test_df[['accommodates', 'bedrooms', 'bathrooms', 'number_of_reviews']])
four_mse = metrics.mean_squared_error(test_df['price'], four_predictions)
four_rmse = np.sqrt(four_mse)
print(four_mse)
print(four_rmse)
knall = KNeighborsRegressor(algorithm='brute')
training_cols = training_df.columns.tolist()
training_cols.remove('price') # all cols. except price
knall.fit(training_df[training_cols], training_df[['price']])
predictions_all = knall.predict(test_df[training_cols])
all_mse = metrics.mean_squared_error(test_df['price'], predictions_all)
all_rsme = np.sqrt(all_mse)
print(all_mse)
print(all_rsme)