import pandas as pd
import numpy as np

airbnb = pd.read_csv('paris_airbnb.csv')
airbnb['distance'] = airbnb['accommodates'].apply(lambda x: abs(x - 3))
seed = np.random.seed(1)
# get random order
random_order = np.random.permutation(len(airbnb))
# put data in random order
airbnb_random_sort = airbnb.iloc[random_order]
airbnb = airbnb_random_sort
airbnb = airbnb.sort_values('distance')
# remove dollar sign and commas
new_price = airbnb['price'].str.replace(',', '')  # remove commas
new_price = new_price.str.replace('$', "")  # remove $
try:
    airbnb['price'] = new_price.astype('float')
except ValueError:
    print("non-numeric price")
first_five = airbnb.iloc[0:5]['price']  # just the first five
mean_price = first_five.mean()  # avg of those first five rows
print(mean_price)
