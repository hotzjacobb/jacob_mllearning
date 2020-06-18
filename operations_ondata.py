import pandas as pd
import numpy as np

airbnb = pd.read_csv('paris_airbnb.csv')
airbnb['distance'] = airbnb['accommodates'].apply(lambda x: abs(x - 3))
seed = np.random.seed(1)
random_order = np.random.permutation(len(airbnb))
# don't know how to properly write line below
airbnb_random_sort = airbnb.iloc[random_order]
airbnb = airbnb_random_sort
airbnb = airbnb.sort_values('distance')
airbnb.iloc[0:10]['price']