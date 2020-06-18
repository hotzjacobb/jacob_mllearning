import pandas as pd
import numpy as np

def predict_price(rooms): # rooms is an int represting desired # of rooms
    temp_df = airbnb.copy()
    temp_df['distance'] = temp_df['accommodates'].apply(lambda x: abs(x - rooms))
    temp_df = temp_df.sort_values('distance')
    print(temp_df.iloc[0:5]['price'])
    return temp_df.iloc[0:5]['price'].mean()

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
    first_five = airbnb.iloc[0:5]['price']  # just the first five
    mean_price = first_five.mean()  # avg of those first five rows
    print(mean_price)
    print('______________________________')
except ValueError:
    print("non-numeric price")
# call function with params to predict price
acc_one = predict_price(1)
print(acc_one)
acc_two = predict_price(2)
print(acc_two)
acc_four = predict_price(4)
print(acc_four)

