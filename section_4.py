import pandas as pd
import numpy as np


airbnb = pd.read_csv('paris_airbnb.csv')

# clean up price data for calculations
new_price = airbnb['price'].str.replace(',', '')  # remove commas
new_price = new_price.str.replace('$', "")  # remove $
try:
    airbnb['price'] = new_price.astype('float')
except ValueError:
    print("non-numeric price")

airbnb_len = len(airbnb)
airbnb_training = airbnb.iloc[0:int(airbnb_len*3/4)].copy() # used in calculation
airbnb_testing = airbnb.iloc[int(airbnb_len*3/4):airbnb_len] # given a predicted price

def predict_price(rooms):
    # get all airbnb with same room count and then calc. avg. price
    training_matches = airbnb_training.loc[airbnb_training['bedrooms'] == rooms]
    return training_matches['price'].mean()

airbnb_testing['predicted_price'] = airbnb_testing['bedrooms'].apply(lambda x: predict_price(x))

airbnb_testing['difference'] = airbnb_testing.apply(lambda row: abs(row['price'] - row['predicted_price']), axis=1)
mae = airbnb_testing['difference'].mean() # error measurement
airbnb_testing['squared_difference'] = airbnb_testing['difference'].apply(lambda difference: difference ** 2) # alternative error measurement to penalize distant values more
mse = airbnb_testing['squared_difference'].mean()
rmse = np.sqrt(mse)
print(mae)
print(mse)
print(rmse)
print(airbnb_testing)
