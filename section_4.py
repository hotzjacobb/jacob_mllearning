import pandas as pd
import numpy as np
np.random.seed(1)

paris_listings = pd.read_csv("paris_airbnb.csv")
paris_listings = paris_listings.loc[np.random.permutation(len(paris_listings))]
stripped_commas = paris_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')
paris_listings['price'] = stripped_dollars.astype('float')
train_df = paris_listings.iloc[0:6000]
test_df = paris_listings.iloc[6000:]

def predict_price(new_listing):
    temp_df = train_df.copy()
    temp_df['distance'] = temp_df['accommodates'].apply(lambda x: np.abs(x - new_listing))
    temp_df = temp_df.sort_values('distance')
    nearest_neighbor_prices = temp_df.iloc[0:5]['price']
    predicted_price = nearest_neighbor_prices.mean()
    return(predicted_price)

test_df['predicted_price'] = test_df['accommodates'].apply(lambda x: predict_price(x))
print(test_df['predicted_price'])