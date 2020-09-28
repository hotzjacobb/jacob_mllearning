import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

housing_ds = pd.read_csv('AmesHousing.txt', delimiter='\t')
train = housing_ds.iloc[:1460,]
test = housing_ds.iloc[1460:,]

features = ['Wood Deck SF', 'Fireplaces', 'Full Bath', '1st Flr SF', 
            'Garage Area', 'Gr Liv Area', 'Overall Qual']
        
x_transpose = np.transpose(train[features])
intermed_matrix = np.dot(x_transpose, train[features])
intermed_matrix = np.linalg.inv(intermed_matrix)
intermed_matrix = np.dot(intermed_matrix, x_transpose)
parameters = np.dot(intermed_matrix, train['SalePrice'])
print(parameters)