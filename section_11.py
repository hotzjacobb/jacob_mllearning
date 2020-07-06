import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from sklearn.linear_model import LinearRegression
import seaborn
import matplotlib.pyplot as plt

housing_ds = pd.read_csv('AmesHousing.txt', delimiter='\t')
housing_ds = housing_ds.drop(['PID', 'Year Built', 'Year Remod/Add', 'Garage Yr Blt', 'Mo Sold', 'Yr Sold'], axis=1)
missing_vals = housing_ds.isnull().apply(any)

# missing_vals.dropna()
missing_vals = missing_vals[missing_vals == True]

housing_ds = housing_ds.drop(missing_vals.index.values, axis=1)
housing_ds = housing_ds.select_dtypes(include=['int64', 'float64'])

# lr = LinearRegression()
# prediction_columns = housing_ds.drop('SalePrice', axis=1).columns.values # all but price
# lr.fit(housing_ds[prediction_columns], housing_ds[['SalePrice']])

correlations = housing_ds.corr().loc['SalePrice'].apply(lambda x: np.abs(x)).sort_values()
correlations = correlations[correlations > .3]
housing_corr = housing_ds[correlations.index].corr()
heatmap = seaborn.heatmap(housing_corr)
print(heatmap)

correlations = correlations.drop(['TotRms AbvGrd', 'SalePrice']) # drop Tot b/c correlated and 
# SalePrice b/c it's the target


features = correlations.index.values

training_data = housing_ds.iloc[:int(len(housing_ds)/2)]
testing_data = housing_ds.iloc[int(len(housing_ds)/2):]

lr = LinearRegression()
lr.fit(training_data[features], training_data[['SalePrice']]) 
train_predictions = lr.predict(training_data[features])
test_predictions = lr.predict(testing_data[features])

train_error = np.sqrt(metrics.mean_squared_error(training_data['SalePrice'], train_predictions))
test_error = np.sqrt(metrics.mean_squared_error(testing_data['SalePrice'], test_predictions))
print(train_error)
print(test_error)