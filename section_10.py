import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

housing_ds = pd.read_csv('AmesHousing.txt', delimiter='\t')
training_data = housing_ds.iloc[0:1460]
testing_data = housing_ds.iloc[1460:]
target = 'SalePrice'
# training_data.plot(x='Overall Cond', y=target, kind='scatter')
# plt.show()
# print(training_data[['SalePrice', 'Overall Cond', 'Gr Liv Area', 'Garage Area']].corr())
lr = LinearRegression()
lr.fit(training_data[['Gr Liv Area']], training_data[target])
coefficient = lr.coef_
y_int = lr.intercept_
print(coefficient)
print(y_int)
pre_price_train = training_data.loc[:, 'Gr Liv Area'] * coefficient + y_int
pre_price_test = testing_data.loc[:, 'Gr Liv Area'] * coefficient + y_int
# alternatively lr.predict(training_data[['Gr Liv Area']])
training_err = np.sqrt(metrics.mean_squared_error(training_data['SalePrice'], 
               pre_price_train))
testing_err = np.sqrt(metrics.mean_squared_error(testing_data['SalePrice'], 
               pre_price_test))
print(testing_err)
print(training_err)
# now use two varriable model
lr.fit(training_data[['Gr Liv Area', "Overall Cond"]], training_data[target])
two_var_pred_train = lr.predict(training_data[['Gr Liv Area', "Overall Cond"]])
two_var_pred_test = lr.predict(testing_data[['Gr Liv Area', "Overall Cond"]])
training_err_two = np.sqrt(metrics.mean_squared_error(training_data['SalePrice'], 
               two_var_pred_train))
testing_err_two = np.sqrt(metrics.mean_squared_error(testing_data['SalePrice'], 
               two_var_pred_test))
print(training_err_two)
print(testing_err_two)
