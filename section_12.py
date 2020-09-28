import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

housing_ds = pd.read_csv('AmesHousing.txt', delimiter='\t')
train = housing_ds.iloc[:1460,]
test = housing_ds.iloc[1460:,]

# returns the part. derivative of the cost func. given a value for the coeff.,
# and prediction and target value lists
def coeff_derivative(weight, targ_intercept, xi_list, yi_list):
    coeff = (2 / len(xi_list))
    sum = 0
    for i in range(len(xi_list)):
        xi = xi_list[i]
        yi = yi_list[i]
        inner_term = xi * ((weight * xi + targ_intercept) - yi)
        sum += inner_term
    deriv = coeff * sum
    return deriv

# returns the part. derivative of the cost func. given a value for the y-int, 
# and prediction and target value lists
def y_int_derivative(targ_intercept, weight, xi_list, yi_list):
    factor = (2 / len(xi_list))
    sum = 0
    for i in range(len(xi_list)):
        xi = xi_list[i]
        yi = yi_list[i]
        inner_term = (weight * xi + targ_intercept) - yi
        sum += inner_term
    deriv = factor * sum
    return deriv   

# returns a coeff. and y-int after performing a 2D gradient descent for a linear fit
# alpha is the speed factor for the coeff.
# beta is the speed factor for the y-int; if not provided is set to the same as alpha
def gradient_desc(xi_list, yi_list, init_coeff, init_y_int, max_iter, alpha, beta=None):
    if beta is None:
        beta = alpha
    curr_coeff = init_coeff
    curr_y_int = init_y_int
    params_for_line = [(init_coeff, init_coeff)]
    for i in range(max_iter):
        coeff_deriv = coeff_derivative(curr_coeff, curr_y_int, xi_list, yi_list)
        new_coeff = curr_coeff - alpha * coeff_deriv
        y_int_deriv = y_int_derivative(curr_y_int, curr_coeff, xi_list, yi_list)
        new_y_int = curr_y_int - beta * y_int_deriv
        params_for_line.append((new_coeff, new_y_int))
    return(params_for_line[max_iter-1])

coeff_weight = gradient_desc(train['Gr Liv Area'], train['SalePrice'], 150, 1000, 20, .0000003)
print(coeff_weight)
test['Pred Price'] = coeff_weight[0] * test['Gr Liv Area'] + coeff_weight[1]
rme = np.sqrt(metrics.mean_squared_error(test['SalePrice'], test['Pred Price']))
print(rme)