import pandas as pd
import numpy as np

housing_ds = pd.read_csv('AmesHousing.txt', delimiter='\t')
train = housing_ds[:1460]
test = housing_ds[1460:]

# returns the derivative of the cost func. given a weight(point) 
# and prediction and target value lists
def derivative(point, xi_list, yi_list):
    coeff = (2 / len(xi_list))
    sum = 0
    for i in range(len(xi_list)):
        xi = xi_list[i]
        yi = yi_list[i]
        inner_term = xi * (point * xi - yi)
        sum += inner_term
    deriv = coeff * sum
    return deriv

# returns a weight after performing a 2D gradient descent
def gradient_desc(xi_list, yi_list, init_point, alpha, max_iter):
    curr_point = init_point
    points = [curr_point]
    for i in range(max_iter):
        deriv = derivative(curr_point, xi_list, yi_list)
        new_point = curr_point - alpha * deriv
        points.append(new_point)
    return(points[max_iter-1])

coeff_weight = gradient_desc(train['Gr Liv Area'], train['SalePrice'], 150, .0000003, 20)
print(coeff_weight)