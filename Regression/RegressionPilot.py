# Header
import numpy as np
from matplotlib import pyplot as plt
from MyRegression import MyRegression, VisualizeError

# Data loading
data = np.genfromtxt('RegressionData.csv', delimiter=',')
X = data[:,:1]
y = data[:,1:2]
split = data[:,2:3].astype(np.int32)

# sub problem 1
error_related_to_order = {}
# check for different order
for order in [2, 3, 4, 5, 6]:
    mse = MyRegression(X, y, split, order)
    error_related_to_order[order] = list(mse.values())

print(error_related_to_order)

# sub problem 2
VisualizeError(error_related_to_order)
