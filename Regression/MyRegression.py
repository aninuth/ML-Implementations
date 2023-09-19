import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Building my own regression model
def MyRegression(X, y, split, order = 2):
    # initialize the error dict, where the key is the k-th fold, and the error_dict[k] is the mean square error of
    # the test sample in the k-th fold.
    error_dict = {}
    for k in range(10):
        error_dict[k] = -1

    for k in range(10):
        # select the training set where the split value is not k 
        train = []
        train_labels = []
        test = []
        test_labels = []
        for i in range(len(split)):
            if split[i] == k:
                test.append(X[i])
                test_labels.append(y[i])
            else:
                train.append(X[i])
                train_labels.append(y[i])
        # select the test set where the split value is k
        poly = PolynomialFeatures(order)
        train = poly.fit_transform(train)
        poly.fit(train, train_labels)
        poly2 = PolynomialFeatures(order)
        test = poly.fit_transform(test)
        poly2.fit(test, test_labels)
        # build the regression model
        reg = LinearRegression()
        reg.fit(train, train_labels)
        # predict the test_X
        predictions = reg.predict(test)
        
        # calculate and record the mean square error
        error = [predictions[i] - test_labels[i] for i in range(len(test))]
        error_dict[k] = float(sum(error)/len(error))

    return error_dict


def VisualizeError(error_related_to_order):
    pass
    # collect the order information
    d = {}
    for k,v in error_related_to_order.items():
        d[k] = sum(v)/len(v)
    # collect the mse and calculate the mean corresponding of different order of polynimal
    lists = sorted(d.items())
    x,y = zip(*lists)

    # Plotting
    plt.plot(x,y)
    plt.savefig('regression.png')
    


