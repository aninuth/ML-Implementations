
import numpy as np

#Building my perceptron
def MyPerceptron(X, y, w0=[0.1,-1.0]):
    #calculate iterations it takes for w to converge
    iter_value = 0
    w = w0
    error_rate = 1.00
    w_prev = 0

    # update w

    while abs(sum(w_prev-w)) > 0.00001:
        w_prev = w[:]
        iter_value += 1
        for t in range(len(X)):
            if (y[t] * np.dot(w,X[t])  <= 0):
                w += y[t] * X[t]

    # compute the error rate
    for i in range(len(X)):
        if (y[t] * np.dot(w,X[t])  <= 0):
            error_rate += 1    
    error_rate = error_rate/len(X)

    return (w, iter_value, error_rate)
