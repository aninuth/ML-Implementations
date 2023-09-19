from matplotlib import pyplot as plt
import numpy as np


def plot_boundary(clf, x, y):
    """
    Plot the decision boundary of the kernel perceptron, and the samples (using different
    colors for samples with different labels)
    """
    # create a meshgrid of points
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    r1,r2 = xx.flatten(), yy.flatten()
    r1,r2 = r1.reshape(len(r1),1), r2.reshape(len(r2), 1)
    grid = np.hstack((r1,r2))
    #print("length: " + str(len(grid)))
    # predict the labels for the meshgrid
    #print("predicting for graph now")
    Z = clf.predict(grid)

    # plot the decision boundary using the predictions
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # plot the samples with different colors for different labels
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.coolwarm)

    plt.savefig('viz.png', bbox_inches='tight')



