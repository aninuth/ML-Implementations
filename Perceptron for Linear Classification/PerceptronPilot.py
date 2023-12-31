# Header
import numpy as np
from matplotlib import pyplot as plt
from MyPerceptron import MyPerceptron

# Data loading
data = np.genfromtxt('perceptronData.csv', delimiter=',')
# feature
X = data[:, :2]
# label
y = data[:, 2]

# Initialize the weight w0
w0 = np.array([0.1, -1.0])

# plot the figure
fig = plt.figure()
plt.plot(X[y < 0, 0], X[y < 0, 1], 'ro')
plt.plot(X[y > 0, 0], X[y > 0, 1], 'bo')

# decision boundary with w0
plt.plot([-w0[1], w0[1]], [w0[0], -w0[0]], 'k-')

# Adjust x-y scale the same and show grid lines
plt.axis('equal')
plt.grid(True)
# add xlabel, ylabel, title
plt.xlabel("x-axis")
plt.xlabel("y-axis")
plt.title("Initial w0")

# save plot into a file
fig.set_size_inches(10, 10)
fig.savefig('initial_result_w0.png') # saving the figure for initialized value
plt.savefig("initial_result_2.png")

# the following figure is used to show the result for w
fig2 = plt.figure()
plt.clf()

# plot the new figure with the samples
plt.plot(X[y < 0, 0], X[y < 0, 1], 'ro')
plt.plot(X[y > 0, 0], X[y > 0, 1], 'bo')

# Get the corresponding final weights, the number of iteration and misclassification rate
w, iter, error_rate = MyPerceptron(X, y, w0)
print("The final weights we obtain are: ", w)

# Plot the boundary decision for final w
plt.plot([-w[1], w[1]]/np.max(abs(w)), [w[0], -w[0]]/np.max(abs(w)), 'k-')
# add xlabel, ylabel, title
plt.xlabel("x-axis")
plt.xlabel("y-axis")
plt.title("Final w")

# Plotting
plt.axis('equal')
plt.grid(True)
fig2.set_size_inches(10, 10)
fig2.savefig('perceptron_result.png') # saving the final figure
plt.show()

# Print the final result
print('Error Rate for the Perceptron Algorithm: %.2f' %error_rate)
print('Number of iterations for convergence: %d' %iter)
