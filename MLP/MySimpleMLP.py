import numpy as np

def preprocess_data(data, mean=None, std=None):
    if mean is not None or std is not None: #case where they are precomputed w training data

        data = (data-mean)/std
        return data
    else:
        # compute the mean and std based on the training data
        mean = std = 0
        mean = np.mean(data)
        std = np.std(data)
        return data, mean, std

def preprocess_label(label):
    # to handle the loss function computation, convert the labels into one-hot vector for training
    one_hot = np.zeros([len(label), 10])
    for i in range(len(label)):
        one_hot[i][label[i]] = 1
    return one_hot

def sigmoid(x):
    # implement the sigmoid activation function for hidden layer
    f_x = 1/(1+ np.exp(-x))  
    return f_x

def Relu(x):
    # implement the Relu activation function for hidden layer
    f_x = np.maximum(0,x)

    return f_x

def tanh(x):
    # implement the tanh activation function for hidden layer
    f_x = f_x = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return f_x


def softmax(x):
    # implement the softmax activation function for output layer
    f_x = []

    for i in range(len(x)):
        expp_1 = np.exp(x[i])
        soft = expp_1 / np.sum(expp_1)
        f_x.append(soft)

    f_x = np.reshape(f_x, [len(x), len(x[0])])
    return f_x

class MLP:
    def __init__(self, num_hid, activation="Relu"):
        # initialize the weights
        np.random.seed(2022)
        self.weight_1 = np.random.random([64, num_hid]) / 100
        self.bias_1 = np.random.random([1, num_hid]) / 100
        self.weight_2 = np.random.random([num_hid, 10]) / 100
        self.bias_2 = np.random.random([1, 10]) / 100
        self.num_hid = num_hid

        #Use the activation function provided in function call
        self.activation = activation

    def fit(self, train_x, train_y, valid_x, valid_y):
        # initialize learning rate
        lr = 5e-4
        # initialize the counter of recording the number of epochs that the model does not improve
        # and log the best validation accuracy
        count = 0
        best_valid_acc = 0
        epoch_count = 0


        while count <= 100: # 100 iterations of no improvement will stop learning 
            z__h = np.dot(train_x, self.weight_1) + self.bias_1
            z = tanh(z__h)  # 1000 x 4

            y = np.dot(z, self.weight_2) + self.bias_2  
            #probability = softmax(y)
            if self.activation == "Relu":
                probability = Relu(y)
            elif self.activation == "Softmax":
                probability = softmax(y)
            elif self.activation == "Sigmoid":
                probability = sigmoid(y)
            else:
                probability = tanh(y)
            train_y = train_y.reshape(1000, 10)
            sub = np.subtract(train_y, probability)

            #Backpropagation
            d_v = lr * (np.dot(z.T, sub))
            b2__d1 = lr * sub
            b2__d1 = np.sum(b2__d1, axis=0)
            b2__d1 = np.reshape(b2__d1, [1, 10])

            w1 = np.dot(sub, self.weight_2.T)
            w2 = w1 * (1 - np.power(z, 2))
            w3 = np.dot(train_x.T, w2)

            w__d = lr * w3

            b__1_d = lr * w2
            b__1_d = np.sum(b__1_d, axis=0)
            b__1_d = np.reshape(b__1_d, [1, self.num_hid])

            #Update parameters using backprop
            self.weight_2 += d_v
            self.bias_2 += b2__d1
            self.weight_1 += w__d
            self.bias_1 += b__1_d
            epoch_count += 1

            # evaluate the accuracy on the validation data
            predictions = self.predict(valid_x)
            cur_valid_acc = (predictions.reshape(-1) == valid_y.reshape(-1)).sum() / len(valid_x)

            # compare the current validation accuracy, if cur_valid_acc > best_valid_acc, we will increase count by 1
            if cur_valid_acc > best_valid_acc:
                best_valid_acc = cur_valid_acc
                count = 0
            else:
                count += 1

        return best_valid_acc

    def predict(self, x):
        y = np.zeros([len(x), ]).astype('int')  # placeholder for now
        z = np.add(np.dot(x, self.weight_1), self.bias_1)  # 1000 x 4 array
        z = tanh(z)  # 1000 x 4 array
        y_soft = np.add(np.dot(z, self.weight_2), self.bias_2)  # 1000 x 10 array
        probability = softmax(y_soft)

        for t in range(len(x)):
            try:
                value = np.hstack(np.where(probability[t] == probability[t].max()))
                y[t] = value[0]
            except:
                value = np.array([0])
                y[t] = value[0]

        return y

    def get_hidden(self, x):
        #hidden layer features
        z = x  
        zh = np.dot(x, self.weight_1)  # 1000 x 4
        z = np.tanh(zh)  # 1000 x 4
        return z

    def params(self):
        return self.weight_1, self.bias_1, self.weight_2, self.bias_2
