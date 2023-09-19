from nis import match
import numpy as np

#build my own Gaussian Discriminant
class GaussianDiscriminant:
    def __init__(self, k=2, d=8, priors=None, shared_cov=False):
        self.mean = np.zeros((k, d))  # mean
        self.shared_cov = (
            shared_cov  # using class-independent covariance or not
        )
        if self.shared_cov:
            self.S = np.zeros((d, d))  # class-independent covariance
        else:
            self.S = np.zeros((k, d, d))  # class-dependent covariance
        if priors is not None:
            self.p = priors
        else:
            self.p = [
                1.0 / k for i in range(k)
            ]  # assume equal priors if not given
        self.k = k
        self.d = d

    def fit(self, Xtrain, ytrain):
        # compute the mean for each class
        c1_train_step = 0
        c1_train_sum=0
        c2_train_step = 0
        c2_train_sum=0

        for i in range(len(self.mean[0])):
            for j in range(len(Xtrain)):
                if (ytrain[j] == 1):
                    c1_train_sum += Xtrain[j][i]
                    c1_train_step += 1
                else:
                    c2_train_sum += Xtrain[j][i]
                    c2_train_step += 1
            
            self.mean[0][i] = c1_train_sum / c1_train_step
            self.mean[1][i] = c2_train_sum / c2_train_step
            c1_train_step = 0
            c1_train_sum = 0
            c2_train_step = 0
            c2_train_sum = 0

        if self.shared_cov:
            # compute the class-independent covariance
            self.S = np.transpose(np.cov(np.transpose(Xtrain), ddof=0))
        else:
            # compute the class-dependent covariance
            c1_list = []
            c2_list = []

            for i in range(len(Xtrain)):
                if ytrain[i] == 1: 
                    c1_list.append(Xtrain[i])
                else:
                    c2_list.append(Xtrain[i])
            

            self.S[0] = np.transpose(np.cov(np.transpose(c1_list), ddof=0))
            self.S[1] = np.transpose(np.cov(np.transpose(c2_list), ddof=0))

    def predict(self, Xtest):
        # predict function to get predictions on test set
        predicted_class = np.ones(Xtest.shape[0])  # placeholder
        for i in np.arange(Xtest.shape[0]):  # for each test set example
            c1_disc = 0
            c2_disc = 0

            # calculate the value of discriminant function for each class
            for c in np.arange(self.k):
                if self.shared_cov:
                    s_det = np.linalg.det(self.S)
                    t2_sub = np.subtract(Xtest[i], self.mean[c])
                    s_inverse = np.linalg.inv(self.S)
                    term_2 = np.dot(np.dot(np.transpose(t2_sub), s_inverse), t2_sub)

                    if c == 0:
                        c1_disc = -0.5 * np.log(s_det) - 0.5*term_2 + np.log(self.p[c])
                    
                    else: 
                        c2_disc = -0.5 * np.log(s_det) - 0.5*term_2 + np.log(self.p[c]) 


                else:
                    s_det = np.linalg.det(self.S[c])
                    t2_sub = np.subtract(Xtest[i], self.mean[c])
                    s_inverse = np.linalg.inv(self.S[c])
                    term_2 = np.dot(np.dot(np.transpose(t2_sub), s_inverse), t2_sub)

                    if c == 0:
                        c1_disc = -0.5 * np.log(s_det) - 0.5*term_2 + np.log(self.p[c])
                    
                    else: 
                        c2_disc = -0.5 * np.log(s_det) - 0.5*term_2 + np.log(self.p[c])
            # determine the predicted class based on the values of discriminant function
            # remember to return 1 or 2 for the predicted class
            if c1_disc > c2_disc:
                predicted_class[i] = 1
            else:
                predicted_class[i] = 2
        return predicted_class
