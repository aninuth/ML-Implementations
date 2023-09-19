import numpy as np

class NaiveBayes:
    def __init__(self, X_train, y_train):
        self.n = X_train.shape[0] # size of the dataset
        self.d = X_train.shape[1] # size of the feature vector
        self.K = len(set(y_train)) # size of the class set

        # these are the shapes of the parameters
        self.psis = np.zeros([self.K, self.d]) 
        self.phis = np.zeros([self.K])

    def fit(self, X_train, y_train):

        # we now compute the parameters
        for k in range(self.K):
            X_k = X_train[y_train == k]
            self.phis[k] = self.get_prior_prob(X_k) # prior
            self.psis[k] = self.get_class_likelihood(X_k) # likelihood

        # clip probabilities to avoid log(0)
        self.psis = self.psis.clip(1e-14, 1-1e-14)

    def predict(self, X_test):
        # compute log-probabilities
        n,d = X_test.shape
        X_test = np.reshape(X_test, (1,n,d))
        psis = np.reshape(self.psis, (self.K, 1, d))
        psis = psis.clip(1e-14, 1-1e-14)

        logy = np.log(self.phis).reshape([self.K,1])
        logxy = X_test * np.log(psis) + (1-X_test) * np.log(1-psis)
        logyx = logxy.sum(2) + logy

        return logyx.argmax(0).flatten()
    def get_prior_prob(self, X):
        # compute the prior probability of class k 
        return len(X)/self.n

    def get_class_likelihood(self, X):
        # estimate Bernoulli parameter theta for each feature for each class
        return np.mean(X, 0)