import numpy as np

def PCA(X,num_dim=None):
    #finding the projection matrix that maximize the variance 
    Sigma = X.T.dot(X) / X.shape[0] # form covariance matrix
    L, Q = np.linalg.eigh(Sigma) # perform eigendecomposition
    
    # reverse to order in descending
    L = np.flip(L)
    Q = np.flip(Q)

    W = []
    if num_dim != None:
        W = Q[:,:num_dim] # get top p eigenvectors
    else:
        sum_eigenvalues = np.sum(L)

        # capture >90% variance
        i = 0
        cum_var = 0
        while cum_var <= 0.9:
            W.append(Q[i])
            cum_var += L[:i]/sum_eigenvalues
            i += 1
        num_dim = i
        W = np.array(W).T

    #W = np.array(W).T
    Z = X.dot(W) # project on these eigenvectors
    return Z, num_dim