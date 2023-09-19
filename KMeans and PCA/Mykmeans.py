
import numpy as np


class Kmeans:
    def __init__(self, k=3):  # k is number of clusters
        self.num_cluster = k
        self.center = None
        self.error_history = []

    def run_kmeans(self, X, y):
        # initialize the centers of clutsers as a set of pre-selected samples
        init_idx = [
            500, 1500, 2500,
        ] 
        self.center = X[init_idx]
        num_iter = 0  # number of iterations for convergence

        # initialize cluster assignment
        prev_cluster_assignment = np.zeros([len(X),]).astype("int")
        cluster_assignment = np.zeros([len(X),]).astype("int")
        is_converged = False

        # iteratively update the centers of clusters till convergence
        while not is_converged:
            # iterate through the samples and compute their cluster assignment (E step)
            for i in range(len(X)):
                # use euclidean distance to measure the distance between sample and cluster centers
                # determine the cluster assignment by selecting the cluster whose center is closest to the sample
                distances = [np.linalg.norm(X[i] - self.center[k]) for k in range(self.num_cluster)]
                cluster_assignment[i] = np.argmin(distances)
            # update the centers based on cluster assignment (M step)
            for k in range(self.num_cluster):
                ssum = 0
                count = 0 
                for i in cluster_assignment:
                    if i==k:
                        ssum += X[i]
                        count += 1
                if count != 0:
                    self.center[k] = ssum/count

            # compute the reconstruction error for the current iteration
            cur_error = self.compute_error(X, cluster_assignment)
            self.error_history.append(cur_error)

            # reach convergence if the assignment does not change anymore
            is_converged = True if (cluster_assignment==prev_cluster_assignment).sum() == len(X) else False
            prev_cluster_assignment = np.copy(cluster_assignment)
            num_iter += 1

        return num_iter, self.error_history, cluster_assignment, self.center

    def compute_error(self,X,cluster_assignment):
        # compute the reconstruction error for given cluster assignment and centers
        error = 0 # placeholder
        for i in range(len(cluster_assignment)):
            error += np.linalg.norm(X[i] - self.center[cluster_assignment[i]])

        return error

    def params(self):
        return self.center