import numpy as np



def split_data(data, label, feature):

    left_data = []
    left_label = []
    right_data = []
    right_label = []

    for i in range(len(data)):
        if data[i][feature] == 0:
            left_data.append(data[i])
            left_label.append(label[i])
        else: 
            right_data.append(data[i])
            right_label.append(label[i])

    return left_data, left_label, right_data, right_label


# Class for decision tree node
class Tree_node:

    def __init__(
        self,
    ):
        self.is_leaf = False  # whether or not the current node is a leaf node
        self.feature = None  # index of the selected feature (for non-leaf node)
        self.label = None  # class label (for leaf node)
        self.left_child = None  # left child node
        self.right_child = None  # right child node

# Class for entire decision tree
class Decision_tree:

    def __init__(self, min_entropy):
        self.min_entropy = min_entropy
        self.root = None

    def fit(self, train_x, train_y):
        # construct the decision-tree with recursion
        self.root = self.generate_tree(train_x, train_y, self.min_entropy)

    def predict(self, test_x):
        # iterate through all samples
        prediction = np.zeros([len(test_x),]).astype(
            "int"
        )  # placeholder for initial prediction
        for i in range(len(test_x)):
            # traverse the decision-tree based on the features of the current sample
            cur_node = self.root
            while cur_node.feature is not None:
                if test_x[i, cur_node.feature] == 0:
                    cur_node = cur_node.left_child
                else:
                    cur_node = cur_node.right_child
            prediction[i] = cur_node.class_label
        return prediction
    

    def compute_node_entropy(self, label):
        # compute the entropy of a tree node (add 1e-15 inside the log2 when computing the entropy to prevent numerical issue)
        unique_labels, counts = np.unique(label, return_counts = True)

        total_count = len(label)
        node_entropy = 0  # placeholder

        for count in counts: 
            p = count/total_count
            node_entropy -= p*np.log2(p+1e-15)
        return node_entropy

    def compute_split_entropy(self, left_y, right_y):
        # compute the entropy of a potential split, left_y and right_y are labels for the two splits
        split_entropy = -1  # placeholder
        left_count = np.size(left_y)
        right_count = np.size(right_y)
        total_count = left_count + right_count

        left_entropy = self.compute_node_entropy(left_y)
        right_entropy = self.compute_node_entropy(right_y)
        split_entropy = (left_count/total_count) * left_entropy + (right_count/total_count) * right_entropy
        return split_entropy
    

    
    def select_feature(self, data, label):
        # iterate through all features and compute their corresponding entropy
        best_feat = 0
        best_entropy = 100
        for i in range(len(data[0])):
            left_data, left_label, right_data, right_label = split_data(data, label, i)
            # compute the entropy of splitting based on the selected features
            cur_entropy = self.compute_split_entropy(
                left_label, right_label
            )  

            # select the feature with minimum entropy
            if cur_entropy < best_entropy:
                best_entropy = cur_entropy
                best_feat = i 
        return best_feat

    def generate_tree(self, data, label, min_entropy):
        # initialize the current tree node
        cur_node = Tree_node()

        # compute the node entropy
        node_entropy = self.compute_node_entropy(label)

        # determine if the current node is a leaf node
        if node_entropy < min_entropy:
            # determine the class label for leaf node
            cur_node.class_label = np.bincount(label).argmax()
            return cur_node

        # select the feature that will best split the current non-leaf node
        selected_feature = self.select_feature(data, label)
        cur_node.feature = selected_feature

        # split the data based on the selected feature and start the next level of recursion
        left_x, left_y, right_x, right_y = split_data(data, label, selected_feature)
#        print("left data: " + str(np.shape(left_x)) + " and riht data: " + str(np.shape(right_x)))
        cur_node.left_child = self.generate_tree(left_x, left_y, min_entropy)
        cur_node.right_child = self.generate_tree(right_x, right_y, min_entropy)
        return cur_node

    

    

    

