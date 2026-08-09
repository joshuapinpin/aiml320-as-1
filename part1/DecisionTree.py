import numpy as np

from Node import Node

class DecisionTree:
    """
    A decision tree classifier for binary classification problems.
    """

    def __init__(self, min_samples=2, max_depth=2, ig_threshold=0.00001):
        """
        Constructor for DecisionTree class.

        Parameters:
            min_samples (int): Minimum number of samples required to split an internal node.
            max_depth (int): Maximum depth of the decision tree.
            ig_threshold (float): Threshold for information gain to determine if a split is valid.
        """
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.ig_threshold = ig_threshold

    def split_data(self, dataset, feature):
        """
        Splits the given dataset into two datasets based on the given feature and threshold.

        Parameters:
            dataset (ndarray): Input dataset.
            feature (int): Index of the feature to be split on.

        Returns:
            left_dataset (ndarray): Subset of the dataset with values equal to the chosen category.
            right_dataset (ndarray): Subset of the dataset with values not equal to the chosen category.
        """
        # Create empty arrays to store the left and right datasets
        left_dataset = []
        right_dataset = []

        # Loop over each row in the dataset and split based on the given feature
        for row in dataset:
            value = None  # TODO
            if row[feature] == value:
                left_dataset.append(row)
            else:
                right_dataset.append(row)

        # Convert the left and right datasets to numpy arrays and return
        left_dataset = np.array(left_dataset)
        right_dataset = np.array(right_dataset)
        return left_dataset, right_dataset

    def entropy(self, y):
        """
        Computes the entropy of the given label values.

        Parameters:
            y (ndarray): Input label values.

        Returns:
            entropy (float): Entropy of the given label values.
        """

        y = np.asarray(y)
        if len(y) == 0:
            return 0.0

        # Count occurrences of each class label
        _, counts = np.unique(y, return_counts=True)

        # Convert counts to proportions
        proportions = counts / counts.sum()

        # Sum of -p * log2(p) over all classes with p > 0
        entropy = -np.sum(proportions * np.log2(proportions))

        # Return the final entropy value
        return entropy

    def information_gain(self, parent, left, right):
        """
        Computes the information gain from splitting the parent dataset into two datasets.

        Parameters:
            parent (ndarray): Input parent dataset.
            left (ndarray): Subset of the parent dataset after split on a feature.
            right (ndarray): Subset of the parent dataset after split on a feature.

        Returns:
            information_gain (float): Information gain of the split.
        """
        # TODO
        information_gain = None  # Placeholder

        return information_gain

    def best_split(self, dataset, num_samples, num_features):
        """
        Finds the best split for the given dataset.

        Args:
        dataset (ndarray): The dataset to split.
        num_samples (int): The number of samples in the dataset.
        num_features (int): The number of features in the dataset.

        Returns:
        dict: A dictionary with the best split feature index, threshold, gain,
              left and right datasets.
        """
        best_split_gain = -1
        for feature_index in range(num_features):
            # TODO get the feature values
            # get left and right datasets
            feature_values = None  # TODO: indexed with feature_index
            left_dataset, right_dataset = self.split_data(dataset, feature_values)

            # check if either datasets is empty
            if len(left_dataset) and len(right_dataset):
                # get y values of the parent and left, right nodes
                y, left_y, right_y = dataset[:, -1], left_dataset[:, -1], right_dataset[:, -1]

                # compute information gain based on the y values
                information_gain = self.information_gain(y, left_y, right_y)

                # update the best split if conditions are met
                if information_gain > best_split_gain:
                    # TODO: update the best_split_gain and the corresponding feature and threshold
                    pass

    def build_tree(self, dataset, current_depth=0):
        """
        Recursively builds a decision tree from the given dataset.

        Args:
        dataset (ndarray): The dataset to build the tree from.
        current_depth (int): The current depth of the tree.

        Returns:
        Node: The root node of the built decision tree.
        """
        # TODO
        leaf_value = None  # Placeholder

        # return leaf node value
        return Node(value=leaf_value)

    def fit(self, X, y):
        """
        Builds and fits the decision tree to the given X and y values.

        Args:
        X (ndarray): The feature matrix.
        y (ndarray): The target values.
        """
        dataset = np.concatenate((X, y), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, X):
        """
        Predicts the class labels for each instance in the feature matrix X.

        Args:
        X (ndarray): The feature matrix to make predictions for.

        Returns:
        list: A list of predicted class labels.
        """
        # TODO
        predictions = []  # Placeholder
        return predictions

# ================================================================================================================
# ================================================================================================================

def check_entropy():
    dt = DecisionTree()
    print(f"Entropy of [0,0.1,1] should be 1.0: {dt.entropy([0,0,1,1])}")
    print(f"Entropy of [1,1,1,1] should be 0.0: {dt.entropy([1,1,1,1])}")


def main():
    check_entropy()


if __name__ == "__main__":
    main()
