import numpy as np
import sys

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


    # ========================================
    # --- 1. Entropy and Information Gain  ---
    # ========================================

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

    def information_gain(self, parent_y, children_y):
        """
        Computes the information gain from splitting the parent labels into
        an arbitrary number of child groups (one per distinct feature value).

        Parameters:
            parent_y (array-like): Labels before the split.
            children_y (list of array-like): Labels of each child group after the split.
                E.g. for a feature with values {0, 1, 2}, this would be
                [y_where_feature==0, y_where_feature==1, y_where_feature==2].

        Returns:
            information_gain (float): IG of this split.
        """
        parent_y = np.asarray(parent_y)
        n = len(parent_y)
        if n == 0:
            return 0.0

        parent_entropy = self.entropy(parent_y)

        weighted_child_entropy = 0.0
        for child_y in children_y:
            child_y = np.asarray(child_y)
            if len(child_y) == 0:
                continue # empty branches contribute 0 weight, skip safely
            weight = len(child_y) / n
            weighted_child_entropy += weight * self.entropy(child_y)

        information_gain = parent_entropy - weighted_child_entropy
        return information_gain

    # ========================================
    # --- 2. Data Splitting  ---
    # ========================================

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
        feature_column = dataset[:, feature]
        unique_values = np.unique(feature_column)

        splits = {}
        for value in unique_values:
            splits[value] = dataset[feature_column == value]

        return splits

    def best_split(self,  dataset, features_available):
        """
        Finds the best feature to split on for the given dataset, out of the
        features still available at this point in the tree.

        Args:
            dataset (ndarray): The dataset at this node (last column = labels).
            features_available (list[int]): Indices of features not yet used
                on the path from the root to this node.

        Returns:
            dict with keys:
                "feature": index of the best feature to split on (or None if no
                           feature gives a useful split)
                "gain": the information gain of that split
                "entropy": the entropy of `dataset` at this node (parent entropy)
                "splits": dict {value: subset_ndarray} for the chosen feature,
                          as returned by self.split_data
        """
        parent_y = dataset[:, -1]
        parent_entropy = self.entropy(parent_y)

        best = {"feature": None, "gain": -1.0, "entropy": parent_entropy, "splits": None}

        for feature_index in features_available:
            splits = self.split_data(dataset, feature_index)

            # If every remaining instance has the same value for this feature,
            # splitting on it achieves nothing - skip it.
            if len(splits) < 2:
                continue

            children_y = [subset[:, -1] for subset in splits.values()]
            gain = self.information_gain(parent_y, children_y)

            if gain > best["gain"]:
                best["feature"] = feature_index
                best["gain"] = gain
                best["splits"] = splits

        return best

    # ========================================
    # --- 3. Tree Building, Fitting, and Prediction ---
    # ========================================

    def _class_counts(self, y):
        """
        Returns {class_label: count} for the given labels. Uses self.classes_
        if it's been set (via fit), otherwise falls back to whatever classes
        appear in y — lets you test build_tree() standalone without calling
        fit() first.
        """
        classes = getattr(self, "classes_", None) or sorted(np.unique(y))
        counts = {c: 0 for c in classes}
        values, freqs = np.unique(y, return_counts=True)
        for v, f in zip(values, freqs):
            counts[v] = int(f)
        return counts

    def _majority_class(self, class_counts):
        return max(class_counts, key=class_counts.get)

    def build_tree(self, dataset, features_available, current_depth=0):
        """
        Recursively builds a decision tree from the given dataset.

        Args:
            dataset (ndarray): rows for this node (last column = labels).
            features_available (list[int]): feature indices not yet used
                on the path from the root to this node.
            current_depth (int): depth of this node in the tree.

        Returns:
            Node: root of the (sub)tree built from `dataset`.
        """

        y = dataset[:, -1]
        class_counts = self._class_counts(y)
        node_entropy = self.entropy(y)

        # Stopping Criteria
        stop = (
            current_depth >= self.max_depth
            or len(dataset) < self.min_samples
            or node_entropy == 0.0  # pure node
            or len(features_available) == 0 # nothing left to split on
        )

        best = None
        if not stop:
            best = self.best_split(dataset, features_available)
            if(best["feature"]) is None or best["gain"] < self.ig_threshold:
                stop = True

        if stop:
            return Node(
                value = self._majority_class(class_counts),
                class_counts=class_counts,
                entropy = node_entropy
            )

        # Recurse into each branch
        chosen_feature = best["feature"]
        remaining_features = [f for f in features_available if f != chosen_feature]

        children = {}
        for value, subset in best["splits"].items():
            children[value] = self.build_tree(
                subset, remaining_features, current_depth + 1
            )

        return Node(
            feature=chosen_feature,
            children=children,
            gain=best["gain"],
            entropy=node_entropy,
            class_counts=class_counts,
        )

    def fit(self, X, y):
        """
        Builds and fits the decision tree to the given X and y values.

        Args:
        X (ndarray): The feature matrix.
        y (ndarray): The target values.
        """
        y = np.asarray(y).reshape(-1, 1)
        dataset = np.concatenate((X, y), axis=1)

        self.classes_ = sorted(np.unique(y))
        features_available = list(range(X.shape[1]))

        self.root = self.build_tree(dataset, features_available)

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value

        feature_value = x[node.feature]
        child = node.children.get(feature_value)

        if child is None:
            # feature value is not seen during training at this branch
            # fall back to majrotiy class at this node
            return self._majority_class(node.class_counts)
        return self._predict_one(x, child)

    def predict(self, X):
        """
        Predicts the class labels for each instance in the feature matrix X.

        Args:
        X (ndarray): The feature matrix to make predictions for.

        Returns:
        list: A list of predicted class labels.
        """
        # TODO
        predictions = [self._predict_one(x, self.root) for x in X]
        return predictions

    # ========================================
    # --- 4. Tree Printing  ---
    # ========================================

    def _format_leaf(self, node):
        counts_str = ", ".join(f"{c}: {n}" for c, n in node.class_counts.items())
        return f"leaf {{{counts_str}}}"

    def _print_node(self, node, depth, lines):
        indent = "    " * depth
        if node.value is not None:
            lines.append(f"{indent}{self._format_leaf(node)}")
            return

        lines.append(
            f"{indent}feature {node.feature} "
            f"(IG: {node.gain:.4f}, Entropy: {node.entropy:.4f})"
        )
        for value in sorted(node.children.keys()):
            child = node.children[value]
            lines.append(f"{indent}-- feature {node.feature} == {int(value)} --")
            self._print_node(child, depth + 1, lines)

    def tree_to_string(self):
        lines = []
        self._print_node(self.root, 0, lines)
        return "\n".join(lines)

    def save_tree(self, output_path):
        with open(output_path, "w") as f:
            f.write(self.tree_to_string())
            f.write("\n")

# ================================================================================================================
# ================================================================================================================

def check_entropy():
    dt = DecisionTree()
    print(f"Entropy of [0,0.1,1] should be 1.0: {dt.entropy([0,0,1,1])}")
    print(f"Entropy of [1,1,1,1] should be 0.0: {dt.entropy([1,1,1,1])}")

def check_information_gain():
    dt = DecisionTree()
    parent = [0] * 9 + [1] * 10
    child0 = [1] * 9  # feature==0 branch: all class 1
    child1 = [0] * 10  # feature==1 branch: all class 0
    ig = dt.information_gain(parent, [child0, child1])
    print(f"IG should be ~0.9980: {ig:.4f}")

def check_split_data():
    dt = DecisionTree()
    data = np.array([
        [1, 0, 1],
        [0, 0, 0],
        [1, 1, 1],
        [0, 1, 0],
    ])
    splits = dt.split_data(data, feature=1)
    for value, subset in splits.items():
        print(f"feature==({value}):\n{subset}")

def check_best_split():
    dt = DecisionTree()
    data = np.array([
        [1, 0, 1],
        [0, 0, 0],
        [1, 1, 1],
        [0, 1, 0],
    ])
    result = dt.best_split(data, features_available=[0, 1])
    print(result)

def check_build_tree():
    dt = DecisionTree(max_depth=100)
    data = np.array([
        [1, 0, 1],
        [0, 0, 0],
        [1, 1, 1],
        [0, 1, 0],
    ])
    X, y = data[:, :-1], data[:, -1]
    dt.fit(X, y)
    preds = dt.predict(X)
    print("Predictions:", preds)
    print("Accuracy:", np.mean(np.array(preds) == y))

def check_tree_printing():
    if len(sys.argv) != 3:
        print("Usage: DecisionTree.py <train_csv> <output_tree_txt>")
        sys.exit(1)

    train_path, output_path = sys.argv[1], sys.argv[2]

    data = np.genfromtxt(train_path, delimiter=",", skip_header=1, dtype=int)
    X, y = data[:, :-1], data[:, -1]

    dt = DecisionTree(min_samples=2, max_depth=1000, ig_threshold=0.00001)
    dt.fit(X, y)

    predictions = np.array(dt.predict(X))
    accuracy = np.mean(predictions == y)
    print(f"Training accuracy: {accuracy * 100:.2f}%")

    dt.save_tree(output_path)
    print(f"Tree written to {output_path}")

def main():
    # check_entropy()
    # check_information_gain()
    # check_split_data()
    # check_best_split()
    # check_build_tree()
    check_tree_printing()

if __name__ == "__main__":
    main()
