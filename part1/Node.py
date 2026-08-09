class Node:
    """
    A class representing a node in a decision tree (supports multi-way splits).
    """

    def __init__(self, feature=None, children=None, gain=None, entropy=None,
                 value=None, class_counts=None):
        """
        Args:
            feature: index of the feature this node splits on (None for leaves).
            children: dict {feature_value: Node} — one child per distinct value
                of `feature`. None for leaves.
            gain: information gain of the split at this node (None for leaves).
            entropy: entropy of this node's dataset (used for split nodes' printout,
                and handy to keep for leaves too).
            value: predicted class label if this is a leaf. None for split nodes.
            class_counts: dict {class_label: count} of instances reaching this
                node — required for leaf printout, e.g. {0: 31, 1: 0}.
        """
        self.feature = feature
        self.children = children if children is not None else {}
        self.gain = gain
        self.entropy = entropy
        self.value = value
        self.class_counts = class_counts