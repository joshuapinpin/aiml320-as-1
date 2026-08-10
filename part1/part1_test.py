import unittest
import numpy as np

from DecisionTree import DecisionTree


class TestDecisionTree(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)  # add assertion here

    def test_entropy(self):
        dt = DecisionTree()
        self.assertAlmostEqual(dt.entropy([0, 0, 1, 1]), 1.0)
        self.assertAlmostEqual(dt.entropy([1, 1, 1, 1]), 0.0)

    def test_information_gain(self):
        dt = DecisionTree()
        parent = [0] * 9 + [1] * 10
        child0 = [1] * 9  # feature==0 branch: all class 1
        child1 = [0] * 10  # feature==1 branch: all class 0
        ig = dt.information_gain(parent, [child0, child1])
        self.assertAlmostEqual(ig, 0.9980, places=4)

    def test_split_data(self):
        dt = DecisionTree()
        data = np.array([
            [1, 0, 1],
            [0, 0, 0],
            [1, 1, 1],
            [0, 1, 0],
        ])
        splits = dt.split_data(data, feature=1)
        self.assertEqual(set(splits.keys()), {0, 1})
        self.assertEqual(len(splits[0]), 2)
        self.assertEqual(len(splits[1]), 2)

    def test_best_split(self):
        dt = DecisionTree()
        data = np.array([
            [1, 0, 1],
            [0, 0, 0],
            [1, 1, 1],
            [0, 1, 0],
        ])
        result = dt.best_split(data, features_available=[0, 1])
        self.assertIn(result["feature"], [0, 1])
        self.assertGreater(result["gain"], 0)

    def test_build_tree_and_predict(self):
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
        accuracy = np.mean(np.array(preds) == y)
        self.assertEqual(accuracy, 1.0)

if __name__ == '__main__':
    unittest.main()
