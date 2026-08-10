# Part 1 – Decision Tree

## Requirements
Python 3, numpy

## How to run
```
python DecisionTree.py <train_csv> <output_tree_txt>
```

Example:
```
python DecisionTree.py rtg_A.csv output_tree_A.txt
python DecisionTree.py rtg_B.csv output_tree_B.txt
```

Prints training accuracy to console, writes the tree structure to the given output file, and writes per-instance results to a csv.

## Files
- `DecisionTree.py` – decision tree implementation (entropy/IG, from scratch)
- `Node.py` – tree node class
- `output_tree_A.txt`, `output_tree_B.txt` – generated trees for rtg_A / rtg_B
- `output_tree_A_instance_results.csv`, `output_tree_B_instance_results.csv` – generated per-instance results for rtg_A / rtg_B