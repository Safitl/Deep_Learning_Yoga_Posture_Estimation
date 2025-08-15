"""
This script divides a dataset into training, validation, and test sets.

The split is in a stratified manner, meaning that the ratio of each class is maintained 
across the new subsets.


The script performs the following steps:
1. Loads the combined features and labels from a CSV file (`combo_features.csv`).
2. Loads the number of rows for each class from a NumPy file (`counter_classes.npy`).
3. Iterates through each class, shuffles the rows belonging to that class,
   and splits them according to the defined ratios (`split_ratios`).
4. Concatenates the per-class subsets to form the final training, validation,
   and test sets.
5. Saves the resulting datasets to individual CSV files (`train_set.csv`, `val_set.csv`, `test_set.csv`).
6. Prints the size of each resulting dataset for verification.

Usage:
    Ensure that 'combo_features.csv' and 'counter_classes.npy' are in the same directory.
    Run the script: `python train_val_test_divide.py`

Dependencies:
    - numpy
    - pandas

"""

import numpy as np
import pandas as pd

# Configuration
split_ratios = dict(train=0.8, val=0.1, test=0.1)
num_classes = 47 # Based on the data.
np.random.seed(42) # For reproducible shuffling.

# STEP 1: Load and divide the dataset for each class.
lengths_per_class = np.load("counter_classes.npy")
combo = pd.read_csv("combo_features.csv")

sub_dfs = dict(train=[], val=[], test=[])
start = 0

for cls, n_rows in enumerate(lengths_per_class):
    # Determine the slice for the current class.
    grp = combo.iloc[start:start + n_rows]
    
    # Sanity-check: all rows should have the same label.
    assert (grp["label_idx"] == cls).all()
    
    # Shuffle the indices to ensure random distribution.
    idx = np.random.permutation(n_rows)
    
    # Calculate the number of rows for each split.
    n_train = int(split_ratios["train"] * n_rows)
    n_val = int(split_ratios["val"] * n_rows)
    
    # Slice the shuffled indices for each split.
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    # Append the three pieces to the sub_dfs dictionary.
    sub_dfs["train"].append(grp.iloc[train_idx])
    sub_dfs["val"].append(grp.iloc[val_idx])
    sub_dfs["test"].append(grp.iloc[test_idx])

    # Advance to the next class slice.
    start += n_rows

# Concatenate the per-class pieces to form the final datasets.
train_set = pd.concat(sub_dfs["train"]).reset_index(drop=True)
val_set = pd.concat(sub_dfs["val"]).reset_index(drop=True)
test_set = pd.concat(sub_dfs["test"]).reset_index(drop=True)

# Save the final datasets to CSV files.
train_set.to_csv("train_set.csv", index=False)
val_set.to_csv("val_set.csv", index=False)
test_set.to_csv("test_set.csv", index=False)

# Print the size of each dataset for verification.
print(f"train: {len(train_set)} val: {len(val_set)} test: {len(test_set)}")
