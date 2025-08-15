"""
This script combines two datasets containing different types of features for the same images.
It merges keypoint data (from YOLO) and embedding data (from a ResNet-18 model) into
a single, unified DataFrame.

The script performs the following key tasks:
1.  Loads two CSV files: one with keypoint features from a YOLO model (`yolo_keypoints_dataset.csv`)
    and another with embeddings from a ResNet-18 model (`resnet18_embeddings.csv`).
2.  Renames the feature columns in each DataFrame to ensure they are unique after
    the merge, by prefixing them with 'kp_' and 'cnn_'. This prevents column name
    conflicts.
3.  Merges the two DataFrames , saves also `image_path`, `label_idx`, and `label_str`. 
5.  Saves the combined dataset to a CSV file (`combo_features.csv`). 

Usage:
    Ensure that 'yolo_keypoints_dataset.csv' and 'resnet18_embeddings.csv'
    are in the same directory.
    Run the script: `python combine_features.py`

Dependencies:
    - pandas
    - pyarrow (for saving to parquet format)
"""
import pandas as pd
import numpy as np # It's a good practice to include all imports, even if not explicitly used in the core logic.

# Load datasets
kp = pd.read_csv("yolo_keypoints_dataset.csv")
cnn = pd.read_csv("resnet18_embeddings.csv")

# Rename feature columns for uniqueness
kp_feat = [c for c in kp.columns if c.startswith("e")]
cnn_feat = [c for c in cnn.columns if c.startswith("e")]

kp.rename(columns={c: f"kp_{c}" for c in kp_feat}, inplace=True)
cnn.rename(columns={c: f"cnn_{c}" for c in cnn_feat}, inplace=True)

# Merge on the common keys
keys = ["image_path", "label_idx", "label_str"]

# First, let's ensure the keys exist in both dataframes.
if not set(keys).issubset(cnn.columns) and "image_path" in kp.columns:
    cnn["image_path"] = kp["image_path"] 
    cnn["label_idx"] = kp["label_idx"]
    cnn["label_str"] = kp["label_str"]

combo = kp.merge(cnn, on=keys, how='inner')  # Perform an inner merge on the keys
front = [c for c in combo.columns if c not in keys] + keys
combo = combo[front]

# Save the combined dataset
combo.to_csv("combo_features.csv", index=False)
