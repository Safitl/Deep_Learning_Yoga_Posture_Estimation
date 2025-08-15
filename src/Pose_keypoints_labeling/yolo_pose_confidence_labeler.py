"""
yolo_pose_labeler.py

This script processes a dataset of yoga posture images, detects keypoints using a YOLOv8 pose model,
and generates a labeled dataset with keypoint coordinates + confidence and metadata.
It can optionally save annotated images and per-image JSON files.

Main output:
- yolo_keypoints_dataset.csv  (columns: kp{j}_x, kp{j}_y, kp{j}_c for j in [0..16], image_path, label_str, label_idx)

Dependencies:
- ultralytics (YOLOv11)
- OpenCV (cv2)
- pandas
- json
- os
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

N_JOINTS = 17  # COCO format


def data_label(dataset_folder: str, saving_flag: bool = False):
    """
    Process a dataset of images, detect pose keypoints using YOLOv11, and return labeled keypoint data.

    Args:
        dataset_folder (str): Path to the root dataset folder. It should contain subfolders named by class labels,
                              each with images of that class.
        saving_flag (bool): If True, saves annotated images and keypoint JSON files to disk.

    Returns:
        list[list]: Each inner list contains:
            - 51 float values: keypoint triplets (x, y, conf) normalized to [0,1],
                              flattened as [x0,y0,c0, x1,y1,c1, ..., x16,y16,c16]
                              or zeros if no keypoints detected.
            - str: image path
            - str: class label
            - int: class index (incremental integer assigned per class folder)
    """
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    model = YOLO("yolo11x-pose.pt")
    rows = []
    class_counter = 0

    for label in sorted(os.listdir(dataset_folder)):
        if label.lower() == "poses.json":
            continue  # skip meta file

        class_dir = os.path.join(dataset_folder, label)
        if not os.path.isdir(class_dir):
            continue

        key_dir = os.path.join(class_dir, "keypoints")
        ann_dir = os.path.join(class_dir, "annotated")
        if saving_flag:
            os.makedirs(key_dir, exist_ok=True)
            os.makedirs(ann_dir, exist_ok=True)

        # iterate images
        for img_name in sorted(os.listdir(class_dir)):
            img_path = os.path.join(class_dir, img_name)
            if not (os.path.isfile(img_path) and img_path.lower().endswith(valid_ext)):
                continue

            # YOLO pose prediction
            results = model.predict(img_path, boxes=False, verbose=False)
            r = results[0]

            # default: zeros (no detection)
            flat_xyc = [0.0] * (N_JOINTS * 3)

            if r.keypoints is not None and r.keypoints.xyn is not None and r.keypoints.xyn.numel() > 0:
                # choose best person if multiple, by highest mean keypoint confidence (if available)
                # xyn: [num_people, 17, 2], conf: [num_people, 17]
                xyn = r.keypoints.xyn.detach().cpu().numpy()
                if getattr(r.keypoints, "conf", None) is not None and r.keypoints.conf is not None and r.keypoints.conf.numel() > 0:
                    conf_all = r.keypoints.conf.detach().cpu().numpy()
                    best_idx = int(conf_all.mean(axis=1).argmax())
                    cf = conf_all[best_idx]  # shape (17,)
                else:
                    best_idx = 0
                    cf = np.ones((N_JOINTS,), dtype=np.float32)

                xy = xyn[best_idx]  # shape (17,2)

                # make sure shapes align (some models can return different keypoint counts)
                if xy.shape[0] >= N_JOINTS:
                    xy = xy[:N_JOINTS]
                    cf = cf[:N_JOINTS]
                else:
                    # pad if fewer than 17 (unlikely with COCO)
                    pad_k = N_JOINTS - xy.shape[0]
                    xy = np.vstack([xy, np.zeros((pad_k, 2), dtype=xy.dtype)])
                    cf = np.concatenate([cf, np.zeros((pad_k,), dtype=cf.dtype)])

                # concatenate per-joint triplets (x, y, conf)
                xyc = np.concatenate([xy, cf[:, None]], axis=1).reshape(-1)  # (17*3,)
                flat_xyc = xyc.astype(float).tolist()

            # append row
            row = flat_xyc + [img_path, label, class_counter]
            rows.append(row)

            # optional artifact saving
            if saving_flag:
                # annotated image
                ann_img = r.plot(boxes=False)
                cv2.imwrite(os.path.join(ann_dir, img_name), ann_img)

                # structured JSON (easier to read)
                base, _ = os.path.splitext(img_name)
                kps_struct = []
                for j in range(N_JOINTS):
                    jx = flat_xyc[3 * j + 0]
                    jy = flat_xyc[3 * j + 1]
                    jc = flat_xyc[3 * j + 2]
                    kps_struct.append({"joint": j, "x": jx, "y": jy, "c": jc})

                with open(os.path.join(key_dir, f"{base}.json"), "w") as f:
                    json.dump(
                        {
                            "image_path": img_path,
                            "label_str": label,
                            "label_idx": class_counter,
                            "keypoints_xyc": kps_struct,
                        },
                        f,
                        indent=2,
                    )

        class_counter += 1

    return rows


if __name__ == "__main__":
    # Default dataset folder: two levels up + "yoga_kaggle_dataset", same as your original script
    dataset_folder = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..","yoga_kaggle_dataset")
    )
    rows = data_label(dataset_folder, saving_flag=False)

    # Build DataFrame with explicit (x,y,c) names
    cols = []
    for j in range(N_JOINTS):
        cols += [f"kp{j}_x", f"kp{j}_y", f"kp{j}_c"]
    cols += ["image_path", "label_str", "label_idx"]

    df = pd.DataFrame(rows, columns=cols)
    df = df.fillna(0.0)
    df["label_idx"] = df["label_idx"].astype(int)
    df.to_csv("yolo_keypoints_confidence_dataset.csv", index=False)
    print("Saved", len(df), "rows to yolo_keypoints_dataset.csv")
