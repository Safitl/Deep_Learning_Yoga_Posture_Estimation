"""
make_counter_classes.py — Create per-class row counts for stratified splitting.


Generate `counter_classes.npy`, a 1D NumPy array where element c equals the
number of rows in the input CSV whose label column equals c. This file is
consumed by `train_val_test_divide.py` to slice each class block consistently.

Inputs
- CSV file with at least one integer label column (default: 'label_idx').

Outputs
- NPY file: `counter_classes.npy` (default) — shape (C,), dtype int64,
  ordered as [count(0), count(1), ..., count(C-1)].


Dependencies
- numpy
- pandas
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def compute_counts(
    csv_path: str | Path,
    label_col: str = "label_idx",
    num_classes: int | None = None,
) -> tuple[np.ndarray, int, int]:
    
    """
    Compute per-class counts ordered by class id.

    Parameters
    ----------
    csv_path : str | Path
        Path to the input CSV (e.g., 'combo_features.csv').
    label_col : str
        Name of the label column (default: 'label_idx').
    num_classes : int | None
        If provided, enforce labels in [0..num_classes-1].
        If None, infer as max(label)+1.

    Returns
    -------
    counts : np.ndarray
        Shape (C,), dtype int64. counts[c] = number of rows with label c.
    C : int
        Number of classes used to build the counts.
    N : int
        Total number of rows in the CSV.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in '{csv_path.name}'.")

    # Ensure integer labels
    try:
        labels = df[label_col].astype(int).to_numpy()
    except Exception as e:
        raise ValueError(f"Column '{label_col}' must be integer-typed.") from e

    inferred_C = int(labels.max()) + 1 if labels.size else 0
    C = int(num_classes) if num_classes is not None else inferred_C

    if num_classes is not None:
        # Hard check: labels must fit the declared range
        lo, hi = labels.min(initial=0), labels.max(initial=-1)
        if lo < 0 or hi >= num_classes:
            raise ValueError(
                f"Found labels outside [0, {num_classes-1}]: min={lo}, max={hi}"
            )

    # Build counts in 0..C-1 order, filling missing classes with 0
    counts = (
        pd.Series(labels)
        .value_counts(sort=False)            # keep label ids as index
        .reindex(range(C), fill_value=0)     # ensure 0..C-1 order
        .astype(np.int64)
        .to_numpy()
    )

    return counts, C, len(df)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create counter_classes.npy (per-class row counts) from a labeled CSV."
    )
    ap.add_argument("--csv", default="combo_features.csv",
                    help="Path to the input CSV (default: combo_features.csv)")
    ap.add_argument("--label-col", default="label_idx",
                    help="Label column name in the CSV (default: label_idx)")
    ap.add_argument("--num-classes", type=int, default=None,
                    help="Total number of classes. If omitted, inferred as max(label)+1.")
    ap.add_argument("--out", default="counter_classes.npy",
                    help="Output .npy path for counts (default: counter_classes.npy)")
    ap.add_argument("--csv-out", default=None,
                    help="Optional path to also write a human-readable counts CSV.")

    args = ap.parse_args()

    counts, C, N = compute_counts(args.csv, args.label_col, args.num_classes)

    # Save NPY
    out_path = Path(args.out)
    np.save(out_path, counts)

    # Optional: also save a CSV for quick inspection
    if args.csv_out:
        pd.DataFrame({"class_id": range(C), "count": counts}).to_csv(
            args.csv_out, index=False
        )

    # Console summary
    print(f"Input rows     : {N}")
    print(f"Classes (C)    : {C}")
    print(f"Counts sum     : {counts.sum()}")
    print(f"Saved NPY      : {out_path.resolve()}")
    if args.csv_out:
        print(f"Saved counts CSV: {Path(args.csv_out).resolve()}")


if __name__ == "__main__":
    main()
