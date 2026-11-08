"""
prepare_dataset.py
------------------
Builds the dataset used in this project from the ORIGINAL UCI file
"Online Shoppers Purchasing Intention Dataset" (online_shoppers_intention.csv).

Source : https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
Credit : Sakar, C. & Kastro, Y. (2018), DOI 10.24432/C5F88Q  (CC BY 4.0)

What this script changes (document these in your README):
  1. Cleaning
       - removes exact duplicate sessions (125 rows)
       - clips any negative page-duration to 0 (none present, kept as a safeguard)
  2. Fixes column types
       - OperatingSystems, Browser, Region, TrafficType are integer *codes*, not
         quantities -> converted to labelled categories (OS_2, Region_3, ...)
         so they are one-hot encoded instead of being scaled like numbers
       - Weekend (True/False) and Revenue (True/False) -> 0/1
  3. Feature engineering (5 new columns)
       - TotalPages        = Administrative + Informational + ProductRelated
       - TotalDuration     = sum of the three *_Duration columns (seconds)
       - AvgTimePerPage    = TotalDuration / TotalPages   (0 if no pages)
       - ProductPageShare  = ProductRelated / TotalPages  (0 if no pages)
       - Quarter           = Q1..Q4 derived from Month

Usage:
    python prepare_dataset.py --src online_shoppers_intention.csv --out online_shoppers_prepared.csv
"""
import argparse

import numpy as np
import pandas as pd

CODE_PREFIX = {
    "OperatingSystems": "OS",
    "Browser": "Browser",
    "Region": "Region",
    "TrafficType": "Traffic",
}
MONTH_TO_QUARTER = {
    "Jan": "Q1", "Feb": "Q1", "Mar": "Q1",
    "Apr": "Q2", "May": "Q2", "Jun": "Q2", "June": "Q2",
    "Jul": "Q3", "Aug": "Q3", "Sep": "Q3",
    "Oct": "Q4", "Nov": "Q4", "Dec": "Q4",
}
DURATION_COLS = ["Administrative_Duration", "Informational_Duration", "ProductRelated_Duration"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="original UCI CSV")
    ap.add_argument("--out", default="online_shoppers_prepared.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.src)
    n_start = len(df)

    # 1. cleaning
    df = df.drop_duplicates().reset_index(drop=True)
    for col in DURATION_COLS:
        df[col] = df[col].clip(lower=0)

    # 2. types
    for col, prefix in CODE_PREFIX.items():
        df[col] = prefix + "_" + df[col].astype(int).astype(str)
    df["Weekend"] = df["Weekend"].astype(int)
    target = df.pop("Revenue").astype(int)

    # 3. engineered features
    df["TotalPages"] = df["Administrative"] + df["Informational"] + df["ProductRelated"]
    df["TotalDuration"] = df[DURATION_COLS].sum(axis=1)
    safe_pages = df["TotalPages"].replace(0, np.nan)
    df["AvgTimePerPage"] = (df["TotalDuration"] / safe_pages).fillna(0).round(4)
    df["ProductPageShare"] = (df["ProductRelated"] / safe_pages).fillna(0).round(4)
    df["Quarter"] = df["Month"].map(MONTH_TO_QUARTER)
    assert df["Quarter"].notna().all(), "unmapped month value found"

    df["Revenue"] = target          # target column last
    df.to_csv(args.out, index=False)

    print(f"Rows: {n_start} -> {len(df)}  (removed {n_start - len(df)} duplicates)")
    print(f"Features: {df.shape[1] - 1} | target: Revenue (1 = purchase)")
    print("Class balance:", df['Revenue'].value_counts(normalize=True).round(3).to_dict())
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
