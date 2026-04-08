"""
OSmAN-Net: Step 01 - Data Preprocessing
========================================
Load CSVs, parse labels from filenames, drop redundant features,
handle NaN/Inf, normalize, save processed data.
"""

import pandas as pd
import numpy as np
import os
import glob
import re
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

from config import (
    DATA_DIR, PROCESSED_DIR, DROP_FEATURES,
    COARSE_LABELS, BINARY_LABELS, SEED
)

np.random.seed(SEED)


def parse_label_from_filename(filename):
    """Extract fine and coarse labels from CSV filename."""
    base = os.path.basename(filename)
    name = base.replace("_train.pcap.csv", "").replace("_test.pcap.csv", "")
    # Remove trailing numbers (ICMP1 -> ICMP, UDP2 -> UDP)
    fine_label = re.sub(r'\d+$', '', name)

    if name == "Benign":
        return fine_label, "Benign"
    elif "DDoS" in name:
        return fine_label, "DDoS"
    elif "DoS" in name:
        return fine_label, "DoS"
    elif "Recon" in name:
        return fine_label, "Recon"
    elif "Spoofing" in name or "ARP" in name:
        return fine_label, "Spoofing"
    elif "MQTT" in name:
        return fine_label, "MQTT"
    else:
        return fine_label, "Unknown"


def load_all_csvs():
    """Load all CSV files, add labels from filenames."""
    dfs = []
    for split in ["train", "test"]:
        csv_dir = os.path.join(DATA_DIR, split)
        csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))

        for f in csv_files:
            df = pd.read_csv(f)
            fine_label, coarse_label = parse_label_from_filename(f)
            df["fine_label"] = fine_label
            df["coarse_label"] = coarse_label
            df["split"] = split
            dfs.append(df)
            print(f"  Loaded {split}/{os.path.basename(f)}: {len(df):,} rows")

    return pd.concat(dfs, ignore_index=True)


def clean_features(df):
    """Drop redundant/constant features, handle NaN/Inf."""
    feature_cols = [c for c in df.columns
                    if c not in ["fine_label", "coarse_label", "split"]]

    # Drop redundant features
    cols_to_drop = [c for c in DROP_FEATURES if c in feature_cols]
    df = df.drop(columns=cols_to_drop)
    print(f"  Dropped {len(cols_to_drop)} features: {cols_to_drop}")

    # Get remaining feature columns
    feature_cols = [c for c in df.columns
                    if c not in ["fine_label", "coarse_label", "split"]]

    # Replace Inf with NaN, then fill NaN with median
    for col in feature_cols:
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            n_inf = np.isinf(df[col]).sum()
            if n_inf > 0:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                print(f"  {col}: replaced {n_inf} Inf values")

    n_nan = df[feature_cols].isna().sum().sum()
    if n_nan > 0:
        for col in feature_cols:
            if df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        print(f"  Filled {n_nan} NaN values with median")

    # Remove duplicate rows
    n_before = len(df)
    df = df.drop_duplicates(subset=feature_cols, keep='first')
    n_dup = n_before - len(df)
    if n_dup > 0:
        print(f"  Removed {n_dup} duplicate rows")

    return df


def encode_labels(df):
    """Encode labels for binary, coarse (6-class), fine (19-class)."""
    # Binary
    df["label_binary"] = df["coarse_label"].apply(
        lambda x: 0 if x == "Benign" else 1
    )

    # Coarse (6-class)
    le_coarse = LabelEncoder()
    df["label_coarse"] = le_coarse.fit_transform(df["coarse_label"])

    # Fine (19-class)
    le_fine = LabelEncoder()
    df["label_fine"] = le_fine.fit_transform(df["fine_label"])

    print(f"\n  Binary: {df['label_binary'].value_counts().to_dict()}")
    print(f"  Coarse classes ({len(le_coarse.classes_)}): {list(le_coarse.classes_)}")
    print(f"  Fine classes ({len(le_fine.classes_)}): {list(le_fine.classes_)}")

    return df, le_coarse, le_fine


def normalize_features(df_train, df_test, feature_cols):
    """StandardScaler fit on train, transform both."""
    scaler = StandardScaler()
    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_test[feature_cols] = scaler.transform(df_test[feature_cols])
    print(f"  Normalized {len(feature_cols)} features (StandardScaler)")
    return df_train, df_test, scaler


def main():
    print("=" * 60)
    print("OSmAN-Net: Step 01 - Data Preprocessing")
    print("=" * 60)

    # 1. Load
    print("\n[1/5] Loading CSV files...")
    df = load_all_csvs()
    print(f"  Total: {len(df):,} rows, {df.shape[1]} columns")

    # 2. Clean
    print("\n[2/5] Cleaning features...")
    df = clean_features(df)

    feature_cols = [c for c in df.columns
                    if c not in ["fine_label", "coarse_label", "split",
                                 "label_binary", "label_coarse", "label_fine"]]
    print(f"  Remaining features ({len(feature_cols)}): {feature_cols}")

    # 3. Encode labels
    print("\n[3/5] Encoding labels...")
    df, le_coarse, le_fine = encode_labels(df)

    # 4. Split (use original train/test)
    print("\n[4/5] Splitting train/test...")
    df_train = df[df["split"] == "train"].copy().reset_index(drop=True)
    df_test = df[df["split"] == "test"].copy().reset_index(drop=True)
    print(f"  Train: {len(df_train):,} | Test: {len(df_test):,}")

    # 5. Normalize
    print("\n[5/5] Normalizing features...")
    df_train, df_test, scaler = normalize_features(df_train, df_test, feature_cols)

    # Save
    print("\n" + "=" * 60)
    print("Saving processed data...")

    # Save as numpy arrays for fast loading
    X_train = df_train[feature_cols].values.astype(np.float32)
    X_test = df_test[feature_cols].values.astype(np.float32)

    y_train_binary = df_train["label_binary"].values
    y_test_binary = df_test["label_binary"].values

    y_train_coarse = df_train["label_coarse"].values
    y_test_coarse = df_test["label_coarse"].values

    y_train_fine = df_train["label_fine"].values
    y_test_fine = df_test["label_fine"].values

    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_DIR, "y_train_binary.npy"), y_train_binary)
    np.save(os.path.join(PROCESSED_DIR, "y_test_binary.npy"), y_test_binary)
    np.save(os.path.join(PROCESSED_DIR, "y_train_coarse.npy"), y_train_coarse)
    np.save(os.path.join(PROCESSED_DIR, "y_test_coarse.npy"), y_test_coarse)
    np.save(os.path.join(PROCESSED_DIR, "y_train_fine.npy"), y_train_fine)
    np.save(os.path.join(PROCESSED_DIR, "y_test_fine.npy"), y_test_fine)

    # Save metadata
    metadata = {
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "scaler": scaler,
        "le_coarse": le_coarse,
        "le_fine": le_fine,
        "train_size": len(df_train),
        "test_size": len(df_test),
        "coarse_classes": list(le_coarse.classes_),
        "fine_classes": list(le_fine.classes_),
        "coarse_distribution_train": df_train["coarse_label"].value_counts().to_dict(),
        "coarse_distribution_test": df_test["coarse_label"].value_counts().to_dict(),
    }

    with open(os.path.join(PROCESSED_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    print(f"\n  Saved to {PROCESSED_DIR}/")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  Features: {len(feature_cols)}")

    # Print class weights (for loss function)
    print("\n  Class weights (coarse) for weighted loss:")
    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight('balanced',
                              classes=np.unique(y_train_coarse),
                              y=y_train_coarse)
    for cls_name, w in zip(le_coarse.classes_, cw):
        print(f"    {cls_name}: {w:.4f}")

    print("\nDONE!")


if __name__ == "__main__":
    main()
