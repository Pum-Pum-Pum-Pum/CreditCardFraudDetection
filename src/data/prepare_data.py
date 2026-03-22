from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/raw/creditcard.csv")
OUTPUT_DIR = Path("artifacts/splits")
RANDOM_STATE = 42


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def summarize_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    duplicated_mask = df.duplicated(keep=False)
    dup_df = df.loc[duplicated_mask].copy()

    summary = pd.DataFrame(
        {
            "total_rows": [len(df)],
            "duplicate_rows": [int(df.duplicated().sum())],
            "rows_in_duplicate_groups": [int(duplicated_mask.sum())],
            "fraud_rate_full": [df["Class"].mean()],
            "fraud_rate_duplicates_only": [dup_df["Class"].mean() if not dup_df.empty else 0.0],
        }
    )
    return summary


def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def split_data(df: pd.DataFrame):
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def save_splits(X_train, X_valid, X_test, y_train, y_valid, y_test, output_dir: Path = OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = X_train.copy()
    train_df["Class"] = y_train.values

    valid_df = X_valid.copy()
    valid_df["Class"] = y_valid.values

    test_df = X_test.copy()
    test_df["Class"] = y_test.values

    train_df.to_csv(output_dir / "train.csv", index=False)
    valid_df.to_csv(output_dir / "valid.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)


def print_split_summary(y_train, y_valid, y_test):
    split_summary = pd.DataFrame(
        {
            "split": ["train", "valid", "test"],
            "rows": [len(y_train), len(y_valid), len(y_test)],
            "fraud_count": [int(y_train.sum()), int(y_valid.sum()), int(y_test.sum())],
            "fraud_rate": [y_train.mean(), y_valid.mean(), y_test.mean()],
        }
    )
    print(split_summary)


def main():
    df = load_data()
    print("=== RAW DATA SHAPE ===")
    print(df.shape)

    print("\n=== DUPLICATE SUMMARY ===")
    print(summarize_duplicates(df))

    df_model = drop_duplicate_rows(df)
    print("\n=== SHAPE AFTER DROPPING EXACT DUPLICATES ===")
    print(df_model.shape)

    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df_model)

    print("\n=== STRATIFIED SPLIT SUMMARY ===")
    print_split_summary(y_train, y_valid, y_test)

    save_splits(X_train, X_valid, X_test, y_train, y_valid, y_test)
    print("\nSaved train/valid/test CSV files to artifacts/splits")

    print("\n=== PREPROCESSING DECISION FOR STEP 2 ===")
    print("1. Keep V1-V28 as numeric features for now.")
    print("2. Keep Time and Amount for baseline modeling.")
    print("3. Do not scale yet inside raw split files; scaling should be fit on train only in the modeling pipeline.")
    print("4. Use stratified splitting because fraud is extremely rare.")


if __name__ == "__main__":
    main()