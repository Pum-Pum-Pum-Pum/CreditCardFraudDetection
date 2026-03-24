from pathlib import Path
import json

import pandas as pd


REFERENCE_PATH = Path("artifacts/splits/train.csv")
CURRENT_INPUT_PATH = Path("artifacts/model/sample_inference_input.csv")
OUTPUT_PATH = Path("artifacts/model/drift_report.json")

NUMERIC_FEATURES = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
    "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount",
]


def load_reference_and_current():
    reference_df = pd.read_csv(REFERENCE_PATH)
    current_df = pd.read_csv(CURRENT_INPUT_PATH)
    return reference_df, current_df


def summarize_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame):
    report = []

    for col in NUMERIC_FEATURES:
        ref_mean = float(reference_df[col].mean())
        cur_mean = float(current_df[col].mean())
        ref_std = float(reference_df[col].std())
        cur_std = float(current_df[col].std())
        mean_shift = cur_mean - ref_mean
        mean_shift_ratio = None if ref_std == 0 else mean_shift / ref_std

        report.append(
            {
                "feature": col,
                "reference_mean": ref_mean,
                "current_mean": cur_mean,
                "reference_std": ref_std,
                "current_std": cur_std,
                "mean_shift": mean_shift,
                "mean_shift_ratio": mean_shift_ratio,
            }
        )

    report_df = pd.DataFrame(report)
    report_df["abs_mean_shift_ratio"] = report_df["mean_shift_ratio"].abs()
    report_df = report_df.sort_values("abs_mean_shift_ratio", ascending=False)
    return report_df


def main():
    if not CURRENT_INPUT_PATH.exists():
        print(f"Current inference sample not found: {CURRENT_INPUT_PATH}")
        print("Create the sample inference file first before running drift monitoring.")
        return

    reference_df, current_df = load_reference_and_current()
    drift_df = summarize_drift(reference_df, current_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(drift_df.to_dict(orient="records"), f, indent=2)

    print("=== DRIFT MONITOR REPORT ===")
    print(f"Reference rows: {len(reference_df)}")
    print(f"Current rows: {len(current_df)}")
    print("\nTop 10 shifted features by standardized mean shift:")
    print(drift_df[["feature", "reference_mean", "current_mean", "mean_shift", "mean_shift_ratio"]].head(10))
    print(f"\nSaved drift report to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()