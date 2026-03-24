from pathlib import Path
import json
import joblib

import pandas as pd


MODEL_PATH = Path("artifacts/model/xgboost_fraud_pipeline.joblib")
METADATA_PATH = Path("artifacts/model/model_metadata.json")
INPUT_PATH = Path("artifacts/model/sample_inference_input.csv")


def load_model_and_metadata():
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata


def predict_from_file(input_path: Path):
    model, metadata = load_model_and_metadata()
    df = pd.read_csv(input_path)

    feature_names = metadata["feature_names"]
    df = df[feature_names]

    fraud_probability = model.predict_proba(df)[:, 1]
    predicted_class = (fraud_probability >= metadata["threshold"]).astype(int)

    result = df.copy()
    result["fraud_probability"] = fraud_probability
    result["predicted_class"] = predicted_class

    print("=== INFERENCE RESULTS ===")
    print(result.head(10))


if __name__ == "__main__":
    if not INPUT_PATH.exists():
        print(f"Input file not found: {INPUT_PATH}")
        print("Create a CSV with the same feature columns used during training.")
    else:
        predict_from_file(INPUT_PATH)