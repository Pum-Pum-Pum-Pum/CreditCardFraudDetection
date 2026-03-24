from pathlib import Path
import json
import joblib

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


TRAIN_PATH = Path("artifacts/splits/train.csv")
VALID_PATH = Path("artifacts/splits/valid.csv")
MODEL_DIR = Path("artifacts/model")
MODEL_PATH = MODEL_DIR / "xgboost_fraud_pipeline.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

RANDOM_STATE = 42
THRESHOLD = 0.17
FP_COST = 1
FN_COST = 5


def load_split(path: Path):
    df = pd.read_csv(path)
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def build_xgboost_pipeline(feature_names):
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_pipeline, feature_names)]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    random_state=RANDOM_STATE,
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main():
    X_train, y_train = load_split(TRAIN_PATH)
    X_valid, y_valid = load_split(VALID_PATH)

    X_final = pd.concat([X_train, X_valid], axis=0).reset_index(drop=True)
    y_final = pd.concat([y_train, y_valid], axis=0).reset_index(drop=True)

    feature_names = X_final.columns.tolist()
    model = build_xgboost_pipeline(feature_names)
    model.fit(X_final, y_final)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "model_name": "xgboost_fraud_pipeline",
        "model_type": "Pipeline(XGBClassifier)",
        "feature_names": feature_names,
        "threshold": THRESHOLD,
        "fp_cost": FP_COST,
        "fn_cost": FN_COST,
        "training_rows": int(len(X_final)),
        "positive_class": 1,
        "negative_class": 0,
        "notes": "Threshold selected from validation cost tuning under FN:FP = 5:1.",
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("=== MODEL PACKAGING COMPLETE ===")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved metadata to: {METADATA_PATH}")
    print("\nMetadata preview:")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()