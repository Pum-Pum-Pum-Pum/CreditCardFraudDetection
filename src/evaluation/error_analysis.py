from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


TRAIN_PATH = Path("artifacts/splits/train.csv")
TEST_PATH = Path("artifacts/splits/test.csv")
RANDOM_STATE = 42
THRESHOLD = 0.29


def load_split(path: Path):
    df = pd.read_csv(path)
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return df, X, y


def build_best_random_forest(feature_names):
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
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=12,
                    min_samples_leaf=1,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def main():
    train_df, X_train, y_train = load_split(TRAIN_PATH)
    test_df, X_test, y_test = load_split(TEST_PATH)

    feature_names = X_train.columns.tolist()
    model = build_best_random_forest(feature_names)
    model.fit(X_train, y_train)

    test_scores = model.predict_proba(X_test)[:, 1]
    test_pred = (test_scores >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()

    summary = pd.DataFrame(
        [
            {
                "threshold": THRESHOLD,
                "roc_auc": roc_auc_score(y_test, test_scores),
                "pr_auc": average_precision_score(y_test, test_scores),
                "precision": precision_score(y_test, test_pred, zero_division=0),
                "recall": recall_score(y_test, test_pred, zero_division=0),
                "f1": f1_score(y_test, test_pred, zero_division=0),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        ]
    )

    print("=== FINAL TEST METRICS ===")
    print(summary)

    error_df = test_df.copy()
    error_df["score"] = test_scores
    error_df["predicted_class"] = test_pred

    false_positives = error_df[(error_df["Class"] == 0) & (error_df["predicted_class"] == 1)].copy()
    false_negatives = error_df[(error_df["Class"] == 1) & (error_df["predicted_class"] == 0)].copy()

    print("\n=== ERROR COUNTS ===")
    print(f"False positives: {len(false_positives)}")
    print(f"False negatives: {len(false_negatives)}")

    print("\n=== FALSE POSITIVE SCORE SUMMARY ===")
    print(false_positives["score"].describe())

    print("\n=== FALSE NEGATIVE SCORE SUMMARY ===")
    print(false_negatives["score"].describe())

    analysis_features = ["Time", "Amount", "V4", "V10", "V12", "V14", "V17", "score"]

    print("\n=== FALSE POSITIVE FEATURE SUMMARY ===")
    print(false_positives[analysis_features].describe().T)

    print("\n=== FALSE NEGATIVE FEATURE SUMMARY ===")
    print(false_negatives[analysis_features].describe().T)

    print("\n=== SAMPLE FALSE POSITIVES ===")
    print(false_positives[analysis_features + ["Class", "predicted_class"]].head(10))

    print("\n=== SAMPLE FALSE NEGATIVES ===")
    print(false_negatives[analysis_features + ["Class", "predicted_class"]].head(10))

    print("\nStep 6 complete: final test evaluation and error analysis generated.")


if __name__ == "__main__":
    main()