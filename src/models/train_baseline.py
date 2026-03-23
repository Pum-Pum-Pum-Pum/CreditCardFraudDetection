from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


TRAIN_PATH = Path("artifacts/splits/train.csv")
VALID_PATH = Path("artifacts/splits/valid.csv")
TEST_PATH = Path("artifacts/splits/test.csv")
RANDOM_STATE = 42


def load_split(path: Path):
    df = pd.read_csv(path)
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def build_logistic_pipeline(feature_names):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_pipeline, feature_names)]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return model


def build_random_forest(feature_names):
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_pipeline, feature_names)]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return model


def evaluate_at_threshold(y_true, y_scores, threshold=0.5):
    y_pred = (y_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, y_scores),
        "pr_auc": average_precision_score(y_true, y_scores),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def find_best_f1_threshold(y_true, y_scores):
    candidate_thresholds = [i / 100 for i in range(5, 100, 5)]
    evaluations = [evaluate_at_threshold(y_true, y_scores, thr) for thr in candidate_thresholds]
    best = max(evaluations, key=lambda x: x["f1"])
    return best, pd.DataFrame(evaluations)


def run_model(model_name, model, X_train, y_train, X_valid, y_valid, X_test, y_test):
    print(f"\n=== {model_name.upper()} ===")
    model.fit(X_train, y_train)

    valid_scores = model.predict_proba(X_valid)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]

    valid_default = evaluate_at_threshold(y_valid, valid_scores, threshold=0.5)
    best_valid, valid_threshold_table = find_best_f1_threshold(y_valid, valid_scores)
    test_tuned = evaluate_at_threshold(y_test, test_scores, threshold=best_valid["threshold"])

    print("Validation metrics at threshold=0.50")
    print(pd.DataFrame([valid_default]))

    print("\nBest validation threshold by F1")
    print(pd.DataFrame([best_valid]))

    print("\nTest metrics using validation-selected threshold")
    print(pd.DataFrame([test_tuned]))

    return {
        "model_name": model_name,
        "valid_default": valid_default,
        "best_valid": best_valid,
        "test_tuned": test_tuned,
        "valid_threshold_table": valid_threshold_table,
    }


def main():
    X_train, y_train = load_split(TRAIN_PATH)
    X_valid, y_valid = load_split(VALID_PATH)
    X_test, y_test = load_split(TEST_PATH)

    feature_names = X_train.columns.tolist()

    logistic_model = build_logistic_pipeline(feature_names)
    rf_model = build_random_forest(feature_names)

    results = []
    results.append(run_model("logistic_regression", logistic_model, X_train, y_train, X_valid, y_valid, X_test, y_test))
    results.append(run_model("random_forest", rf_model, X_train, y_train, X_valid, y_valid, X_test, y_test))

    summary_rows = []
    for result in results:
        summary_rows.append(
            {
                "model": result["model_name"],
                "valid_pr_auc_at_0.5": result["valid_default"]["pr_auc"],
                "valid_recall_at_0.5": result["valid_default"]["recall"],
                "valid_precision_at_0.5": result["valid_default"]["precision"],
                "best_valid_threshold": result["best_valid"]["threshold"],
                "test_pr_auc": result["test_tuned"]["pr_auc"],
                "test_recall": result["test_tuned"]["recall"],
                "test_precision": result["test_tuned"]["precision"],
                "test_f1": result["test_tuned"]["f1"],
            }
        )

    print("\n=== BASELINE MODEL SUMMARY ===")
    print(pd.DataFrame(summary_rows))

    print("\nStep 3 complete: trained baseline models and compared default vs tuned threshold behavior.")


if __name__ == "__main__":
    main()