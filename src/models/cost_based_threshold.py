from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline


TRAIN_PATH = Path("artifacts/splits/train.csv")
VALID_PATH = Path("artifacts/splits/valid.csv")
TEST_PATH = Path("artifacts/splits/test.csv")
RANDOM_STATE = 42

# NOTE:
# In most fraud settings, a false negative is usually more expensive than a false positive.
# Example default business assumption below:
FN_COST = 5
FP_COST = 1


def load_split(path: Path):
    df = pd.read_csv(path)
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


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


def evaluate_threshold(y_true, y_scores, threshold, fp_cost=FP_COST, fn_cost=FN_COST):
    y_pred = (y_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = fp * fp_cost + fn * fn_cost

    return {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_scores),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "fp_cost": fp_cost,
        "fn_cost": fn_cost,
        "total_cost": total_cost,
    }


def find_best_cost_threshold(y_true, y_scores, fp_cost=FP_COST, fn_cost=FN_COST):
    candidate_thresholds = [i / 100 for i in range(5, 100, 1)]
    evaluations = [
        evaluate_threshold(y_true, y_scores, thr, fp_cost=fp_cost, fn_cost=fn_cost)
        for thr in candidate_thresholds
    ]
    evaluation_df = pd.DataFrame(evaluations)
    best_row = evaluation_df.sort_values(["total_cost", "threshold"], ascending=[True, False]).iloc[0]
    return best_row, evaluation_df


def main():
    X_train, y_train = load_split(TRAIN_PATH)
    X_valid, y_valid = load_split(VALID_PATH)
    X_test, y_test = load_split(TEST_PATH)

    feature_names = X_train.columns.tolist()
    model = build_best_random_forest(feature_names)
    model.fit(X_train, y_train)

    valid_scores = model.predict_proba(X_valid)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]

    print("=== BUSINESS COST ASSUMPTION ===")
    print(f"FP_COST = {FP_COST}")
    print(f"FN_COST = {FN_COST}")
    print("Interpretation: missing a fraud (FN) is assumed to cost more than falsely flagging a legitimate transaction (FP).")

    best_valid_cost, valid_table = find_best_cost_threshold(y_valid, valid_scores, fp_cost=FP_COST, fn_cost=FN_COST)

    print("\n=== BEST VALIDATION THRESHOLD BY TOTAL BUSINESS COST ===")
    print(pd.DataFrame([best_valid_cost]))

    print("\n=== TOP 10 LOWEST-COST VALIDATION THRESHOLDS ===")
    print(valid_table.sort_values(["total_cost", "threshold"], ascending=[True, False]).head(10))

    test_result = evaluate_threshold(
        y_test,
        test_scores,
        threshold=float(best_valid_cost["threshold"]),
        fp_cost=FP_COST,
        fn_cost=FN_COST,
    )

    print("\n=== TEST PERFORMANCE USING VALIDATION-SELECTED COST THRESHOLD ===")
    print(pd.DataFrame([test_result]))

    print("\nStep 5 complete: threshold selected using explicit business cost assumptions.")


if __name__ == "__main__":
    main()