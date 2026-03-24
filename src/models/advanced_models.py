from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


TRAIN_PATH = Path("artifacts/splits/train.csv")
VALID_PATH = Path("artifacts/splits/valid.csv")
RANDOM_STATE = 42


def load_split(path: Path):
    df = pd.read_csv(path)
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def build_numeric_preprocessor(feature_names):
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    return ColumnTransformer(
        transformers=[("num", numeric_pipeline, feature_names)]
    )


def build_adaboost_pipeline(feature_names):
    preprocessor = build_numeric_preprocessor(feature_names)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                AdaBoostClassifier(
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_random_forest_pipeline(feature_names):
    preprocessor = build_numeric_preprocessor(feature_names)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_xgboost_pipeline(feature_names):
    preprocessor = build_numeric_preprocessor(feature_names)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    random_state=RANDOM_STATE,
                    n_estimators=300,
                    max_depth=5,
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


def evaluate_ranking_metrics(model, X, y, dataset_name):
    y_scores = model.predict_proba(X)[:, 1]
    return {
        "dataset": dataset_name,
        "pr_auc": average_precision_score(y, y_scores),
        "roc_auc": roc_auc_score(y, y_scores),
    }


def run_grid_search(model_name, pipeline, param_grid, X_train, y_train, X_valid, y_valid):
    print(f"\n=== TUNING {model_name.upper()} ===")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)

    cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    print(cv_results[["rank_test_score", "mean_test_score", "std_test_score", "params"]].head(5))
    print("\nBest params")
    print(search.best_params_)
    print("\nBest mean CV PR-AUC")
    print(search.best_score_)

    train_metrics = evaluate_ranking_metrics(search.best_estimator_, X_train, y_train, "train")
    valid_metrics = evaluate_ranking_metrics(search.best_estimator_, X_valid, y_valid, "valid")

    print("\nTrain ranking metrics")
    print(pd.DataFrame([train_metrics]))
    print("\nValidation ranking metrics")
    print(pd.DataFrame([valid_metrics]))

    return {
        "model_name": model_name,
        "search": search,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
    }


def main():
    X_train, y_train = load_split(TRAIN_PATH)
    X_valid, y_valid = load_split(VALID_PATH)
    feature_names = X_train.columns.tolist()

    results = []

    rf_pipeline = build_random_forest_pipeline(feature_names)
    rf_param_grid = {
        "classifier__n_estimators": [200, 400],
        "classifier__max_depth": [8, 12],
        "classifier__min_samples_leaf": [1, 5],
    }
    results.append(run_grid_search("random_forest", rf_pipeline, rf_param_grid, X_train, y_train, X_valid, y_valid))

    ada_pipeline = build_adaboost_pipeline(feature_names)
    ada_param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__learning_rate": [0.5, 1.0],
    }
    results.append(run_grid_search("adaboost", ada_pipeline, ada_param_grid, X_train, y_train, X_valid, y_valid))

    if XGBOOST_AVAILABLE:
        xgb_pipeline = build_xgboost_pipeline(feature_names)
        xgb_param_grid = {
            "classifier__n_estimators": [200, 300],
            "classifier__max_depth": [4, 6],
            "classifier__learning_rate": [0.03, 0.05],
        }
        results.append(run_grid_search("xgboost", xgb_pipeline, xgb_param_grid, X_train, y_train, X_valid, y_valid))
    else:
        print("\n=== XGBOOST NOT AVAILABLE ===")
        print("Install it with: pip install xgboost")

    summary_rows = []
    for result in results:
        summary_rows.append(
            {
                "model": result["model_name"],
                "best_cv_pr_auc": result["search"].best_score_,
                "valid_pr_auc": result["valid_metrics"]["pr_auc"],
                "valid_roc_auc": result["valid_metrics"]["roc_auc"],
                "best_params": result["search"].best_params_,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("valid_pr_auc", ascending=False)
    print("\n=== ADVANCED MODEL COMPARISON SUMMARY ===")
    print(summary_df)


if __name__ == "__main__":
    main()