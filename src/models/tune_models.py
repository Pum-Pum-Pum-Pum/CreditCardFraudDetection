from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TRAIN_PATH = Path("artifacts/splits/train.csv")
VALID_PATH = Path("artifacts/splits/valid.csv")
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

    return Pipeline(
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


def build_random_forest_pipeline(feature_names):
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
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_ranking_metrics(model, X, y, dataset_name: str):
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
    cols_to_show = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "params",
    ]

    print("Top CV results")
    print(cv_results[cols_to_show].head(5))

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
        "cv_results": cv_results,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
    }


def main():
    X_train, y_train = load_split(TRAIN_PATH)
    X_valid, y_valid = load_split(VALID_PATH)
    feature_names = X_train.columns.tolist()

    logistic_pipeline = build_logistic_pipeline(feature_names)
    logistic_param_grid = {
        "classifier__C": [0.1, 1.0, 3.0],
        "classifier__solver": ["liblinear", "lbfgs"],
    }

    rf_pipeline = build_random_forest_pipeline(feature_names)
    rf_param_grid = {
        "classifier__n_estimators": [200, 400],
        "classifier__max_depth": [6, 8, 12],
        "classifier__min_samples_leaf": [1, 5],
    }

    results = []
    results.append(run_grid_search("logistic_regression", logistic_pipeline, logistic_param_grid, X_train, y_train, X_valid, y_valid))
    results.append(run_grid_search("random_forest", rf_pipeline, rf_param_grid, X_train, y_train, X_valid, y_valid))

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
    print("\n=== TUNED MODEL SUMMARY ===")
    print(summary_df)

    winner = summary_df.iloc[0]
    print("\n=== CURRENT WINNER BASED ON VALIDATION PR-AUC ===")
    print(winner)


if __name__ == "__main__":
    main()