"""Addestra baseline classiche sulle feature manuali estratte dai crop."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from nmcd_common import DEFAULT_OUTPUT_ROOT, ensure_dir


NON_FEATURE_COLUMNS = {
    "split",
    "annotation_id",
    "image_id",
    "source_file",
    "category_id",
    "label",
    "crop_file",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline classifiers on handcrafted features.")
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "features" / "handcrafted_features.csv",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "models")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified folds used by grid search on the training split.",
    )
    parser.add_argument(
        "--grid-scoring",
        default="f1_macro",
        help="Scikit-learn scoring name used to select hyperparameters.",
    )
    parser.add_argument(
        "--grid-n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs used by GridSearchCV.",
    )
    parser.add_argument(
        "--no-grid-search",
        action="store_true",
        help="Train the default model configurations without hyperparameter search.",
    )
    return parser.parse_args()


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])
    ]


def build_models(random_state: int) -> dict[str, Pipeline]:
    return {
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "linear_svm": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LinearSVC(
                        class_weight="balanced",
                        dual="auto",
                        max_iter=30000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
}


def parameter_grids() -> dict[str, dict[str, list[object]]]:
    return {
        "random_forest": {
            "model__n_estimators": [300, 500],
            "model__max_depth": [None, 20, 40],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2"],
        },
        "linear_svm": {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__tol": [1e-4, 1e-3],
        },
        "logistic_regression": {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__tol": [1e-4, 1e-3],
        },
    }


def effective_cv_folds(y: pd.Series, requested_folds: int) -> int:
    if requested_folds < 2:
        raise ValueError("--cv-folds must be at least 2.")

    min_class_count = int(y.value_counts().min())
    if min_class_count < 2:
        raise ValueError("Grid search requires at least two training samples for every class.")

    if requested_folds > min_class_count:
        print(
            f"Requested {requested_folds} CV folds, but the rarest class has "
            f"{min_class_count} samples. Using {min_class_count} folds."
        )
        return min_class_count

    return requested_folds


def clean_param_name(name: str) -> str:
    return name.removeprefix("model__")


def tune_models(
    models: dict[str, Pipeline],
    grids: dict[str, dict[str, list[object]]],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int,
    scoring: str,
    n_jobs: int,
    random_state: int,
    output_dir: Path,
) -> dict[str, Pipeline]:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    tuned_models: dict[str, Pipeline] = {}
    best_rows: list[dict[str, object]] = []
    cv_result_frames: list[pd.DataFrame] = []

    for name, model in models.items():
        grid = grids[name]
        candidates = len(ParameterGrid(grid))
        print(f"Grid search {name}: {candidates} candidates, {cv_folds} folds, scoring={scoring}...")
        search = GridSearchCV(
            estimator=model,
            param_grid=grid,
            scoring=scoring,
            cv=cv,
            refit=True,
            n_jobs=n_jobs,
            return_train_score=True,
        )
        search.fit(x_train, y_train)
        tuned_models[name] = search.best_estimator_

        best_row: dict[str, object] = {
            "model": name,
            "best_cv_macro_f1": float(search.best_score_),
            "best_rank": int(search.cv_results_["rank_test_score"][search.best_index_]),
        }
        best_row.update({clean_param_name(key): value for key, value in search.best_params_.items()})
        best_rows.append(best_row)

        cv_results = pd.DataFrame(search.cv_results_)
        parameter_columns = [column for column in cv_results.columns if column.startswith("param_")]
        keep_columns = [
            "mean_fit_time",
            "std_fit_time",
            "mean_score_time",
            "std_score_time",
            "mean_train_score",
            "std_train_score",
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
            *parameter_columns,
        ]
        cv_results = cv_results[keep_columns].copy()
        cv_results.insert(0, "model", name)
        cv_result_frames.append(cv_results.sort_values("rank_test_score"))

        print(f"Best {name} CV macro-F1: {search.best_score_:.4f}")
        print(f"Best {name} params: {search.best_params_}")

    pd.DataFrame(best_rows).to_csv(output_dir / "grid_search_best_params.csv", index=False)
    pd.concat(cv_result_frames, ignore_index=True).to_csv(
        output_dir / "grid_search_results.csv", index=False
    )
    return tuned_models


def evaluate(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict[str, float | str]:
    pred = model.predict(x)
    return {
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
    }


def save_report(
    model_name: str,
    model: Pipeline,
    split: str,
    x: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
) -> None:
    pred = model.predict(x)
    report_text = classification_report(y, pred, zero_division=0)
    (output_dir / f"{model_name}_{split}_classification_report.txt").write_text(
        report_text, encoding="utf-8"
    )

    report_df = pd.DataFrame(classification_report(y, pred, zero_division=0, output_dict=True)).T
    report_df.to_csv(output_dir / f"{model_name}_{split}_classification_report.csv")

    labels = sorted(y.unique())
    matrix = confusion_matrix(y, pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model_name} - {split}")
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_{split}_confusion_matrix.png", dpi=160)
    plt.close()


def save_feature_importance(model_name: str, model: Pipeline, columns: list[str], output_dir: Path) -> None:
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return
    importance = pd.DataFrame(
        {
            "feature": columns,
            "importance": estimator.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(output_dir / f"{model_name}_feature_importance.csv", index=False)


def main() -> None:
    args = parse_args()
    if not args.features_csv.exists():
        raise FileNotFoundError(
            f"Features not found: {args.features_csv}. Run 03_extract_handcrafted_features.py first."
        )

    output_dir = ensure_dir(args.output_root)
    data = pd.read_csv(args.features_csv)
    columns = feature_columns(data)
    if not columns:
        raise ValueError("No numeric feature columns found.")

    train = data[data["split"] == "train"].copy()
    valid = data[data["split"] == "valid"].copy()
    test = data[data["split"] == "test"].copy()

    x_train, y_train = train[columns], train["label"]
    x_valid, y_valid = valid[columns], valid["label"]
    x_test, y_test = test[columns], test["label"]

    metrics = []
    models = build_models(args.random_state)
    if not args.no_grid_search:
        cv_folds = effective_cv_folds(y_train, args.cv_folds)
        models = tune_models(
            models=models,
            grids=parameter_grids(),
            x_train=x_train,
            y_train=y_train,
            cv_folds=cv_folds,
            scoring=args.grid_scoring,
            n_jobs=args.grid_n_jobs,
            random_state=args.random_state,
            output_dir=output_dir,
        )

    best_name = None
    best_valid_macro_f1 = -1.0

    for name, model in models.items():
        print(f"Training {name}...")
        if args.no_grid_search:
            model.fit(x_train, y_train)
        for split, x_split, y_split in [
            ("valid", x_valid, y_valid),
            ("test", x_test, y_test),
        ]:
            split_metrics = evaluate(model, x_split, y_split)
            metrics.append({"model": name, "split": split, **split_metrics})
            save_report(name, model, split, x_split, y_split, output_dir)

        save_feature_importance(name, model, columns, output_dir)
        valid_macro_f1 = [m for m in metrics if m["model"] == name and m["split"] == "valid"][0][
            "macro_f1"
        ]
        if valid_macro_f1 > best_valid_macro_f1:
            best_valid_macro_f1 = float(valid_macro_f1)
            best_name = name

    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(output_dir / "baseline_metrics.csv", index=False)

    if best_name is not None:
        best_model = models[best_name]
        joblib.dump(
            {
                "model": best_model,
                "feature_columns": columns,
                "label_column": "label",
            },
            output_dir / "best_feature_baseline.joblib",
        )
        print(f"Best model by validation macro-F1: {best_name} ({best_valid_macro_f1:.4f})")

    print(f"Wrote model outputs to: {output_dir}")


if __name__ == "__main__":
    main()
