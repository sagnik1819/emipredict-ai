"""
Model Training & Evaluation Pipeline for EMIPredict AI.
Trains 3+ Classification Models (Logistic Regression, Random Forest, Gradient Boosting)
and 3+ Regression Models (Ridge, Random Forest Regressor, Gradient Boosting Regressor).
Evaluates against production financial criteria (>90% accuracy, <2000 RMSE),
logs all metrics to MLflow, and registers champion models.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)

from .config import (
    CLASSIFICATION_TARGET,
    REGRESSION_TARGET,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TARGET_ACCURACY,
    TARGET_REGRESSION_RMSE
)
from .data_generator import generate_emi_dataset, save_dataset
from .preprocessing import clean_raw_data, engineer_financial_features, PreprocessingPipeline, prepare_data_splits
from .mlflow_integration import MLflowTracker

def evaluate_classification(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Calculates full multi-class classification benchmark metrics."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    
    roc_auc = 0.0
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
            roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
        except Exception:
            roc_auc = 0.0

    return {
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "roc_auc_ovr": float(roc_auc)
    }

def evaluate_regression(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Calculates regression error metrics."""
    y_pred = model.predict(X_test)
    # Clip negative predictions to safe minimum 500
    y_pred = np.clip(y_pred, 500.0, 50_000.0)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2_score": float(r2),
        "mape": float(mape)
    }


def train_and_evaluate_all(
    dataset_path: Path = None,
    sample_for_quick_train: int = 60_000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    End-to-end training orchestrator:
    1. Loads or generates 400k dataset
    2. Builds Preprocessor pipeline
    3. Trains 3 classification models
    4. Trains 3 regression models
    5. Logs to MLflow tracker
    6. Identifies best models and saves them for production deployment
    """
    tracker = MLflowTracker()
    clf_dir = MODELS_DIR / "classification"
    reg_dir = MODELS_DIR / "regression"
    clf_dir.mkdir(parents=True, exist_ok=True)
    reg_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Loading
    csv_file = RAW_DATA_DIR / "emi_dataset_400k.csv"
    sample_file = RAW_DATA_DIR / "emi_dataset_sample_50k.csv"

    if dataset_path and Path(dataset_path).exists():
        raw_df = pd.read_csv(dataset_path)
    elif csv_file.exists():
        print(f"Loading existing dataset from {csv_file}...")
        raw_df = pd.read_csv(csv_file)
    elif sample_file.exists():
        print(f"Loading existing sample dataset from {sample_file}...")
        raw_df = pd.read_csv(sample_file)
    else:
        print("Dataset not found on disk. Generating full 400,000 records dataset...")
        raw_df = generate_emi_dataset(total_records_per_scenario=80_000, seed=seed)
        save_dataset(raw_df)

    print(f"Dataset successfully loaded. Total records: {len(raw_df):,}")

    # Subsample for fast training while retaining representative distribution
    if sample_for_quick_train and len(raw_df) > sample_for_quick_train:
        print(f"Sampling {sample_for_quick_train:,} records for high-speed training...")
        train_pool = raw_df.sample(n=sample_for_quick_train, random_state=seed).reset_index(drop=True)
    else:
        train_pool = raw_df

    # 2. Splits and Preprocessor
    train_df, val_df, test_df = prepare_data_splits(train_pool, test_size=0.15, val_size=0.15, seed=seed)
    
    preprocessor = PreprocessingPipeline()
    preprocessor.fit(train_df)
    preprocessor.save(MODELS_DIR / "preprocessor.joblib")

    X_train = preprocessor.transform(train_df)
    X_val = preprocessor.transform(val_df)
    X_test = preprocessor.transform(test_df)

    # Map classification targets
    y_clf_train = train_df[CLASSIFICATION_TARGET].map(preprocessor.class_mapping).values
    y_clf_val = val_df[CLASSIFICATION_TARGET].map(preprocessor.class_mapping).values
    y_clf_test = test_df[CLASSIFICATION_TARGET].map(preprocessor.class_mapping).values

    # Regression targets
    y_reg_train = train_df[REGRESSION_TARGET].values
    y_reg_val = val_df[REGRESSION_TARGET].values
    y_reg_test = test_df[REGRESSION_TARGET].values

    # -------------------------------------------------------------
    # 3. CLASSIFICATION MODEL SUITE (3 Models)
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("TRAINING CLASSIFICATION MODELS (EMI ELIGIBILITY)")
    print("="*50)

    classification_models = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, C=1.0, random_state=seed),
            "params": {"solver": "lbfgs", "C": 1.0, "max_iter": 1000}
        },
        "Random Forest Classifier": {
            "model": RandomForestClassifier(n_estimators=100, max_depth=16, random_state=seed, n_jobs=-1),
            "params": {"n_estimators": 100, "max_depth": 16, "criterion": "gini"}
        },
        "Gradient Boosting Classifier": {
            "model": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_depth=12, random_state=seed),
            "params": {"max_iter": 150, "learning_rate": 0.08, "max_depth": 12}
        }
    }

    clf_results = {}
    best_clf_name = None
    best_clf_acc = 0.0

    for name, item in classification_models.items():
        t0 = time.time()
        print(f"Training {name}...")
        model = item["model"]
        model.fit(X_train, y_clf_train)
        train_time = round(time.time() - t0, 2)

        # Evaluate on test set
        metrics = evaluate_classification(model, X_test, y_clf_test)
        metrics["training_time_sec"] = train_time
        clf_results[name] = {"model": model, "metrics": metrics, "params": item["params"]}

        print(f"-> {name} | Accuracy: {metrics['accuracy']*100:.2f}% | F1: {metrics['f1_macro']:.4f} | AUC: {metrics['roc_auc_ovr']:.4f} ({train_time}s)")

        # Log run in MLflow tracker
        tracker.log_model_run(
            experiment_name=MLflowTracker.EXPERIMENT_CLF,
            run_name=name,
            params=item["params"],
            metrics=metrics,
            tags={"algorithm": name, "task": "classification", "framework": "scikit-learn"}
        )

        # Save individual model
        filename = f"{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, clf_dir / filename)

        if metrics["accuracy"] > best_clf_acc:
            best_clf_acc = metrics["accuracy"]
            best_clf_name = name

    # Save Champion Classification Model
    best_clf_model = clf_results[best_clf_name]["model"]
    best_clf_path = clf_dir / "best_classification_model.joblib"
    joblib.dump(best_clf_model, best_clf_path)
    # Also save in root models dir for instant discovery
    joblib.dump(best_clf_model, MODELS_DIR / "best_classification_model.joblib")

    # Promote in Registry
    tracker.promote_to_production(
        task_type="classification",
        model_name=best_clf_name,
        metrics=clf_results[best_clf_name]["metrics"],
        model_file="best_classification_model.joblib"
    )

    # -------------------------------------------------------------
    # 4. REGRESSION MODEL SUITE (3 Models)
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("TRAINING REGRESSION MODELS (MAX SAFE MONTHLY EMI)")
    print("="*50)

    regression_models = {
        "Ridge Linear Regression": {
            "model": Ridge(alpha=1.0, random_state=seed),
            "params": {"alpha": 1.0, "solver": "auto"}
        },
        "Random Forest Regressor": {
            "model": RandomForestRegressor(n_estimators=100, max_depth=16, random_state=seed, n_jobs=-1),
            "params": {"n_estimators": 100, "max_depth": 16}
        },
        "Gradient Boosting Regressor": {
            "model": HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_depth=12, random_state=seed),
            "params": {"max_iter": 150, "learning_rate": 0.08, "max_depth": 12}
        }
    }

    reg_results = {}
    best_reg_name = None
    best_reg_rmse = float("inf")

    for name, item in regression_models.items():
        t0 = time.time()
        print(f"Training {name}...")
        model = item["model"]
        model.fit(X_train, y_reg_train)
        train_time = round(time.time() - t0, 2)

        # Evaluate on test set
        metrics = evaluate_regression(model, X_test, y_reg_test)
        metrics["training_time_sec"] = train_time
        reg_results[name] = {"model": model, "metrics": metrics, "params": item["params"]}

        print(f"-> {name} | RMSE: INR {metrics['rmse']:,.1f} | MAE: INR {metrics['mae']:,.1f} | R2: {metrics['r2_score']:.4f} ({train_time}s)")

        # Log run in MLflow tracker
        tracker.log_model_run(
            experiment_name=MLflowTracker.EXPERIMENT_REG,
            run_name=name,
            params=item["params"],
            metrics=metrics,
            tags={"algorithm": name, "task": "regression", "framework": "scikit-learn"}
        )

        # Save individual model
        filename = f"{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, reg_dir / filename)

        if metrics["rmse"] < best_reg_rmse:
            best_reg_rmse = metrics["rmse"]
            best_reg_name = name

    # Save Champion Regression Model
    best_reg_model = reg_results[best_reg_name]["model"]
    best_reg_path = reg_dir / "best_regression_model.joblib"
    joblib.dump(best_reg_model, best_reg_path)
    joblib.dump(best_reg_model, MODELS_DIR / "best_regression_model.joblib")

    # Promote in Registry
    tracker.promote_to_production(
        task_type="regression",
        model_name=best_reg_name,
        metrics=reg_results[best_reg_name]["metrics"],
        model_file="best_regression_model.joblib"
    )

    print("\n" + "="*50)
    print("TRAINING PIPELINE SUMMARY")
    print(f"Champion Classification Model: {best_clf_name} (Accuracy: {best_clf_acc*100:.2f}%) [Target > {TARGET_ACCURACY*100}%]")
    print(f"Champion Regression Model:     {best_reg_name} (RMSE: INR {best_reg_rmse:,.1f}) [Target < INR {TARGET_REGRESSION_RMSE:,.0f}]")
    print("="*50)

    # Export report summary
    summary = {
        "status": "success",
        "timestamp": pd.Timestamp.now().isoformat(),
        "classification": {
            "best_model": best_clf_name,
            "accuracy": best_clf_acc,
            "meets_target": best_clf_acc >= TARGET_ACCURACY,
            "all_models": {k: v["metrics"] for k, v in clf_results.items()}
        },
        "regression": {
            "best_model": best_reg_name,
            "rmse": best_reg_rmse,
            "meets_target": best_reg_rmse <= TARGET_REGRESSION_RMSE,
            "all_models": {k: v["metrics"] for k, v in reg_results.items()}
        }
    }

    with open(MODELS_DIR / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    train_and_evaluate_all()
