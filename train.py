from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

REFERENCE_DATE = pd.Timestamp("2026-09-30")
CHURN_THRESHOLD_DAYS = 60
RANDOM_STATE = 42

DATA_PATH = Path("data/transactions.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "avg_order_value",
    "avg_quantity",
    "customer_lifetime_days",
]


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(1)
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").fillna(
        df["unit_price"].median()
    )
    df["total_revenue"] = df["quantity"] * df["unit_price"]

    customer = (
        df.groupby("customer_id")
        .agg(
            last_purchase=("order_date", "max"),
            first_purchase=("order_date", "min"),
            frequency=("order_id", "nunique"),
            monetary=("total_revenue", "sum"),
            avg_quantity=("quantity", "mean"),
        )
        .reset_index()
    )

    customer["recency"] = (
        REFERENCE_DATE - customer["last_purchase"]
    ).dt.days.clip(lower=0)

    customer["customer_lifetime_days"] = (
        customer["last_purchase"] - customer["first_purchase"]
    ).dt.days.clip(lower=0)

    customer["avg_order_value"] = (
        customer["monetary"] / customer["frequency"].replace(0, 1)
    )

    customer["churn"] = (
        customer["recency"] > CHURN_THRESHOLD_DAYS
    ).astype(int)

    return customer


def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob),
    }

    print(f"\n{name}")
    print("-" * len(name))
    for key, value in metrics.items():
        print(f"{key:10s}: {value:.4f}")

    return metrics


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "data/transactions.csv not found. Run: python generate_data.py"
        )

    raw = pd.read_csv(DATA_PATH)
    customer = build_customer_features(raw)

    X = customer[FEATURES].copy()
    y = customer["churn"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # XGBoost handles differently scaled features well, but we save a scaler
    # so the API has a consistent preprocessing artifact and the project
    # demonstrates production-style preprocessing.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=350,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=4,
        ),
    }

    results = {}
    fitted = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        results[name] = evaluate(name, model, X_test_scaled, y_test)
        fitted[name] = model

    # Select primarily by F1, then ROC-AUC.
    best_name = sorted(
        results,
        key=lambda n: (results[n]["f1"], results[n]["roc_auc"]),
        reverse=True,
    )[0]

    best_model = fitted[best_name]

    joblib.dump(best_model, MODEL_DIR / "churn_model.joblib")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    joblib.dump(FEATURES, MODEL_DIR / "feature_columns.joblib")

    info = {
        "selected_model": best_name,
        "churn_threshold_days": CHURN_THRESHOLD_DAYS,
        "features": FEATURES,
        "metrics": results[best_name],
        "all_model_metrics": results,
    }

    with open(MODEL_DIR / "model_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Selected model: {best_name}")
    print("Saved model artifacts in models/")
    print("=" * 60)


if __name__ == "__main__":
    main()
