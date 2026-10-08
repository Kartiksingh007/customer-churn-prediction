from pathlib import Path
import json
import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "churn_model.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
FEATURE_PATH = MODEL_DIR / "feature_columns.joblib"
INFO_PATH = MODEL_DIR / "model_info.json"

app = Flask(__name__)

model = None
scaler = None
features = None
model_info = None


def load_artifacts():
    global model, scaler, features, model_info

    missing = [
        p.name for p in [MODEL_PATH, SCALER_PATH, FEATURE_PATH, INFO_PATH]
        if not p.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Model artifacts missing: "
            + ", ".join(missing)
            + ". Run `python generate_data.py` and `python train.py` first."
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURE_PATH)

    with open(INFO_PATH, "r") as f:
        model_info = json.load(f)


def validate_input(data):
    values = {}
    for feature in features:
        if feature not in data:
            raise ValueError(f"Missing field: {feature}")

        try:
            values[feature] = float(data[feature])
        except (TypeError, ValueError):
            raise ValueError(f"{feature} must be numeric.")

    if values["recency"] < 0:
        raise ValueError("recency cannot be negative.")
    if values["frequency"] < 0:
        raise ValueError("frequency cannot be negative.")
    if values["monetary"] < 0:
        raise ValueError("monetary cannot be negative.")
    if values["avg_order_value"] < 0:
        raise ValueError("avg_order_value cannot be negative.")
    if values["avg_quantity"] < 0:
        raise ValueError("avg_quantity cannot be negative.")
    if values["customer_lifetime_days"] < 0:
        raise ValueError("customer_lifetime_days cannot be negative.")

    return values


@app.route("/")
def home():
    return render_template(
        "index.html",
        model_name=model_info["selected_model"] if model_info else "Not loaded",
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model": model_info["selected_model"],
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Send a JSON body."}), 400

        values = validate_input(data)

        X = np.array([[values[f] for f in features]], dtype=float)
        X_scaled = scaler.transform(X)

        prediction = int(model.predict(X_scaled)[0])
        probability = float(model.predict_proba(X_scaled)[0][1])

        return jsonify(
            {
                "churn": prediction,
                "prediction": "Churned" if prediction == 1 else "Not Churned",
                "churn_probability": round(probability, 4),
                "risk_level": (
                    "High"
                    if probability >= 0.70
                    else "Medium"
                    if probability >= 0.40
                    else "Low"
                ),
                "model": model_info["selected_model"],
            }
        )

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500


if __name__ == "__main__":
    load_artifacts()
    app.run(host="127.0.0.1", port=5000, debug=True)
