from pathlib import Path
import logging
import warnings

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# =========================================================
# 1. APP CONFIGURATION
# =========================================================
app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "best_knn_model.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor_scaler.joblib"

# Baseline threshold only. Tune using validation data if available.
THRESHOLD = 0.50

# These names and order match the X columns used in the supplied notebook.
FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

INTEGER_FEATURES = {
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "slope", "ca", "thal",
}

# Broad input bounds; categorical bounds are checked against dataset values below.
INPUT_BOUNDS = {
    "age": (1, 120),
    "sex": (0, 1),
    "cp": (0, 3),
    "trestbps": (50, 300),
    "chol": (50, 700),
    "fbs": (0, 1),
    "restecg": (0, 2),
    "thalach": (40, 250),
    "exang": (0, 1),
    "oldpeak": (-5, 10),
    "slope": (0, 2),
    "ca": (0, 4),
    "thal": (0, 3),
}

# =========================================================
# 2. LOAD MODEL AND PREPROCESSOR
# =========================================================
def load_artifacts():
    """Load trusted model artifacts saved by the training notebook."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not PREPROCESSOR_PATH.is_file():
        raise FileNotFoundError(f"Preprocessor file not found: {PREPROCESSOR_PATH}")

    # joblib/pickle files should only be loaded if you trust their source.
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        loaded_model = joblib.load(MODEL_PATH)
        loaded_preprocessor = joblib.load(PREPROCESSOR_PATH)

    if not hasattr(loaded_model, "predict_proba"):
        raise TypeError("The saved KNN model does not support predict_proba().")
    if not hasattr(loaded_model, "classes_"):
        raise TypeError("The saved model has no classes_ attribute.")
    if 1 not in list(loaded_model.classes_):
        raise ValueError("Positive target class 1 was not found in model.classes_.")
    if not hasattr(loaded_preprocessor, "transform"):
        raise TypeError("The saved preprocessor does not support transform().")

    # Validate expected feature count and, when available, names/order.
    expected_count = getattr(loaded_preprocessor, "n_features_in_", None)
    if expected_count is not None and expected_count != len(FEATURE_NAMES):
        raise ValueError(
            f"Preprocessor expects {expected_count} features, but app provides "
            f"{len(FEATURE_NAMES)}."
        )

    scaler_names = getattr(loaded_preprocessor, "feature_names_in_", None)
    if scaler_names is not None and list(scaler_names) != FEATURE_NAMES:
        raise ValueError(
            "Feature names/order in app.py do not match the saved preprocessor: "
            f"{list(scaler_names)}"
        )

    logger.info("Model and preprocessor loaded successfully.")
    logger.info("Model classes: %s", list(loaded_model.classes_))
    return loaded_model, loaded_preprocessor

    
# =========================================================
# 3. INITIALIZE MODEL
# =========================================================
try:
    model, preprocessor = load_artifacts()
except Exception:
    logger.exception("Failed to load model or preprocessor.")
    raise


# =========================================================
# 4. ROUTES
# =========================================================
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return {
        "status": "ready",
        "model": type(model).__name__,
        "features": len(FEATURE_NAMES),
    }


@app.route("/predict", methods=["POST"])
def predict():
    is_ajax = (
        request.headers.get("X-Requested-With") == "XMLHttpRequest"
        or request.accept_mimetypes.best == "application/json"
    )

    def respond(payload, status=200):
        if is_ajax:
            return jsonify(payload), status

        if "error" in payload:
            return render_template("index.html", error=payload["error"]), status

        return render_template(
            "index.html",
            prediction=payload["prediction"],
            probability=payload["probability"],
        ), status

    try:
        values = {}

        for feature in FEATURE_NAMES:
            raw_value = request.form.get(feature, "").strip()

            if not raw_value:
                raise ValueError(f"Please provide a value for {feature}.")

            value = float(raw_value)

            if not np.isfinite(value):
                raise ValueError(f"{feature} must be a valid number.")

            lower, upper = INPUT_BOUNDS[feature]
            if not lower <= value <= upper:
                raise ValueError(
                    f"{feature} must be between {lower} and {upper}."
                )

            if feature in INTEGER_FEATURES and not value.is_integer():
                raise ValueError(f"{feature} must be a whole number.")

            values[feature] = (
                int(value) if feature in INTEGER_FEATURES else value
            )

        input_df = pd.DataFrame([values], columns=FEATURE_NAMES)
        scaled_input = preprocessor.transform(input_df)

        probabilities = model.predict_proba(scaled_input)[0]
        positive_index = list(model.classes_).index(1)
        probability = float(probabilities[positive_index])

        prediction = "Yes" if probability >= THRESHOLD else "No"

        return respond({
            "success": True,
            "prediction": prediction,
            "probability": probability,
            "threshold": THRESHOLD,
        })

    except ValueError as exc:
        return respond({"success": False, "error": str(exc)}, 400)

    except Exception:
        logger.exception("Prediction failed.")
        return respond({
            "success": False,
            "error": "An unexpected error occurred. Please try again.",
        }, 500)


# =========================================================
# 5. RUN FLASK SERVER
# =========================================================

if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

