from pathlib import Path
import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


MODEL_PATH = Path("artifacts/model/xgboost_fraud_pipeline.joblib")
METADATA_PATH = Path("artifacts/model/model_metadata.json")
LOG_PATH = Path("artifacts/model/inference_log.jsonl")


def load_model_and_metadata():
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata


model, metadata = load_model_and_metadata()
FEATURE_NAMES = metadata["feature_names"]
THRESHOLD = metadata["threshold"]


class FraudRequest(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


app = FastAPI(title="Credit Card Fraud Detection API", version="1.0.0")


def log_inference(record: dict):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_name": metadata["model_name"],
        "threshold": THRESHOLD,
    }


@app.post("/predict")
def predict(request: FraudRequest):
    input_dict = request.model_dump()
    df = pd.DataFrame([input_dict])[FEATURE_NAMES]

    fraud_probability = float(model.predict_proba(df)[:, 1][0])
    predicted_class = int(fraud_probability >= THRESHOLD)

    response = {
        "fraud_probability": fraud_probability,
        "predicted_class": predicted_class,
        "threshold": THRESHOLD,
    }

    log_record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input": input_dict,
        "output": response,
    }
    log_inference(log_record)

    return response