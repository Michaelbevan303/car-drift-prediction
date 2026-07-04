import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "..", "model_artifact.joblib")

artifact = joblib.load(ARTIFACT_PATH)
model = artifact["model"]
threshold = artifact["threshold"]
feature_order = artifact["features"]

app = FastAPI(title="Car Drift Prediction API")


class DriftInput(BaseModel):
    Vx: float = Field(..., description="Vehicle forward speed (km/h)")
    Ax: float = Field(..., description="Longitudinal acceleration (m/s^2)")
    Ay: float = Field(..., description="Lateral acceleration (m/s^2)")
    Az: float = Field(..., description="Vertical acceleration (m/s^2)")
    steering_angle: float = Field(..., description="Steering input (degrees)")
    yaw_rate: float = Field(..., description="Yaw rate (deg/s)")
    slip_angle: float = Field(..., description="Slip angle (degrees)")
    V_fl: float = Field(..., description="Front-left wheel speed (km/h)")
    V_fr: float = Field(..., description="Front-right wheel speed (km/h)")
    V_rl: float = Field(..., description="Rear-left wheel speed (km/h)")
    V_rr: float = Field(..., description="Rear-right wheel speed (km/h)")
    tire_slip_ratio: float = Field(..., description="Normalized rear/front wheel speed difference")
    throttle: float = Field(..., description="Throttle position (%)")
    brake: float = Field(..., description="Brake pressure (%)")
    gear: int = Field(..., description="Current gear (1-6)")
    road_curvature: float = Field(..., description="Road curvature (1/m)")
    road_friction: float = Field(..., description="Road friction coefficient")
    road_slope: float = Field(..., description="Road slope / banking angle (degrees)")
    Ay_Ax_ratio: float = Field(..., description="Ratio of lateral to longitudinal acceleration")
    normalized_yaw_rate: float = Field(..., description="Yaw rate per unit speed")
    slip_angle_approx: float = Field(..., description="Approximated slip angle atan(Ay/Vx) (radians)")
    rear_traction_loss: float = Field(..., description="Difference between rear and front wheel speeds")


class DriftOutput(BaseModel):
    drift_probability: float
    drift: bool
    verdict: str
    threshold: float


@app.get("/api/health")
def health():
    return {"status": "ok", "threshold": threshold, "n_features": len(feature_order)}


@app.post("/api/predict", response_model=DriftOutput)
def predict(payload: DriftInput):
    row = payload.model_dump()
    try:
        X = pd.DataFrame([row])[feature_order]
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Missing feature: {exc}")

    prob = float(model.predict_proba(X)[:, 1][0])
    is_drift = prob >= threshold

    return DriftOutput(
        drift_probability=round(prob, 4),
        drift=is_drift,
        verdict="Drift" if is_drift else "No Drift",
        threshold=threshold,
    )
