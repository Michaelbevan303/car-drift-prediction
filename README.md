# Car Drift Prediction

Predicts whether a car is drifting (rear wheels losing traction) from vehicle
telemetry, using a Random Forest classifier trained on a synthetic sensor
dataset. Deployed as a static HTML form backed by a FastAPI serverless
function on Vercel.

- **Test accuracy:** 99.5%
- **Test F1 score:** 0.976
- **Decision threshold:** tuned on a held-out validation set (not the test
  set) to maximize F1, then evaluated once on the test set for an honest
  performance estimate.

## How it works

1. `train.py` loads `synthetic_car_drift.csv`, splits it 60/20/20 into
   train/validation/test, and fits a `SMOTE` + `RandomForestClassifier`
   pipeline via `RandomizedSearchCV` (5-fold CV, optimizing F1).
2. The decision threshold is tuned on the validation set, then the model is
   scored once on the test set.
3. The model, tuned threshold, and feature order are bundled into
   `model_artifact.joblib`.
4. `api/index.py` loads that artifact and exposes it as a FastAPI app with
   `/api/predict` and `/api/health` routes.
5. `index.html` is a static form that posts telemetry values to
   `/api/predict` and renders the drift probability and verdict.

## Features

| Feature | Plain-English meaning | Typical range |
|---|---|---|
| `Vx` | Vehicle forward speed | 20 – 120 km/h |
| `Ax` | Forward (longitudinal) acceleration from braking/accelerating | -5 – 5 m/s² |
| `Ay` | Sideways (lateral) acceleration; high values often trigger drift | -5 – 5 m/s² |
| `Az` | Vertical acceleration; reflects uneven road surfaces | -1 – 1 m/s² |
| `steering_angle` | Driver's steering input | -30 – 30 degrees |
| `yaw_rate` | Rotation rate around the vertical axis; correlates with oversteer | -10 – 10 deg/s |
| `slip_angle` | Difference between vehicle heading and actual velocity direction | -15 – 15 degrees |
| `V_fl` / `V_fr` | Front-left / front-right wheel speed | ≈ `Vx` ± small noise |
| `V_rl` / `V_rr` | Rear-left / rear-right wheel speed; rear slip indicates drift | ≈ `Vx` ± larger noise |
| `tire_slip_ratio` | Rear/front wheel speed difference normalized by vehicle speed | -0.2 – 0.2 |
| `throttle` | Gas pedal position | 0 – 100% |
| `brake` | Brake pedal pressure | 0 – 100% |
| `gear` | Current gear; affects torque delivered to the rear wheels | 1 – 6 |
| `road_curvature` | Sharper curves increase drift likelihood | 0 – 0.05 1/m |
| `road_friction` | Surface grip; lower friction increases drift risk | 0.6 – 1.0 |
| `road_slope` | Road tilt/banking; affects lateral forces | -5 – 5 degrees |
| `Ay_Ax_ratio` | Derived: ratio of lateral to longitudinal acceleration | variable |
| `normalized_yaw_rate` | Derived: yaw rate divided by speed | variable |
| `slip_angle_approx` | Derived: `atan(Ay / Vx)` in radians | variable |
| `rear_traction_loss` | Derived: difference between rear and front wheel speeds | variable |
| `drift` | Target label: 1 = drifting, 0 = not drifting | 0 or 1 |

## Running locally

```bash
python -m venv venv
source venv/Scripts/activate   # or venv/bin/activate on macOS/Linux
pip install -r requirements.txt

# (optional) retrain the model from the CSV
python train.py

# run the API locally
uvicorn api.index:app --reload
```

Then open `index.html` in a browser (or serve it with any static file
server) and point its fetch calls at your local API, or run `vercel dev` to
serve both the static frontend and the API together exactly as they run in
production.

## Deploying to Vercel

```bash
npm install -g vercel
vercel login
vercel --prod
```

`model_artifact.joblib` is committed to the repo (it's small — a few MB) so
the serverless function can load it directly; the raw training CSV is
excluded via `.gitignore` since it isn't needed at runtime.
