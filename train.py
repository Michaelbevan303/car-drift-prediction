# ==============================
# CAR DRIFT PREDICTION - Training Script
#   1. Train / Validation / Test split (60/20/20)
#   2. Threshold tuned on VALIDATION set (honest test metrics)
#   3. Model + threshold saved to a file for deployment
# ==============================

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, accuracy_score

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# ==============================
# Load Dataset
# ==============================
data = pd.read_csv("synthetic_car_drift.csv")
data = data.dropna()

# ==============================
# Split Features and Target
# ==============================
X = data.drop("drift", axis=1)
y = data["drift"]

# ==============================
# Train / Validation / Test Split (60% / 20% / 20%)
# ==============================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print("Train size:", len(X_train))
print("Val size  :", len(X_val))
print("Test size :", len(X_test))

# ==============================
# Pipeline (SMOTE + Random Forest)
# ==============================
pipeline = Pipeline([
    ("smote", SMOTE(sampling_strategy=0.7, random_state=42)),
    ("rf", RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

param_dist = {
    "rf__n_estimators": [100, 200],
    "rf__max_depth": [10, 15, 20],
    "rf__min_samples_split": [5, 10],
    "rf__min_samples_leaf": [3, 5, 10],
    "rf__max_features": ["sqrt"]
}

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=20,
    scoring="f1",
    cv=5,
    random_state=42,
    verbose=1,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("\nBest Hyperparameters:")
print(random_search.best_params_)

best_model = random_search.best_estimator_

# ==============================
# Threshold Tuning (on VALIDATION set - no test leakage)
# ==============================
val_prob = best_model.predict_proba(X_val)[:, 1]

best_f1 = 0
best_threshold = 0.5

for t in np.arange(0.2, 0.8, 0.01):
    y_thresh = (val_prob >= t).astype(int)
    f1 = f1_score(y_val, y_thresh)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"\nBest Threshold (from validation): {best_threshold:.2f}")
print(f"Validation F1 at this threshold : {best_f1:.4f}")

# ==============================
# FINAL Evaluation on TEST set (honest numbers)
# ==============================
test_prob = best_model.predict_proba(X_test)[:, 1]
y_final = (test_prob >= best_threshold).astype(int)

print("\n========== FINAL TEST METRICS (trust these) ==========\n")
print(classification_report(y_test, y_final, target_names=["No Drift", "Drift"]))

train_pred = best_model.predict(X_train)
print("Training Accuracy:", accuracy_score(y_train, train_pred))
print("Testing Accuracy :", accuracy_score(y_test, y_final))
print("Training F1:", f1_score(y_train, train_pred))
print("Testing F1 :", f1_score(y_test, y_final))

# ==============================
# Save Model for Deployment
#     (model + threshold + feature order in ONE file)
# ==============================
artifact = {
    "model": best_model,
    "threshold": float(best_threshold),
    "features": list(X.columns),
    "metrics": {
        "test_accuracy": float(accuracy_score(y_test, y_final)),
        "test_f1": float(f1_score(y_test, y_final))
    }
}

joblib.dump(artifact, "model_artifact.joblib", compress=3)

print("\nSaved model_artifact.joblib "
      f"({os.path.getsize('model_artifact.joblib') / 1e6:.1f} MB)")

# ==============================
# Sanity Check - reload and predict a few test rows
# ==============================
loaded = joblib.load("model_artifact.joblib")

sample = X_test.iloc[:5][loaded["features"]]
sample_prob = loaded["model"].predict_proba(sample)[:, 1]
sample_pred = (sample_prob >= loaded["threshold"]).astype(int)

check = sample.copy()
check["Drift_Probability"] = sample_prob.round(4)
check["Prediction"] = ["Drift" if p == 1 else "No Drift" for p in sample_pred]
check["Actual"] = ["Drift" if a == 1 else "No Drift" for a in y_test.iloc[:5]]

print("\nSanity check on 5 real test rows using the SAVED model:\n")
print(check[["Drift_Probability", "Prediction", "Actual"]])

import sklearn, imblearn
print("\nscikit-learn version:", sklearn.__version__)
print("imbalanced-learn version:", imblearn.__version__)
