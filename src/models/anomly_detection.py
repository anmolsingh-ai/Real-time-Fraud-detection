import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

TRAIN_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "train.csv"))
TEST_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "test.csv"))

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))

XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")

print(f"Loading processed data ....")

train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)

print(f"Train Shape: {train_df.shape}")
print(f"Test Shape: {test_df.shape}")

X_train = train_df.drop("is_fraud", axis=1)
y_train = train_df["is_fraud"]
X_test = test_df.drop("is_fraud", axis=1)
y_test = test_df["is_fraud"]

print("Loading trained XGBoost model...")

xgb_model = joblib.load(XGB_MODEL_PATH)

mlflow.set_experiment("fraud_detection_ensemble")

with mlflow.start_run(run_name="xgboost_isolation_forest_ensemble"):
    iso_model = IsolationForest(
        n_estimators=100,
        contamination=0.02,
        random_state=42
    )

    iso_model.fit(X_train)

    iso_predictions = iso_model.predict(X_test)

    "libraries like scikit-learn, anomaly detection models natively output -1 for outliers (anomalies) and 1 for inliers (normal data)."
    iso_predictions = np.where(
    iso_predictions == -1,
    1,
    0
    )


    xgb_probabilities = xgb_model.predict_proba(X_test)[:, 1]

    xgb_predictions = xgb_model.predict(X_test)

    print("Generating ensemble fraud scores...")

    ensemble_scores = (
        0.7 * xgb_probabilities
        +
        0.3 * iso_predictions
    )


    ensemble_predictions = np.where(
        ensemble_scores > 0.5,
        1,
        0
    )
    
    "Model Performance metrices"

    accuracy = accuracy_score(
        y_test,
        ensemble_predictions
    )

    precision = precision_score(
        y_test,
        ensemble_predictions
    )

    recall = recall_score(
        y_test,
        ensemble_predictions
    )

    f1 = f1_score(
        y_test,
        ensemble_predictions
    )

    roc_auc = roc_auc_score(
        y_test,
        ensemble_scores
    )


    
    print("\nEnsemble Model Evaluation:\n")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")


  
    iso_model_path = (
        f"{MODEL_DIR}/isolation_forest.pkl"
    )

    joblib.dump(
        iso_model,
        iso_model_path
    )

    print(
        f"\nIsolation Forest saved to: "
        f"{iso_model_path}"
    )

    mlflow.log_param("xgboost_weight",0.7)
    mlflow.log_param("isolation_forest_weight",0.3)
    mlflow.log_param("contamination",0.02)
    mlflow.log_param("ensemble_threshold",0.5)
    mlflow.log_param("n_estimators",100)

    mlflow.log_metric("accuracy",accuracy)
    mlflow.log_metric("precision",precision)
    mlflow.log_metric("recall",recall)
    mlflow.log_metric("f1_score",f1)
    mlflow.log_metric("roc_auc",roc_auc)


    mlflow.sklearn.log_model(
        sk_model=iso_model,
        artifact_path="isolation_forest_model",
        registered_model_name=
        "IsolationForestFraudModel"
    )

    mlflow.log_artifact(
        iso_model_path
    )


print(
    "\nEnsemble fraud detection "
    "pipeline completed successfully."
)