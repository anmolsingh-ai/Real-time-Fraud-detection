import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

TRAIN_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "train.csv"))
TEST_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "test.csv"))

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))

os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Loading processed data ....")

train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)

print(f"Train Shape: {train_df.shape}")
print(f"Test Shape: {test_df.shape}")

X_train = train_df.drop("is_fraud", axis=1)
y_train = train_df["is_fraud"]
X_test = test_df.drop("is_fraud", axis=1)
y_test = test_df["is_fraud"]

mlflow.set_experiment("credit_card_fraud_detection")

with mlflow.start_run(run_name="XGBoost Classifier"):
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("\nModel Evaluation:\n")

    print(f"Accuracy : {accuracy:.4f}")

    print(f"Precision: {precision:.4f}")

    print(f"Recall   : {recall:.4f}")

    print(f"F1 Score : {f1:.4f}")

    print(f"ROC AUC  : {roc_auc:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("learning_rate", 0.1)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    model_path = os.path.join(MODEL_DIR, "xgboost_model.pkl")
    
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")

    mlflow.log_artifact(model_path)
    mlflow.sklearn.log_model(model, "xgboost_model")   

print("\nTraining pipeline completed successfully.") 
        