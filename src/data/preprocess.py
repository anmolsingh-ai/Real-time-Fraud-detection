import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from imblearn.over_sampling import SMOTE

RAW_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "credit_card_fraud_10k.csv"))
RANDOM_STATE = 42

def main():
    print("Loading dataset...")

    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Dataset Shape: {df.shape}")

    # Basic checks
    print("Checking missing values...")
    print(df.isnull().sum())

    # Feature engineering (use shared utilities)
    print("Applying feature engineering...")
    #df = engineer_features(df) 

    
    features_to_scale = ['amount', 'transaction_hour', 'device_trust_score', 
                         'velocity_last_24h', 'cardholder_age']
    features_to_encode = ['merchant_category']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), features_to_scale),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features_to_encode)
        ],
        remainder='passthrough' # This keeps your binary flags (0/1) as they are
    )

    # 3. Define X and y
    X = df.drop(['transaction_id', 'is_fraud'], axis=1)
    y = df['is_fraud']

    # 4. Split and Transform
    X_train, X_test, y_train, y_test = train_test_split(
           X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train Shape: {X_train.shape}")
    print(f"Test Shape: {X_test.shape}")

    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    print(f"Train Shape: {X_train_scaled.shape}")
    print(f"Test Shape: {X_test_scaled.shape}")


if __name__ == "__main__":
    main()


"""import os
import sys
import joblib
import mlflow
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE


# ===================================================
# CONFIG & PATHS (OS-portable)
# ===================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # src

RAW_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "credit_card_fraud_10k.csv")

PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")

MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

ENCODER_PATH = os.path.join(MODEL_DIR, "merchant_encoder.pkl")

RANDOM_STATE = 42


# Ensure dirs exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# Make `src` importable so we can reuse feature utilities
sys.path.append(BASE_DIR)
try:
    from features.feature_engineering import engineer_features, encode_merchant_category
except Exception:
    # fallback: try relative import path
    from src.features.feature_engineering import engineer_features, encode_merchant_category


def main():
    print("Loading dataset...")

    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Dataset Shape: {df.shape}")

    # Basic checks
    print("Checking missing values...")
    print(df.isnull().sum())

    # Feature engineering (use shared utilities)
    print("Applying feature engineering...")
    df = engineer_features(df)

    # Split features/target
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Train Shape: {X_train.shape}")
    print(f"Test Shape: {X_test.shape}")

    # Encode categorical: fit on train only to avoid leakage
    print("Encoding merchant category using training data only...")
    X_train_enc, encoder = encode_merchant_category(X_train, encoder_path=ENCODER_PATH)

    # transform test using fitted encoder
    X_test_enc = X_test.copy()
    X_test_enc["merchant_category_encoded"] = encoder.transform(
        X_test_enc["merchant_category"].astype(str)
    )

    # Drop raw categorical / id columns
    drop_cols = [c for c in ["transaction_id", "merchant_category"] if c in X_train_enc.columns]
    X_train_enc = X_train_enc.drop(columns=drop_cols)
    X_test_enc = X_test_enc.drop(columns=drop_cols)

    # Feature scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enc)
    X_test_scaled = scaler.transform(X_test_enc)

    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to: {SCALER_PATH}")

    # Handle imbalanced data with SMOTE
    print("Applying SMOTE to training set...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    print("Class distribution after SMOTE:")
    print(pd.Series(y_train_resampled).value_counts())

    # Convert arrays back to DataFrames using the encoded columns
    columns = X_train_enc.columns.tolist()
    X_train_processed = pd.DataFrame(X_train_resampled, columns=columns)
    X_test_processed = pd.DataFrame(X_test_scaled, columns=columns)

    train_df = X_train_processed.copy()
    train_df["is_fraud"] = y_train_resampled

    test_df = X_test_processed.copy()
    test_df["is_fraud"] = y_test.reset_index(drop=True)

    train_csv = os.path.join(PROCESSED_DATA_DIR, "train.csv")
    test_csv = os.path.join(PROCESSED_DATA_DIR, "test.csv")

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"Saved processed datasets to {PROCESSED_DATA_DIR}")

    # MLflow logging
    print("Logging preprocessing run to MLflow...")
    mlflow.set_experiment("fraud_detection_preprocessing")

    with mlflow.start_run(run_name="feature_engineering_pipeline"):
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("smote", True)
        mlflow.log_param("scaler", "StandardScaler")
        mlflow.log_param("merchant_encoding", "LabelEncoder")

        mlflow.log_metric("train_rows_after_smote", len(train_df))
        mlflow.log_metric("test_rows", len(test_df))
        mlflow.log_metric("fraud_cases_train", int(sum(y_train_resampled)))

        # Artifacts: scaler, encoder, processed csvs
        if os.path.exists(SCALER_PATH):
            mlflow.log_artifact(SCALER_PATH)

        if os.path.exists(ENCODER_PATH):
            mlflow.log_artifact(ENCODER_PATH)

        mlflow.log_artifact(train_csv)
        mlflow.log_artifact(test_csv)

    print("MLflow logging complete.")


if __name__ == "__main__":
    main()"""