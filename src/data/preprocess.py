import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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
    
    print(f"Train_scaled Shape: {X_train_scaled.shape}")
    print(f"Test_scaled Shape: {X_test_scaled.shape}")

    joblib.dump(
        preprocessor,
        os.path.join(os.path.dirname(__file__), "preprocess_scaler.pkl"))

    smote = SMOTE(random_state = 42)

    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_scaled,
        y_train
    )
    print(f"Done")

    train_df = pd.DataFrame(
        X_train_resampled
    )
    train_df["is_fraud"] = y_train_resampled

    test_df = pd.DataFrame(
        X_test_scaled
    )
    test_df["is_fraud"] = y_test.reset_index(drop=True)

    train_df.to_csv(
        "data/processed/train.csv",
        index=False
    )
    test_df.to_csv(
        "data/processed/test.csv",
        index=False
    )

    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()
