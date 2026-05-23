import pandas as pd
import numpy as np
import joblib


def main():

    df = pd.read_csv(r"E:\fraud-detection\fraud-detection\data\raw\credit_card_fraud_10k.csv")
    
    df["is_night_transaction"] = df["transaction_hour"].apply(
        lambda x: 1 if x >= 22 or x <= 5 else 0
    )
	
    threshold = df["amount"].quantile(0.95)
    df["high_amount_flag"] = df["amount"].apply(
        lambda x: 1 if x > threshold else 0
    )

    df["log_amount"] = np.log1p(df["amount"])
    
    df["high_velocity_flag"] = df["velocity_last_24h"].apply(
        lambda x: 1 if x > 10 else 0
    )
    
    df.to_csv(
        "data/processed/credit_card_fraud_engineered.csv",
        index=False)
    print("Feature engineering completed successfully.")




if __name__ == "__main__":
    main()
	