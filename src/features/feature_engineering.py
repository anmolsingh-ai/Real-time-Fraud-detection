import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
	"""Create basic engineered features used by the pipeline.

	Adds:
	- `is_night_transaction` (1/0)
	- `high_amount_flag` (1/0, >95th percentile)
	- `log_amount` (log1p of amount)
	- `high_velocity_flag` (1/0 where velocity_last_24h > 10)

	Returns a new DataFrame with the added columns.
	"""
	df = df.copy()

	df["is_night_transaction"] = df["transaction_hour"].apply(
		lambda x: 1 if x >= 22 or x <= 5 else 0
	)

	amount_threshold = df["amount"].quantile(0.95)

	df["high_amount_flag"] = df["amount"].apply(
		lambda x: 1 if x > amount_threshold else 0
	)

	df["log_amount"] = np.log1p(df["amount"])

	df["high_velocity_flag"] = df["velocity_last_24h"].apply(
		lambda x: 1 if x > 10 else 0
	)

	return df


def encode_merchant_category(df: pd.DataFrame, encoder_path: str = None):
	"""Fit a LabelEncoder on `merchant_category` and add `merchant_category_encoded`.

	If `encoder_path` is provided the fitted encoder will be saved with joblib.

	Returns: (df_with_encoding, fitted_encoder)
	"""
	df = df.copy()

	encoder = LabelEncoder()
	df["merchant_category_encoded"] = encoder.fit_transform(df["merchant_category"].astype(str))

	if encoder_path:
		joblib.dump(encoder, encoder_path)

	return df, encoder
