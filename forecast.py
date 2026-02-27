import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "energy_forecast_model.pkl")

model = joblib.load(MODEL_PATH)

def predict_30_days(input_df: pd.DataFrame) -> pd.DataFrame:
    predictions = model.predict(
        input_df[["day_index", "month", "day_of_week", "lag_1"]]
    )

    output = input_df.copy()
    output["predicted_kwh"] = predictions

    return output