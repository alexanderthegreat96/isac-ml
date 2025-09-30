import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import json

class CheaterDetector:
    def __init__(self, model_path: str, scaler_path: str, median_path: str, **stats):
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.threshold = 0.45
        self.stats = stats

        # Load training medians
        with open(median_path, "r") as f:
            self.feature_medians = pd.Series(json.load(f))

        self.feature_columns = [
            "timePlayed", "weaponHits", "headshots", "bodyshots",
            "headshotKills", "npcKills", "headshotsPerHour", "bodyshotsPerHour",
            "weaponHitsPerHour", "suspiciousHeadshotRate", "headshotRatio",
            "headshotKillRatio", "hsOverBs", "isExtremePlaytime"
        ]

    def preprocess(self):
      s = self.stats.copy()
      time_hours = s["timePlayed"]  # already in hours

      # Compute per-hour features from raw totals
      s["weaponHitsPerHour"] = s["weaponHits"] / (time_hours + 1)
      s["headshotsPerHour"] = s["headshots"] / (time_hours + 1)
      s["bodyshotsPerHour"] = s["bodyshots"] / (time_hours + 1)
      s["headshotKillsPerHour"] = s["headshotKills"] / (time_hours + 1)
      s["npcKillsPerHour"] = s["npcKills"] / (time_hours + 1)

      # Derived features
      s["suspiciousHeadshotRate"] = int(s["headshotsPerHour"] > 750)
      s["headshotRatio"] = s["headshots"] / (s["weaponHits"] + 1)
      s["headshotKillRatio"] = s["headshotKills"] / (s["npcKills"] + 1)
      s["hsOverBs"] = s["headshots"] / (s["bodyshots"] + 1)
      s["isExtremePlaytime"] = int(s["timePlayed"] > 5000)

      return pd.DataFrame([s])[self.feature_columns]


    def predict(self):
        X = self.preprocess()

        # Ensure consistent NumPy input
        X_scaled = self.scaler.transform(X)
        X_scaled = np.array(X_scaled, dtype=np.float32)  # cast + fixed dtype

        # Predict with retracing-safe input
        prob = self.model.predict(X_scaled, verbose=0).flatten()[0]
        pred = int(prob > self.threshold)

        # Calculate deviation from median
        deviation = X.iloc[0] - self.feature_medians
        with np.errstate(divide='ignore', invalid='ignore'):
            deviation_percent = ((X.iloc[0] / self.feature_medians) - 1) * 100
            deviation_percent = deviation_percent.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-500, 500).round(2)

        deviation_report = {
            col: {
                "value": float(X.iloc[0][col]),
                "median": float(self.feature_medians[col]),
                "deviation": round(float(deviation[col]), 4),
                "percent_deviation": round(float(deviation_percent[col]), 2)
            }
            for col in self.feature_columns
        }

        return {
            "cheater_probability": round(float(prob), 4),
            "predicted_isCheater": pred,
            "deviation_report": deviation_report
        }

stats = {
    "timePlayed": 4300,
    "weaponHits": 64760757,
    "headshots": 2302384,
    "bodyshots": 62458373,
    "headshotKills": 690716,
    "npcKills": 732276,
    "headshotsPerHour": 475,
    "bodyshotsPerHour": 12909,
    "weaponHitsPerHour": 13385
}

detector = CheaterDetector(
    "v2_cheater_nn_model.keras",
    "v2_cheater_scaler.pkl",
    "v2_feature_medians.json",
    **stats
)

result = detector.predict()

from pprint import pprint
pprint(result)



