import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import json

import math
from typing import Dict, Any
from train.calc import IsacCalculator

class IsacAntiCheat:
    """
    Detects potential cheaters based on gameplay statistics using a trained neural network model.
    Features are standardized, and both standard deviation and MAD-based z-scores are calculated
    for flagging statistical anomalies.
    """

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        median_path: str,
        std_path: str,
        mad_path: str,
        **stats: Dict[str, Any]
    ):
        """Initialize the detector with model paths and gameplay statistics."""
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.threshold = 0.45  # Classification threshold
        self.stats = stats

        # Load feature statistics
        self.feature_medians = pd.Series(json.load(open(median_path)))
        self.feature_stds = pd.Series(json.load(open(std_path)))
        self.feature_mads = pd.Series(json.load(open(mad_path)))

        # List of features used by the model
        self.feature_columns = [
            "headshotsPerHour", "bodyshotsPerHour", "weaponHitsPerHour",
            "headshotKillsPerHour", "npcKillsPerHour",
            "headshotRatio", "headshotKillRatio", "hsOverBs",
            "suspiciousHeadshotRate",
            "hsToBsAbove129", "moreHeadshotsThanBodyshots", "hsRatioAboveBsRatio",
            "hsToBsRatio", "bsToHsRatio",
            "percentageOfHeadshots", "percentageOfBodyshots"
        ]

    def preprocess(self) -> pd.DataFrame:
        """
        Compute derived features using the provided stats and return a single-row DataFrame.
        """
        s = self.stats.copy()

        calc = IsacCalculator(
            time_played_total=s["timePlayed"],
            headshots=s["headshots"],
            sum_hits=s["weaponHits"],
            kills_npc=s["npcKills"],
            kills_headshot=s["headshotKills"]
        )

        ratio_data = calc.calculate_percentages_and_ratios() or {}

        # Ensure bodyshots is derived
        s["bodyshots"] = s["weaponHits"] - s["headshots"]

        # Derived per-hour features
        s["weaponHitsPerHour"] = calc.calculate_weapon_hits_per_hour()
        s["headshotsPerHour"] = calc.calculate_headshots_per_hour()
        s["bodyshotsPerHour"] = calc.calculate_bodyshots_per_hour()
        s["headshotKillsPerHour"] = calc.calculate_headshot_kills_per_hour()
        s["npcKillsPerHour"] = calc.calculate_npc_kills_per_hour()

        # Derived ratio features
        s["headshotRatio"] = s["headshots"] / (s["weaponHits"] + 1)
        s["headshotKillRatio"] = s["headshotKills"] / (s["npcKills"] + 1)
        s["hsToBsRatio"] = ratio_data.get("hsToBsRatio", 0)
        s["bsToHsRatio"] = ratio_data.get("bsToHsRatio", 0)
        s["hsOverBs"] = ratio_data.get("hsOverBs", 0)
        s["percentageOfHeadshots"] = ratio_data.get("percentageOfHeadshots", 0)
        s["percentageOfBodyshots"] = ratio_data.get("percentageOfBodyshots", 0)

        # Flags
        s["suspiciousHeadshotRate"] = int(s["headshotsPerHour"] > 750)
        s["hsToBsAbove129"] = int(s["hsToBsRatio"] > 1.29)
        s["moreHeadshotsThanBodyshots"] = int(s["headshots"] > s["bodyshots"])
        s["hsRatioAboveBsRatio"] = int(s["headshotRatio"] > (s["bodyshots"] / (s["weaponHits"] + 1)))

        return pd.DataFrame([s])[self.feature_columns]

    def interpret_z(self, z: float, percent_deviation: float) -> str:
        """Return a label based on z-score and percentage deviation."""
        abs_z = abs(z)
        abs_pd = abs(percent_deviation)

        if abs_z >= 3 or abs_pd >= 300:
            return "Extreme"
        elif abs_z >= 2 or abs_pd >= 200:
            return "Strong"
        elif abs_z >= 1 or abs_pd >= 100:
            return "Mild"
        return "Normal"

    def predict(self) -> Dict[str, Any]:
        """
        Predict whether the player is a cheater, and generate a full deviation report.

        Returns:
            Dict[str, Any]: Classification result and feature deviation analysis.
        """
        X = self.preprocess()
        X_scaled = self.scaler.transform(X)
        X_scaled = np.array(X_scaled, dtype=np.float32)

        prob = self.model.predict(X_scaled, verbose=0).flatten()[0]
        pred = int(prob > self.threshold)

        row = X.iloc[0]
        deviation = row - self.feature_medians

        z_scores = (row - self.feature_medians) / (self.feature_stds + 1e-6)
        robust_z_scores = (row - self.feature_medians) / (self.feature_mads.replace(0, 1e-6))

        with np.errstate(divide='ignore', invalid='ignore'):
            deviation_percent = ((row / self.feature_medians) - 1) * 100
            deviation_percent = deviation_percent.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-500, 500).round(2)

        deviation_report = {
            col: {
                "value": float(row[col]),
                "median": float(self.feature_medians[col]),
                "deviation": round(float(deviation[col]), 4),
                "percent_deviation": round(float(deviation_percent[col]), 2),
                "std_z_score": round(float(z_scores[col]), 2),
                "z_score": round(float(robust_z_scores[col]), 2),  # MAD-based
                "flag": self.interpret_z(robust_z_scores[col], deviation_percent[col])
            }
            for col in self.feature_columns
        }

        return {
            "is_cheater": bool(pred),
            "cheater_probability": round(float(prob), 4),
            "confidence_percentage": f"{prob * 100:.2f}%",
            "computed": deviation_report
        }
