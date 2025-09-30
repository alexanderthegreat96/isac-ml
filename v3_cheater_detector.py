import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import json

import math
import re
import math
from typing import Union, Dict, Any
from collections import namedtuple


class IsacCalculator:
    """
    IsacCalculator provides various scaled per-hour computations based on gameplay stats such as hits,
    headshots, critical hits, and kills. It also includes utility functions for time normalization
    and converting string-based time units to total hours.
    """

    Interval = namedtuple("Interval", ["min", "max", "multiplier"])
    DIVISOR_INTERVALS = [
        Interval(0, 100, 3.0),
        Interval(100, 200, 2.5),
        Interval(200, 300, 2.0),
        Interval(300, 400, 1.5),
        Interval(400, 600, 1.2),
        Interval(600, 800, 1.1),
    ]

    def __init__(
        self,
        time_played_total: int = 0,
        headshots: int = 0,
        sum_hits: int = 0,
        sum_critical_hits: int = 0,
        kills_npc: int = 0,
        kills_headshot: int = 0
    ):
        """Initialize the calculator with gameplay statistics."""
        self.time_played_total = time_played_total or 0
        self.headshots = headshots
        self.sum_hits = sum_hits
        self.sum_critical_hits = sum_critical_hits
        self.kills_npc = kills_npc
        self.kills_headshot = kills_headshot
        self.bodyshots = max(self.sum_hits - self.headshots, 0)

    def _get_divisor(self) -> float:
        """Determine scaling divisor based on total play time and pre-defined intervals."""
        for interval in self.DIVISOR_INTERVALS:
            if interval.min <= self.time_played_total < interval.max:
                return self.time_played_total * interval.multiplier
        return self.time_played_total

    def _safe_divide(self, numerator: int) -> int:
        """Safely divide the given value by the calculated divisor."""
        divisor = self._get_divisor()
        return math.floor(numerator / divisor) if divisor > 0 else 0

    def calculate_scaled_hits_per_hour(self) -> int:
        """Calculate weapon hits per hour, scaled by play time."""
        return self._safe_divide(self.sum_hits)

    def calculate_scaled_critical_hits_per_hour(self) -> int:
        """Calculate critical hits per hour, scaled by play time."""
        return self._safe_divide(self.sum_critical_hits)

    def calculate_scaled_headshots_per_hour(self) -> int:
        """Calculate headshots per hour, scaled by play time."""
        return self._safe_divide(self.headshots)

    def calculate_scaled_bodyshots_per_hour(self) -> int:
        """Calculate bodyshots per hour, scaled by play time."""
        return self._safe_divide(self.bodyshots)

    def calculate_scaled_npc_kills_per_hour(self) -> int:
        """Calculate NPC kills per hour, scaled by play time."""
        return self._safe_divide(self.kills_npc)

    def calculate_scaled_headshot_kills_per_hour(self) -> int:
        """Calculate headshot kills per hour, scaled by play time."""
        return self._safe_divide(self.kills_headshot)

    def time_unit_convert_to_hours(self, input_value: Union[str, None]) -> int:
        """
        Convert a time string (e.g. "2h30m") to total hours as an integer.
        Accepts strings with hours (h), minutes (m), and seconds (s).
        """
        if not input_value:
            return 0

        if isinstance(input_value, int):
            return input_value

        input_value = re.sub(r"[^a-zA-Z0-9]", "", input_value)

        hours = int(re.search(r"(\d+)h", input_value).group(1)) if re.search(r"(\d+)h", input_value) else 0
        minutes = int(re.search(r"(\d+)m", input_value).group(1)) if re.search(r"(\d+)m", input_value) else 0
        seconds = int(re.search(r"(\d+)s", input_value).group(1)) if re.search(r"(\d+)s", input_value) else 0

        total_hours = hours + (minutes / 60) + (seconds / 3600)
        return round(total_hours)

    def calculate_bodyshots_per_hour(self) -> int:
        """Return bodyshots per hour if valid for scaling."""
        return self.calculate_scaled_bodyshots_per_hour() if self.bodyshots > self.time_played_total else 0

    def calculate_headshots_per_hour(self) -> int:
        """Return headshots per hour if valid for scaling."""
        return self.calculate_scaled_headshots_per_hour() if self.time_played_total > 0 and self.headshots > 0 else 0

    def calculate_npc_kills_per_hour(self) -> int:
        """Return NPC kills per hour if valid for scaling."""
        return self.calculate_scaled_npc_kills_per_hour() if self.time_played_total > 0 and self.kills_npc > 0 else 0

    def calculate_headshot_kills_per_hour(self) -> int:
        """Return headshot kills per hour if valid for scaling."""
        return self.calculate_scaled_headshot_kills_per_hour() if self.time_played_total > 0 and self.kills_headshot > 0 else 0

    def calculate_weapon_hits_per_hour(self) -> int:
        """Return weapon hits per hour if valid for scaling."""
        if self.time_played_total > 0 and self.sum_hits > self.time_played_total:
            return self.calculate_scaled_hits_per_hour()
        return 0

    def calculate_critical_hits_per_hour(self) -> int:
        """Return critical hits per hour if valid for scaling."""
        if self.time_played_total > 0 and self.sum_critical_hits > self.time_played_total:
            return self.calculate_scaled_critical_hits_per_hour()
        return 0

    def calculate_total_headshot_kills(self, total_kills=0, total_headshots=0, avg_bullets_to_kill=50, avg_crit_chance=30) -> int:
        """
        Estimate total headshot kills based on average bullets to kill and critical hit chance.

        Formula:
            H = (T / A) * P * (headshots / T) * A

            Where:
                H = Estimated number of headshot kills
                B = Number of bodyshot kills
                C = Number of critical hits
                T = Total kills
                A = Average bullets to kill
                P = Crit chance (as a decimal)

        Parameters:
            total_kills (int): Total kills made.
            total_headshots (int): Number of headshots.
            avg_bullets_to_kill (int): Average bullets needed to kill.
            avg_crit_chance (int): Average critical hit chance in percent.

        Returns:
            int: Estimated headshot kills.
            Total kills made.
            total_headshots (int): Number of headshots.
            avg_bullets_to_kill (int): Average bullets needed to kill.
            avg_crit_chance (int): Average critical hit chance in percent.

        Returns:
            int: Estimated headshot kills.
        """
        if total_kills and total_headshots:
            return math.ceil(
                (total_kills / avg_bullets_to_kill)
                * (avg_crit_chance / 100)
                * (total_headshots / total_kills)
                * avg_bullets_to_kill
            )
        return 0


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
            "suspiciousHeadshotRate", "isExtremePlaytime"
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
            sum_critical_hits=0,
            kills_npc=s["npcKills"],
            kills_headshot=s["headshotKills"]
        )

        # Derived features
        s["weaponHitsPerHour"] = calc.calculate_weapon_hits_per_hour()
        s["headshotsPerHour"] = calc.calculate_headshots_per_hour()
        s["bodyshotsPerHour"] = calc.calculate_bodyshots_per_hour()
        s["headshotKillsPerHour"] = calc.calculate_headshot_kills_per_hour()
        s["npcKillsPerHour"] = calc.calculate_npc_kills_per_hour()

        s["suspiciousHeadshotRate"] = int(s["headshotsPerHour"] > 750)
        s["headshotRatio"] = s["headshots"] / (s["weaponHits"] + 1)
        s["headshotKillRatio"] = s["headshotKills"] / (s["npcKills"] + 1)
        s["hsOverBs"] = s["headshots"] / (s["bodyshots"] + 1)
        s["isExtremePlaytime"] = int(s["timePlayed"] > 5000)

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
            for col in self.feature_columns if col != "isExtremePlaytime"
        }

        return {
            "is_cheater": bool(pred),
            "cheater_probability": round(float(prob), 4),
            "confidence_percentage": f"{prob * 100:.2f}%",
            "computed": deviation_report
        }

stats = {
    "timePlayed": 506,
    "weaponHits": 3483964,
    "headshots": 1847820,
    "bodyshots": 1636144,
    "headshotKills": 554346,
    "npcKills": 123896
}

detector = IsacAntiCheat(
    "models/v4_cheater_nn_model.keras",
    "models/v4_cheater_scaler.pkl",
    "models/v4_feature_medians.json",
    "models/v4_feature_stds.json",
    "models/v4_feature_mads.json",
    **stats
)

print(detector.predict())