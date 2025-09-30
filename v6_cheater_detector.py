import pandas as pd
import numpy as np
import joblib
import json
from tensorflow.keras.models import load_model
from typing import Dict, Any
from train.calc import IsacCalculator


class IsacAntiCheat:
    """
    Detects potential cheaters based on gameplay statistics using a trained neural network model.
    Derived features are standardized, and z-score/MAD-based outlier detection is applied.
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
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.threshold = 0.55
        self.stats = stats

        # Load feature stats
        self.feature_medians = pd.Series(json.load(open(median_path)))
        self.feature_stds = pd.Series(json.load(open(std_path)))
        self.feature_mads = pd.Series(json.load(open(mad_path)))

        self.feature_columns = [
            "weaponHitsPerHour", "headshotsPerHour", "bodyshotsPerHour",
            "headshotKillsPerHour", "npcKillsPerHour",
            "percentageOfHeadshots", "percentageOfBodyshots",
            "hsToBsRatio", "bsToHsRatio",
            "headshotRatio", "headshotKillRatio",
            "headshotsPerHourCritical", "criticalBsToHsRatio", "criticalHsToBsRatio",
            "moreHeadshotsThanBodyshots", "hsRatioAboveBsRatio", "cheaterRuleMatch"
        ]

    def preprocess(self) -> pd.DataFrame:
        s = self.stats.copy()

        calc = IsacCalculator(
            time_played_total=s["timePlayed"],
            headshots=s["headshots"],
            sum_hits=s["weaponHits"],
            kills_npc=s["npcKills"],
            kills_headshot=s["headshotKills"]
        )

        ratio_data = calc.calculate_percentages_and_ratios() or {}
        total_bodyshots = ratio_data.get("totalBodyshots", 0)
        total_headshots = ratio_data.get("totalHeadshots", 0)
        hs_ratio = ratio_data.get("percentageOfHeadshots", 0)
        bs_ratio = ratio_data.get("percentageOfBodyshots", 0)
        hs_to_bs_ratio = ratio_data.get("hsToBsRatio", 0)
        bs_to_hs_ratio = ratio_data.get("bsToHsRatio", 0)

        s["bodyshots"] = s["weaponHits"] - s["headshots"]
        s["weaponHitsPerHour"] = calc.calculate_weapon_hits_per_hour()
        s["headshotsPerHour"] = calc.calculate_headshots_per_hour()
        s["bodyshotsPerHour"] = calc.calculate_bodyshots_per_hour()
        s["headshotKillsPerHour"] = calc.calculate_headshot_kills_per_hour()
        s["npcKillsPerHour"] = calc.calculate_npc_kills_per_hour()
        s["percentageOfHeadshots"] = hs_ratio
        s["percentageOfBodyshots"] = bs_ratio
        s["hsToBsRatio"] = hs_to_bs_ratio
        s["bsToHsRatio"] = bs_to_hs_ratio
        s["headshotRatio"] = s["headshots"] / (s["weaponHits"] + 1)
        s["headshotKillRatio"] = s["headshotKills"] / (s["npcKills"] + 1)
        s["headshotsPerHourCritical"] = int(s["headshotsPerHour"] > 803)
        s["criticalBsToHsRatio"] = int(bs_to_hs_ratio < 1.29)
        s["criticalHsToBsRatio"] = int(hs_to_bs_ratio > 1.29)
        s["moreHeadshotsThanBodyshots"] = int(total_headshots > total_bodyshots)
        s["hsRatioAboveBsRatio"] = int(s["headshotRatio"] > (s["bodyshots"] / (s["weaponHits"] + 1)))


        s["cheaterRuleMatch"] = int(
            s["headshotsPerHour"] > 803 and
            s["bsToHsRatio"] < 1.29 and
            s["percentageOfHeadshots"] > s["percentageOfBodyshots"]
        )
        df = pd.DataFrame([s])[self.feature_columns]
        print("[DEBUG] Preprocessed features:\n", df.to_string(index=False))
        return df

    def interpret_z(self, z: float, percent_deviation: float) -> str:
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
          X = self.preprocess()
          X_scaled = self.scaler.transform(X).astype(np.float32)

          prob = self.model.predict(X_scaled, verbose=0).flatten()[0]
          pred = int(prob > self.threshold)

          # Tiered classification
          if prob >= 0.90:
              tier = "Confirmed cheater"
          elif prob >= 0.50:
              tier = "Likely cheater"
          elif prob >= 0.25:
              tier = "REgular"
          else:
              tier = "Likely clean"

          row = X.iloc[0]
          deviation = row - self.feature_medians
          safe_stds = self.feature_stds.replace(0, 1e-6)
          safe_mads = self.feature_mads.replace(0, 1e-6)
          z_scores = (row - self.feature_medians) / safe_stds
          robust_z_scores = (row - self.feature_medians) / safe_mads

          with np.errstate(divide='ignore', invalid='ignore'):
              deviation_percent = np.where(
                  self.feature_medians != 0,
                  ((row - self.feature_medians) / self.feature_medians) * 100,
                  0.0
              )
              deviation_percent = pd.Series(deviation_percent, index=row.index).clip(-500, 500).round(2)

          deviation_report = {}
          for col in self.feature_columns:
              val = float(row[col])
              med = float(self.feature_medians[col])
              dev = round(float(deviation[col]), 4)
              pct_dev = round(float(deviation_percent[col]), 2)
              std_z = round(float(z_scores[col]), 2)
              robust_z = round(float(robust_z_scores[col]), 2)

              if abs(robust_z) > 100:
                  robust_z = 100.0 if robust_z > 0 else -100.0

              if val in [0.0, 1.0] and med in [0.0, 1.0]:
                  flag = "Binary"
              else:
                  flag = self.interpret_z(robust_z, pct_dev)

              deviation_report[col] = {
                  "value": val,
                  "median": med,
                  "deviation": dev,
                  "percent_deviation": pct_dev,
                  "std_z_score": std_z,
                  "z_score": robust_z,
                  "flag": flag
              }

          print(f"[DEBUG] Probability: {prob:.4f} | Tier: {tier} | Cheater: {bool(pred)}")

          return {
              "is_cheater": bool(pred),
              "cheater_probability": round(float(prob), 4),
              "confidence_percentage": f"{prob * 100:.2f}%",
              "classification_tier": tier,
              "computed": deviation_report
          }


stats = {
    "timePlayed": 8940,
    "weaponHits": 37037181,
    "headshots": 2565068,
    "bodyshots": 34472113,
    "headshotKills": 769521,
    "npcKills": 454610
}

detector = IsacAntiCheat(
    "models/v6_cheater_nn_model.keras",
    "models/v6_cheater_scaler.pkl",
    "models/v6_feature_medians.json",
    "models/v6_feature_stds.json",
    "models/v6_feature_mads.json",
    **stats
)

print(detector.predict())
