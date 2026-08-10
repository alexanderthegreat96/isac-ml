# ------------------------------------------------------------------------------
# ISAC Anti-Cheat - inference, v8
# ------------------------------------------------------------------------------

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
import pprint

try:
    from libs.Logger import Logger
    _log = Logger("ISAC AntiCheat v8").get_logger()
except ImportError:
    import logging
    _log = logging.getLogger("ISAC AntiCheat v8")

# The calculator lives in different places depending on how this is deployed
# (package, flat directory, notebook). Try each rather than hard-failing.
try:
    from train.calc import IsacCalculator
except ImportError:
    try:
        from calc import IsacCalculator
    except ImportError:
        from .calc import IsacCalculator


class IsacAntiCheat:
    """Two-stage cheat detection. Stage 1 is a rule; stage 2 ranks for review."""

    def __init__(self, model_dir: str, manifest: str = "v8_manifest.json",
                 **stats: Any):
        # v6 signature was IsacAntiCheat(model, scaler, medians, stds, mads,
        # **stats) and then predict() with no arguments. Both calling styles
        # work here: pass stats to the constructor and call predict(), or pass
        # them to predict() directly.
        self.stats_input = stats or None
        self.dir = Path(model_dir)
        with open(self.dir / manifest) as f:
            self.manifest = json.load(f)

        self.ban_threshold   = self.manifest.get("auto_ban_threshold",
                                                  self.manifest["ban_threshold"])
        self.band_floor      = self.manifest.get("review_band_floor",
                                                 self.manifest.get("queue_floor", 700.0))
        self.queue_floor     = self.manifest.get("queue_floor", 700.0)
        self.inflation       = self.manifest.get("inflation_factor", 1.0)
        self.divisor_mode    = self.manifest.get("divisor_mode", "tiered")

        # Stage 2 is a single model since daily routing was removed. The loader
        # still iterates stage2_models so older two-model manifests keep loading.
        self.models, self.stats, self.features, self.thresholds = {}, {}, {}, {}
        self.missing_stats = []
        for m in self.manifest["stage2_models"]:
            name = m["name"]

            # Older train_v8.py builds wrote neither model_file nor stats_file
            # into the manifest. Fall back to the naming convention the trainer
            # has always used, so old and new manifests both load.
            version = self.manifest.get("version", "v8")
            model_p = self.dir / m.get("model_file", f"{version}_stage2_{name}.pkl")
            stats_p = self.dir / m.get("stats_file", f"{version}_stats_{name}.json")

            if not model_p.exists():
                raise FileNotFoundError(
                    f"stage-2 model for population '{name}' not found at {model_p}. "
                    f"Files present: {sorted(q.name for q in self.dir.iterdir())}"
                )
            expected = m.get("model_sha256")
            if expected:
                actual = hashlib.sha256(model_p.read_bytes()).hexdigest()
                if actual != expected:
                    raise IOError(
                        f"{model_p.name} is corrupted: sha256 {actual[:12]}... does not "
                        f"match the manifest's {expected[:12]}... . The file was altered "
                        f"after training - typically a text-mode or truncated download. "
                        f"Re-download as binary, or retrain on this machine."
                    )
            self.models[name] = joblib.load(model_p)
            self.features[name] = m["features"]
            self.thresholds[name] = m["threshold"]

            # Stats drive the deviation report only. Verdicts do not need them,
            # so a missing file degrades the response rather than killing it.
            if stats_p.exists():
                with open(stats_p) as f:
                    self.stats[name] = json.load(f)
            else:
                self.stats[name] = None
                self.missing_stats.append(stats_p.name)

        if self.missing_stats:
            _log.warning("no stats files: %s. Verdicts still work; the 'computed' "
                         "deviation report will be omitted. Re-run training to "
                         "generate them.", self.missing_stats)

        # Whole-dataset calibrated P(isCheater), DISPLAY only (see trainer). It
        # feeds cheater_percentage and drives no verdict. Absent -> field is null.
        self.cheater_scorer = None
        self.cheater_scorer_feats = None
        cs = self.manifest.get("cheater_scorer")
        if cs:
            version = self.manifest.get("version", "v8")
            cs_p = self.dir / cs.get("model_file", f"{version}_cheater_scorer.pkl")
            if cs_p.exists():
                exp = cs.get("model_sha256")
                if exp and hashlib.sha256(cs_p.read_bytes()).hexdigest() != exp:
                    raise IOError(f"{cs_p.name} is corrupted: sha256 does not match "
                                  f"the manifest. Re-download as binary, or retrain.")
                self.cheater_scorer = joblib.load(cs_p)
                self.cheater_scorer_feats = cs["features"]
            else:
                _log.warning("cheater_scorer model missing at %s; cheater_percentage "
                             "will be null.", cs_p)

    # -- feature construction -------------------------------------------------
    def validate(self, s: Dict[str, Any]) -> Optional[str]:
        """Return a reason string if this input cannot be scored, else None."""
        required = ["timePlayed", "weaponHits", "criticalHits", "headshots", "npcKills", "headshotKills"]
        missing = [k for k in required if s.get(k) is None]
        if missing:
            return f"missing required fields: {missing}"
        if s["timePlayed"] <= 0:
            return "timePlayed is 0 - no rate can be computed"
        if s["weaponHits"] <= 0:
            return "weaponHits is 0 - no ratio can be computed"
        # headshots > weaponHits is handled in predict(), not rejected here:
        # the fault is weaponHits failing to populate (measured ~30x low, some
        # rows literally 1) while headshots stays plausible. Stage 1 only needs
        # headshots and timePlayed, so it still runs; stage 2 cannot.
        return None

    def build_features(self, s: Dict[str, Any]) -> pd.Series:
        s = dict(s)
        if self.inflation != 1.0:
            for k in ["weaponHits", "criticalHits", "headshots", "headshotKills", "npcKills"]:
                s[k] = s[k] / self.inflation

        calc = IsacCalculator(
            time_played_total=int(s["timePlayed"]),
            headshots=int(s["headshots"]),
            sum_hits=int(s["weaponHits"]),
            sum_critical_hits=int(s["criticalHits"]),
            kills_npc=int(s["npcKills"]),
            kills_headshot=int(s["headshotKills"]),
        )
        ratios = calc.calculate_percentages_and_ratios()
        if ratios is None:
            raise ValueError("calculator returned no ratios - inputs are degenerate")

        f = {
            "timePlayed":               float(s["timePlayed"]),
            "headshotsPerHour":         calc.calculate_headshots_per_hour(),
            "bodyshotsPerHour":         calc.calculate_bodyshots_per_hour(),
            "weaponHitsPerHour":        calc.calculate_weapon_hits_per_hour(),
            "npcKillsPerHour":          calc.calculate_npc_kills_per_hour(),
            "criticalHitsPerHour":      calc.calculate_critical_hits_per_hour(),
            "headshotKillsPerHour":     calc.calculate_headshot_kills_per_hour(),
            "percentageOfHeadshots":    ratios["percentageOfHeadshots"],
            "percentageOfCriticalHits": 100 * s["criticalHits"] / s["weaponHits"],
            "percentageOfBodyshots":    ratios["percentageOfBodyshots"],
            "hsToBsRatio":              ratios["hsToBsRatio"],
            "bsToHsRatio":              ratios["bsToHsRatio"],
            "totalHeadshots":           ratios["totalHeadshots"],
            "totalBodyshots":           ratios["totalBodyshots"],
            "totalCriticalHits":        float(s["criticalHits"]),
            "headshotRatio":            s["headshots"] / (s["weaponHits"] + 1),
            "headshotKillRatio":        s["headshotKills"] / (s["npcKills"] + 1),
        }
        return pd.Series(f)

    def _cheater_pct(self, f: pd.Series) -> Optional[float]:
        """Whole-dataset calibrated P(isCheater) as a percentage. DISPLAY only -
        no verdict depends on it. None when the scorer was not loaded."""
        if self.cheater_scorer is None:
            return None
        fs = self.cheater_scorer_feats
        X = pd.DataFrame([f[fs].values], columns=fs).astype(float)
        return round(100 * float(self.cheater_scorer.predict_proba(X)[0, 1]), 2)

    # -- reporting ------------------------------------------------------------
    @staticmethod
    def interpret(z: float, pct: float) -> str:
        az, ap = abs(z), abs(pct)
        if az >= 3 or ap >= 300:
            return "Extreme"
        if az >= 2 or ap >= 200:
            return "Strong"
        if az >= 1 or ap >= 100:
            return "Mild"
        return "Normal"

    def deviation_report(self, f: pd.Series, pop: str) -> Optional[Dict[str, Any]]:
        """How far this player sits from a normal (legit) player in their
        population. This is the appeal-facing artifact - it must be readable by
        someone who is not an ML engineer."""
        if self.stats.get(pop) is None:
            return None
        out = {}
        for col, st in self.stats[pop].items():
            val, med = float(f[col]), st["median"]
            mad, std = st["mad"] or 1e-6, st["std"] or 1e-6
            pct = 0.0 if med == 0 else max(min(((val-med)/med)*100, 500), -500)
            rz  = max(min((val-med)/mad, 100.0), -100.0)
            out[col] = {
                "input_value": round(val, 4),
                "median_value": round(med, 4),
                "deviation_value": round(val-med, 4),
                "deviation_percentage": round(pct, 2),
                "std_z_score": round((val-med)/std, 2),
                "z_score": round(rz, 2),
                "flag": "Binary" if st["binary"] else self.interpret(rz, pct),
                "p99": round(st["p99"], 4),   # new in v8, additive
            }
        return out

    # -- response assembly ----------------------------------------------------
    @staticmethod
    def _envelope(code: int, status: bool, data: Dict[str, Any]) -> Dict[str, Any]:
        """Wrap the payload in the API contract: {status, code, data}. status is
        whether the request could be SCORED, not whether the player is clean - a
        confirmed cheater is still 200/true; only an unscorable input is
        422/false."""
        return {"status": status, "code": code, "data": data}

    @staticmethod
    def _finalize(data: Dict[str, Any]) -> Dict[str, Any]:
        """If the deviation report is absent because the stats files were not
        shipped (rather than corrupt input), say so instead of a bare null."""
        if data.get("computed") is None and "computed_unavailable" not in data:
            data["computed_unavailable"] = ("stats files missing; re-run training "
                                            "to generate them")
        return data

    @staticmethod
    def _summary_fields(confidence: float, verdict: str, action: str,
                        cheater_pct: Optional[float]) -> Dict[str, Any]:
        """The summary block present on every response, whatever the path.

        Two independent numbers, deliberately named apart so neither is mistaken
        for the other:
          - confidence_score : the QUEUE model's confidence in THIS verdict
            (0..1). 1.0 on a rule flag; the stage-2 score on a review. A ranking
            score, NOT a probability of cheating.
          - cheater_probability / cheater_percentage : the calibrated
            whole-dataset P(cheater) - the actual 'how likely a cheater' number.
            Pinned to 100% on a rule flag; the scorer's estimate otherwise; None
            when it cannot be computed.

        should_flag / should_review are the booleans a consumer acts on (flag =
        automatic action now; review = put in front of a human; never both).
        classification_tier follows verdict + action, so it can never contradict
        those booleans."""
        if verdict == "RULE_FLAG":
            tier = "Confirmed cheater"
        elif verdict == "UNSCORABLE":
            tier = "Unscorable"
        elif action == "queue_for_review":
            tier = "Flagged for review"
        elif action == "monitor":
            tier = "Monitor"
        else:
            tier = "Cleared"

        return {
            "is_cheater": verdict == "RULE_FLAG",
            "should_flag": verdict == "RULE_FLAG",
            "should_review": action == "queue_for_review",
            "classification_tier": tier,
            "cheater_probability": (f"{cheater_pct:.2f}%" if cheater_pct is not None else None),
            "cheater_percentage": cheater_pct,
            "confidence_score": round(float(confidence), 4),
        }

    # -- verdict --------------------------------------------------------------
    def predict(self, stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        stats = stats if stats is not None else self.stats_input
        if stats is None:
            raise ValueError(
                "no stats provided. Either IsacAntiCheat(model_dir, **stats) "
                "then .predict(), or .predict(stats)."
            )
        bad = self.validate(stats)
        if bad:
            return self._envelope(422, False, {
                **self._summary_fields(0.0, "UNSCORABLE", "none", None),
                "verdict": "UNSCORABLE", "action": "none", "scorable": False,
                "reason": bad,
                "advisory": {"available": False, "reason": "input failed validation"},
                "computed": None,
            })

        # Partial-data path: weaponHits is corrupt but headshots is intact. Run
        # the rule on the salvageable field; never run the models, whose ratio
        # features (bsToHsRatio, percentageOfHeadshots) would read the clamped
        # bodyshots=0 as a pure-aimbot signature. The deviation report is omitted
        # here on purpose - every ratio it shows derives from the broken weaponHits.
        if stats["headshots"] > stats["weaponHits"]:
            calc = IsacCalculator(
                time_played_total=int(stats["timePlayed"]),
                headshots=int(stats["headshots"] / self.inflation),
                sum_hits=int(stats["headshots"] / self.inflation),  # neutralises the ratio guards
                kills_npc=int(stats["npcKills"] / self.inflation),
                kills_headshot=int(stats["headshotKills"] / self.inflation),
            )
            hsph = float(calc.calculate_headshots_per_hour())
            flagged = hsph > self.ban_threshold
            verdict = "RULE_FLAG" if flagged else "CLEAR"
            action  = "automatic" if flagged else "flag_data_fault"
            data = {
                **self._summary_fields(1.0 if flagged else 0.0, verdict, action,
                                       100.0 if flagged else None),
                "verdict": verdict, "action": action, "scorable": True, "stage": 1,
                "headshotsPerHour": hsph,
                "data_fault": ("weaponHits did not populate "
                               f"({stats['weaponHits']:,} recorded vs {stats['headshots']:,} "
                               f"headshots). Ratio features unusable; models skipped."),
                "advisory": {"available": False,
                             "reason": "stage-2 features unusable (weaponHits corrupt)"},
                "computed": None,
                "computed_unavailable": ("weaponHits corrupt; the deviation report "
                                         "needs valid ratios"),
            }
            if flagged:
                data["rule"] = f"headshotsPerHour > {self.ban_threshold}"
                data["margin_over_threshold"] = round(hsph - self.ban_threshold, 1)
                data["reason"] = (f"{hsph:.0f} headshots/hour against a threshold of "
                                  f"{self.ban_threshold:.0f}. weaponHits is corrupt but the "
                                  f"rule does not depend on it.")
            else:
                data["reason"] = (f"{hsph:.0f} headshots/hour, below the "
                                  f"{self.ban_threshold:.0f} threshold. Model review "
                                  f"unavailable until weaponHits is repaired at ingest.")
            return self._envelope(200, True, data)

        # Full-data path. Build features once and compute BOTH model scores and
        # the deviation report up front, so every branch - stage 1 or 2 - returns
        # the same rich payload and 'computed' is ALWAYS present.
        f = self.build_features(stats)
        hsph = float(f["headshotsPerHour"])
        scorer_pct = self._cheater_pct(f)                        # calibrated P(cheater)
        pop   = next(iter(self.models))                          # single stage-2 model
        feats = self.features[pop]
        X = pd.DataFrame([f[feats].values], columns=feats).astype(float)
        prob = float(self.models[pop].predict_proba(X)[0, 1])    # stage-2 queue score
        thr  = self.thresholds[pop]
        computed = self.deviation_report(f, pop)

        # STAGE 1 - deterministic. The rule alone decides; the model scores are
        # reported under 'advisory' for transparency but carry no weight, because
        # they are circular here (the rule generated their training labels).
        if hsph > self.ban_threshold:
            data = {
                **self._summary_fields(1.0, "RULE_FLAG", "automatic", 100.0),
                "verdict": "RULE_FLAG", "action": "automatic", "scorable": True,
                "stage": 1,
                "rule": f"headshotsPerHour > {self.ban_threshold}",
                "headshotsPerHour": hsph,
                "margin_over_threshold": round(hsph - self.ban_threshold, 1),
                "reason": (f"{hsph:.0f} headshots/hour against a threshold of "
                           f"{self.ban_threshold:.0f}."),
                "note": ("The rule alone decides this verdict. The model scores are "
                         "reported under advisory for transparency, but they are "
                         "circular here and carry no weight."),
                "advisory": {
                    "available": True, "advisory_only": True,
                    "model_population": pop,
                    "score": round(prob, 4),
                    "queue_threshold": round(thr, 4),
                    "would_queue_independently": bool(prob >= thr),
                    "cheater_percentage": scorer_pct,
                    "caveat": ("Circular: the stage-1 rule generated these models' "
                               "training labels, so agreement is not independent "
                               "confirmation. The verdict on this response is the "
                               "rule's, not these scores'."),
                },
                "computed": computed,
            }
            return self._envelope(200, True, self._finalize(data))

        # STAGE 2 - review priority only, never automatic. Here the queue model
        # IS the basis of the verdict, so its score is the top-level confidence.
        in_band = hsph > self.band_floor   # 700-1000: always reviewed, never auto
        if prob >= thr:
            priority, action = "HIGH", "queue_for_review"
        elif in_band:
            priority, action = "MEDIUM", "queue_for_review"
        elif prob >= thr * 0.5:
            priority, action = "LOW", "monitor"
        else:
            priority, action = "NONE", "none"
        verdict = "REVIEW" if action == "queue_for_review" else "CLEAR"

        data = {
            **self._summary_fields(prob, verdict, action, scorer_pct),
            "verdict": verdict, "action": action, "scorable": True, "stage": 2,
            "review_priority": priority,
            "model_population": pop,
            "score": round(prob, 4),
            "queue_threshold": round(thr, 4),
            "headshotsPerHour": hsph,
            "reason": ((f"In the {self.band_floor:.0f}-{self.ban_threshold:.0f} review band. "
                        if in_band else
                        f"Below the {self.ban_threshold:.0f} auto threshold. ")
                       + f"Stage-2 score {prob:.3f} against a queue cutoff of {thr:.3f}."),
            "caveat": ("Stage 2 is trained on ~291 cheaters confirmed below the "
                       "rule threshold. It ranks for human review; it does not "
                       "establish cheating."),
            "advisory": {"available": False,
                         "reason": ("the stage-2 model is the basis of this verdict; "
                                    "see confidence_score / score")},
            "computed": computed,
        }
        return self._envelope(200, True, self._finalize(data))


if __name__ == "__main__":
    detector = IsacAntiCheat("models")

    test_players = {
        "Renqh.": dict(
            timePlayed=803,
            weaponHits=3930785,
            criticalHits=1572000,
            headshots=750885,
            headshotKills=225266,
            npcKills=192361,
        ),
        "ToasteyXGod": dict(
            timePlayed=5124,
            weaponHits=69856037,
            criticalHits=27942000,
            headshots=3859603,
            headshotKills=1157881,
            npcKills=2746597,
        ),
        "New_Liberty": dict(
            timePlayed=3036,
            weaponHits=198734879,
            criticalHits=79493000,
            headshots=11134421,
            headshotKills=3340327,
            npcKills=1478289,
        ),
    }

    for name, stats in test_players.items():
        result = detector.predict(stats)
        print(f"\n=== {name} ===")
        pprint.pprint(result)
        # from pprint import pprint; pprint(result)   # full payload incl. deviation report