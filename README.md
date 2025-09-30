# ISAC Neural Net - Anti-Cheat

This repository contains a neural network project designed to detect cheaters in **The Division 2**.
The goal of releasing this project is to encourage Ubisoft to consider implementing robust AI-driven anti-cheat systems.

---

## 📂 Project Structure

```
models/                  # Pre-trained models and scalers
    vX_cheater_nn_model.keras   # Neural network models (v2 - v6)
    vX_cheater_scaler.pkl       # Feature scalers
    vX_feature_*.json           # Feature statistics (medians, stds, mads)
    cheater_model.pkl           # Legacy baseline model
    scaler.pkl                  # Legacy scaler

train/                   # Training scripts
    vX_neural_net_train.py      # Training programs for different versions
    neural_net_train.py         # General training entry script
    isac-ml-training-dataset.csv # Training dataset
    calc.py                      # Utility functions
    run.py / train.py            # Training runners

cheater_detectors/       # Cheater detection scripts
    vX_cheater_detector.py       # Detection code for each version
```

* **vX\_** → represents model versions (`v2` - `v6`).
  Each iteration contains improvements, with **`v6` being production-ready and recommended**.
* **models/** → already contains pre-trained data, no need to retrain unless experimenting.
* **train/** → scripts for training your own models if you want to improve or experiment.
* **cheater\_detectors/** → run-time cheater detection scripts using the trained models.

---

## ⚙️ Prerequisites

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Ensure Python 3.8+ is installed.

---

## 🚀 Usage

### Running Detection

Pick the latest detector (recommended `v6`):

```bash
python v6_cheater_detector.py
```

This will load the pre-trained **v6 model** and evaluate input data.

---

### Training (Optional)

If you want to retrain or experiment with your own dataset:

```bash
python v6_neural_net_train.py
```

This uses `isac-ml-training-dataset.csv` and will regenerate the model + scaler inside the `models/` directory.

---

## 🔬 Model Versions

* **v1**: Baseline cheater model (pkl-based, simple)
* **v2 - v3**: Early neural network attempts with basic feature scaling
* **v4**: Improved feature engineering (added medians, mads, stds)
* **v5**: More stable training pipeline, better accuracy
* **v6**: **Production-ready**, best performing, and recommended

---

## 📌 Notes

* The models are provided for **research and demonstration purposes only**.
* The **v6 model** is already pre-trained and can be used out-of-the-box.
* This project is not affiliated with or endorsed by Ubisoft.

---

## 🛡️ Disclaimer

This project is intended for **educational and research purposes only**.
The Division 2 is a trademark of Ubisoft. Please respect the terms of service of any game you use this with.


# 📘 ISAC Calculator Documentation

The **ISAC Calculator** provides utility methods for computing gameplay-derived statistics on a **per-hour basis**, normalizing values to enable cheat detection in *The Division 2*.

This class powers the **ISAC Anti-Cheat Neural Network**, ensuring raw stats are transformed into meaningful, scaled features.

---

## 🔧 Class: `IsacCalculator`

```python
from calc import IsacCalculator
```

### Purpose

The `IsacCalculator` standardizes gameplay statistics such as hits, headshots, bodyshots, critical hits, and NPC kills into **scaled per-hour metrics**. These metrics are used both for reporting and as features for the AI-based anti-cheat model.

---

## ⚙️ Initialization

```python
calc = IsacCalculator(
    time_played_total=8940,
    headshots=2565068,
    sum_hits=37037181,
    sum_critical_hits=500000,
    kills_npc=454610,
    kills_headshot=769521
)
```

### Parameters

* `time_played_total (int)` → Total time played in hours (or converted via `time_unit_convert_to_hours`).
* `headshots (int)` → Total number of headshots.
* `sum_hits (int)` → Total weapon hits.
* `sum_critical_hits (int)` → Total critical hits.
* `kills_npc (int)` → NPC kills.
* `kills_headshot (int)` → Headshot kills.

---

## ⏱️ Time Handling

### `time_unit_convert_to_hours(input_value: str | int) -> int`

Converts formatted strings like `"2h30m15s"` into integer hours.

```python
calc.time_unit_convert_to_hours("2h30m")  # returns 3
```

---

## 📊 Core Per-Hour Metrics

All methods below scale values using a **time-based divisor**. This ensures that abnormal stats (e.g., excessive headshots per hour) are properly weighted.

* `calculate_scaled_hits_per_hour()` → Scaled weapon hits/hour
* `calculate_scaled_critical_hits_per_hour()` → Scaled critical hits/hour
* `calculate_scaled_headshots_per_hour()` → Scaled headshots/hour
* `calculate_scaled_bodyshots_per_hour()` → Scaled bodyshots/hour
* `calculate_scaled_npc_kills_per_hour()` → Scaled NPC kills/hour
* `calculate_scaled_headshot_kills_per_hour()` → Scaled headshot kills/hour

Example:

```python
calc.calculate_scaled_headshots_per_hour()
```

---

## ✅ Validated Metrics

These return **0** if the raw stats are not valid for scaling (e.g., too few hours played).

* `calculate_bodyshots_per_hour()`
* `calculate_headshots_per_hour()`
* `calculate_npc_kills_per_hour()`
* `calculate_headshot_kills_per_hour()`
* `calculate_weapon_hits_per_hour()`
* `calculate_critical_hits_per_hour()`

---

## 🎯 Ratios and Percentages

### `calculate_percentages_and_ratios() -> dict`

Computes headshot/bodyshot ratios and percentages.

Returns:

```json
{
  "percentageOfHeadshots": 12.34,
  "percentageOfBodyshots": 87.66,
  "totalHeadshots": 2565068,
  "totalBodyshots": 34472113,
  "hsToBsRatio": 0.07,
  "bsToHsRatio": 13.44
}
```

---

## 🧮 Headshot Kill Estimation

### `calculate_total_headshot_kills(total_kills, total_headshots, avg_bullets_to_kill=50, avg_crit_chance=30) -> int`

Estimates total headshot kills based on kill count, accuracy, and crit chance.

Example:

```python
calc.calculate_total_headshot_kills(
    total_kills=10000,
    total_headshots=3000,
    avg_bullets_to_kill=45,
    avg_crit_chance=25
)
# → Estimated number of headshot kills
```

---

## 🔑 Usage in Anti-Cheat

The `IsacCalculator` is a **feature engineering utility** used by the **ISAC Anti-Cheat model** to:

* Normalize gameplay stats by playtime.
* Detect extreme ratios (e.g., headshot/bodyshot imbalance).
* Generate input features for the ML model (`weaponHitsPerHour`, `headshotsPerHour`, etc.).

Example integration:

```python
calc = IsacCalculator(
    time_played_total=8940,
    headshots=2565068,
    sum_hits=37037181,
    kills_npc=454610,
    kills_headshot=769521
)

features = calc.calculate_percentages_and_ratios()
print(features)
```

---

## 📌 Notes

* All scaling is divisor-based to prevent anomalies in short play sessions.
* Extreme values are handled gracefully (floored division, capped deviations).
* Outputs from this class feed directly into the **v6 production model**.
