# isac_model_training.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
from calc import IsacCalculator

# Ensure models/ directory exists
os.makedirs("models", exist_ok=True)

# 1. Load dataset
df = pd.read_csv("data/isac-ml-training-dataset-v2.csv")

# 2. Compute scaled features per row using IsacCalculator
def compute_row_metrics(row):
    calc = IsacCalculator(
        time_played_total=int(row["timePlayed"]),
        headshots=int(row["headshots"]),
        sum_hits=int(row["weaponHits"]),
        kills_npc=int(row["npcKills"]),
        kills_headshot=int(row["headshotKills"])
    )

    ratio_data = calc.calculate_percentages_and_ratios() or {}

    return pd.Series({
        # Per-hour features
        "weaponHitsPerHour": calc.calculate_weapon_hits_per_hour(),
        "headshotsPerHour": calc.calculate_headshots_per_hour(),
        "bodyshotsPerHour": calc.calculate_bodyshots_per_hour(),
        "headshotKillsPerHour": calc.calculate_headshot_kills_per_hour(),
        "npcKillsPerHour": calc.calculate_npc_kills_per_hour(),
        # Ratio-based indicators
        "percentageOfHeadshots": ratio_data.get("percentageOfHeadshots", np.nan),
        "percentageOfBodyshots": ratio_data.get("percentageOfBodyshots", np.nan),
        "hsToBsRatio": ratio_data.get("hsToBsRatio", np.nan),
        "bsToHsRatio": ratio_data.get("bsToHsRatio", np.nan),
        # Flags
        "hsToBsAbove129": int(ratio_data.get("hsToBsRatio", 0) > 1.29),
        "moreHeadshotsThanBodyshots": int(ratio_data.get("totalHeadshots", 0) > ratio_data.get("totalBodyshots", 0)),
        "hsRatioAboveBs": int(ratio_data.get("percentageOfHeadshots", 0) > ratio_data.get("percentageOfBodyshots", 0)),
    })

# Apply metrics
df = pd.concat([df, df.apply(compute_row_metrics, axis=1)], axis=1)

# 3. Add derived features
df["suspiciousHeadshotRate"] = (df["headshotsPerHour"] > 750).astype(int)
df["headshotRatio"] = df["headshots"] / (df["weaponHits"] + 1)
df["headshotKillRatio"] = df["headshotKills"] / (df["npcKills"] + 1)
df["hsOverBs"] = df["headshots"] / (df["bodyshots"] + 1)
df["hsRatioAboveBsRatio"] = (df["headshotRatio"] > (df["bodyshots"] / (df["weaponHits"] + 1))).astype(int)

# 4. Replace any NaN or inf values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# 5. Define features and labels
feature_columns = [
    "headshotsPerHour", "bodyshotsPerHour", "weaponHitsPerHour",
    "headshotKillsPerHour", "npcKillsPerHour",
    "headshotRatio", "headshotKillRatio", "hsOverBs",
    "suspiciousHeadshotRate",
    "hsToBsAbove129", "moreHeadshotsThanBodyshots", "hsRatioAboveBsRatio",
    "hsToBsRatio", "percentageOfHeadshots"
]

X = df[feature_columns]
y = df["isCheater"]

# 6. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Scale inputs
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Build model
model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# 9. Compile
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 10. Train with class weighting
history = model.fit(
    X_train_scaled, y_train,
    epochs=15,
    batch_size=256,
    validation_split=0.1,
    class_weight={0: 1, 1: 5},
    verbose=1
)

# 11. Evaluate
y_proba = model.predict(X_test_scaled).flatten()
y_pred = (y_proba > 0.45).astype(int)
print(classification_report(y_test, y_pred))

# 12. Plot loss curves
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Binary Crossentropy Loss')
plt.legend()
plt.grid(True)
plt.show()

# 13. Save feature medians
feature_medians = pd.Series(X.median(numeric_only=True))
feature_medians.to_json("models/v5_feature_medians.json")

# 14. Save feature std deviations
feature_stds = pd.Series(X.std(numeric_only=True))
feature_stds.to_json("models/v5_feature_stds.json")

# 15. Save MAD (median absolute deviation)
feature_mads = X.apply(lambda col: np.median(np.abs(col - np.median(col))))
feature_mads.to_json("models/v5_feature_mads.json")

# 16. Save model and scaler
model.save("models/v5_cheater_nn_model.keras")
joblib.dump(scaler, "models/v5_cheater_scaler.pkl")

