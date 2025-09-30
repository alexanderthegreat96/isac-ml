# ------------------------------------------------------------------------------
# ML Model Goal:
# Improve current cheat detection methods using a neural network.
#
# Training dataset columns:
# - timePlayed, weaponHits, headshots, bodyshots, headshotKills, npcKills
# - headshotsPerHour, bodyshotsPerHour, weaponHitsPerHour
# - isCheater (label)
#
# Key detection signals:
# - Headshots/hour > 790
# - Bodyshot-to-headshot ratio < 1.29 (1.29 is lowest legit ratio)
# - More headshots than bodyshots
# - Headshot percentage > bodyshot percentage
# - NPC kills, headshot kills
# - Relative value of NPC kills vs. headshot kills
#
# The model should learn to detect abnormal player behavior by analyzing
# how cheaters (isCheater = 1) deviate from regular patterns.
#
# Pre-requisite:
# Ensure the 'models/' directory exists before saving outputs.
# ------------------------------------------------------------------------------


import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from calc import IsacCalculator

os.makedirs("models", exist_ok=True)
print("[INFO] Loading dataset...")
df = pd.read_csv("data/isac-ml-training-dataset-v2.csv")

print("[INFO] Performing feature engineering...")
def compute_row_metrics(row):
    calc = IsacCalculator(
        time_played_total=int(row["timePlayed"]),
        headshots=int(row["headshots"]),
        sum_hits=int(row["weaponHits"]),
        kills_npc=int(row["npcKills"]),
        kills_headshot=int(row["headshotKills"])
    )
    ratio_data = calc.calculate_percentages_and_ratios() or {}
    total_bodyshots = ratio_data.get("totalBodyshots", 0)
    total_headshots = ratio_data.get("totalHeadshots", 0)
    hs_ratio = ratio_data.get("percentageOfHeadshots", 0)
    bs_ratio = ratio_data.get("percentageOfBodyshots", 0)
    hs_to_bs_ratio = ratio_data.get("hsToBsRatio", np.nan)
    bs_to_hs_ratio = ratio_data.get("bsToHsRatio", np.nan)

    return pd.Series({
        "weaponHitsPerHour": calc.calculate_weapon_hits_per_hour(),
        "headshotsPerHour": calc.calculate_headshots_per_hour(),
        "bodyshotsPerHour": calc.calculate_bodyshots_per_hour(),
        "headshotKillsPerHour": calc.calculate_headshot_kills_per_hour(),
        "npcKillsPerHour": calc.calculate_npc_kills_per_hour(),
        "percentageOfHeadshots": hs_ratio,
        "percentageOfBodyshots": bs_ratio,
        "hsToBsRatio": hs_to_bs_ratio,
        "bsToHsRatio": bs_to_hs_ratio,
        "criticalBsToHsRatio": int(bs_to_hs_ratio < 1.29),
        "criticalHsToBsRatio": int(hs_to_bs_ratio > 1.29),
        "moreHeadshotsThanBodyshots": int(total_headshots > total_bodyshots),
        "hsRatioAboveBsRatio": int(hs_ratio > bs_ratio),
    })

df = pd.concat([df, df.apply(compute_row_metrics, axis=1)], axis=1)

if "bodyshots" not in df.columns:
    df["bodyshots"] = df["weaponHits"] - df["headshots"]

df["headshotsPerHourCritical"] = (df["headshotsPerHour"] > 803).astype(int)
df["headshotRatio"] = df["headshots"] / (df["weaponHits"] + 1)
df["headshotKillRatio"] = df["headshotKills"] / (df["npcKills"] + 1)
df["hsRatioAboveBsRatio"] = (df["headshotRatio"] > (df["bodyshots"] / (df["weaponHits"] + 1))).astype(int)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

print("[INFO] Upsampling cheater rows to balance signal...")
df_majority = df[df.isCheater == 0]
df_minority = df[df.isCheater == 1]

# Upsample cheaters to ~20% of the dataset
df_minority_upsampled = df_minority.sample(
    n=len(df_majority) // 8,
    replace=True,
    random_state=42
)

df = pd.concat([df_majority, df_minority, df_minority_upsampled])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"[INFO] Dataset after upsampling: {len(df)} rows")

df["cheaterRuleMatch"] = (
    (df["headshotsPerHour"] > 803).astype(int) &
    (df["bsToHsRatio"] < 1.29).astype(int) &
    (df["percentageOfHeadshots"] > df["percentageOfBodyshots"]).astype(int)
)

feature_columns = [
    "weaponHitsPerHour", "headshotsPerHour", "bodyshotsPerHour",
    "headshotKillsPerHour", "npcKillsPerHour",
    "percentageOfHeadshots", "percentageOfBodyshots",
    "hsToBsRatio", "bsToHsRatio",
    "headshotRatio", "headshotKillRatio",
    "headshotsPerHourCritical", "criticalBsToHsRatio", "criticalHsToBsRatio",
    "moreHeadshotsThanBodyshots", "hsRatioAboveBsRatio", "cheaterRuleMatch"
]

X = df[feature_columns]
y = df["isCheater"]

print(f"[INFO] Loaded {len(df)} rows")

print("[INFO] Sample of features before scaling:")
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("[INFO] Training model...")
model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train_scaled, y_train,
    epochs=15,
    batch_size=256,
    validation_split=0.2,
    class_weight={0: 1, 1: 5},
    verbose=1
)

print("[INFO] Evaluating model...")
y_proba = model.predict(X_test_scaled).flatten()
y_pred = (y_proba > 0.55).astype(int)
print(classification_report(y_test, y_pred))

# Show top 10 cheater probabilities for validation insight
cheaters = X_test[y_test == 1]
cheater_probs = model.predict(X_test_scaled[y_test == 1]).flatten()
top_cheaters = np.sort(cheater_probs)[::-1][:10]

print("[DEBUG] Top cheater probabilities (class 1):")
print(top_cheaters)

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Binary Crossentropy Loss')
plt.legend()
plt.grid(True)
plt.show()

print("[INFO] Saving model and scaler...")
model.save("models/v6_cheater_nn_model.keras")
joblib.dump(scaler, "models/v6_cheater_scaler.pkl")

print("[INFO] Computing and saving feature statistics...")
feature_medians = X.median(numeric_only=True)
feature_stds = X.std(numeric_only=True)

# Robust MAD computation
binary_columns = [
    "headshotsPerHourCritical", "criticalBsToHsRatio", "criticalHsToBsRatio",
    "moreHeadshotsThanBodyshots", "hsRatioAboveBsRatio"
]

feature_mads = pd.Series(index=feature_columns, dtype=np.float64)
for col in feature_columns:
    if col in binary_columns:
        feature_mads[col] = 1.0
    else:
        col_values = X[col]
        col_median = np.median(col_values)
        mad = np.median(np.abs(col_values - col_median))
        feature_mads[col] = mad if mad != 0 else 1e-6

# Save stats
feature_medians.to_json("models/v6_feature_medians.json")
feature_stds.to_json("models/v6_feature_stds.json")
feature_mads.to_json("models/v6_feature_mads.json")

print("[INFO] Done. Model, scaler, and stats saved.")
