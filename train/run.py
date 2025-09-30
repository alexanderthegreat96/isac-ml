import joblib

# Load trained model and scaler
model = joblib.load("cheater_model.pkl")
scaler = joblib.load("scaler.pkl")

import pandas as pd

# Example: new sample(s) in a dataframe (can be a single row or many)
new_data = pd.DataFrame([{
    "timePlayed": 3500,
    "weaponHits": 400000,
    "headshots": 300000,
    "bodyshots": 100000,
    "headshotKills": 90000,
    "npcKills": 5000000,
    "headshotsPerHour": 800,   # suspicious
    "bodyshotsPerHour": 250,
    "weaponHitsPerHour": 1050,
    "suspiciousHeadshotRate": 0  # you should calculate this if you're using it
}])

new_data_scaled = scaler.transform(new_data)

prob = model.predict_proba(new_data_scaled)[0][1]  # probability of being a cheater

# Predict label (0 = legit, 1 = cheater)
label = model.predict(new_data_scaled)[0]

print(f"Predicted isCheater: {label} (probability: {prob:.2f})")

