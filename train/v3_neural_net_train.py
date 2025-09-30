# 1. Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
import json

# 2. Load dataset
df = pd.read_csv("isac-ml-training-dataset.csv")

# 3. Feature Engineering
# Normalize by playtime in hours
playtime_hours = df["timePlayed"]

df["weaponHitsPerHour"] = df["weaponHits"] / (playtime_hours + 1)
df["headshotsPerHour"] = df["headshots"] / (playtime_hours + 1)
df["bodyshotsPerHour"] = df["bodyshots"] / (playtime_hours + 1)
df["headshotKillsPerHour"] = df["headshotKills"] / (playtime_hours + 1)
df["npcKillsPerHour"] = df["npcKills"] / (playtime_hours + 1)

# Derived features
df["suspiciousHeadshotRate"] = (df["headshotsPerHour"] > 750).astype(int)
df["headshotRatio"] = df["headshots"] / (df["weaponHits"] + 1)
df["headshotKillRatio"] = df["headshotKills"] / (df["npcKills"] + 1)
df["hsOverBs"] = df["headshots"] / (df["bodyshots"] + 1)
df["isExtremePlaytime"] = (df["timePlayed"] > 5000).astype(int)

# Final list of features to use in training
feature_columns = [
    "headshotsPerHour", "bodyshotsPerHour", "weaponHitsPerHour",
    "headshotKillsPerHour", "npcKillsPerHour",
    "headshotRatio", "headshotKillRatio", "hsOverBs",
    "suspiciousHeadshotRate", "isExtremePlaytime"
]

# 4. Prepare X and y
X = df[feature_columns]
y = df["isCheater"]

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Build the model
model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Binary output
])

# 8. Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 9. Train the model with class weights
history = model.fit(
    X_train_scaled, y_train,
    epochs=10,
    batch_size=256,
    validation_split=0.1,
    class_weight={0: 1, 1: 5},
    verbose=1
)

# 10. Predict and evaluate
y_proba = model.predict(X_test_scaled).flatten()
y_pred = (y_proba > 0.45).astype(int)  # custom threshold for better recall

print(classification_report(y_test, y_pred))

# 11. Plot training/validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Binary Crossentropy Loss')
plt.legend()
plt.grid(True)
plt.show()

# 12. Save feature medians for deviation reports
feature_medians = pd.Series(X.median(numeric_only=True))
feature_medians.to_json("v4_feature_medians.json")

# 13. Save featue medians for deviation reports in a better format
feature_stds = pd.Series(X.std(numeric_only=True))
feature_stds.to_json("v4_feature_stds.json")

# 13. Save model and scaler
model.save("v4_cheater_nn_model.keras")
joblib.dump(scaler, "v4_cheater_scaler.pkl")

