# 1. Imports
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

# 2. Load dataset
df = pd.read_csv("isac-ml-training-dataset.csv")

# 3. Feature Engineering
X = df.drop(columns=["username", "identifier", "isCheater"])
y = df["isCheater"]

# Rule-based and derived features
X["suspiciousHeadshotRate"] = (df["headshotsPerHour"] > 750).astype(int)
X["headshotRatio"] = df["headshots"] / (df["weaponHits"] + 1)
X["headshotKillRatio"] = df["headshotKills"] / (df["npcKills"] + 1)
X["hsOverBs"] = df["headshots"] / (df["bodyshots"] + 1)
X["isExtremePlaytime"] = (df["timePlayed"] > 5000).astype(int)

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Build the model
model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Binary output
])

# 7. Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 8. Train the model with class weights
history = model.fit(
    X_train_scaled, y_train,
    epochs=10,
    batch_size=256,
    validation_split=0.1,
    class_weight={0: 1, 1: 5},
    verbose=1
)

# 9. Predict and Evaluate with custom threshold
y_proba = model.predict(X_test_scaled).flatten()
y_pred = (y_proba > 0.45).astype(int)  # lower threshold to boost recall

print(classification_report(y_test, y_pred))

# 10. Plot training/validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Binary Crossentropy Loss')
plt.legend()
plt.grid(True)
plt.show()

# 11. Save medians
feature_medians = pd.Series(X.median(numeric_only=True))
feature_medians.to_json("v2_feature_medians.json")

# 12. Save model and scaler
model.save("v2_cheater_nn_model.keras")
joblib.dump(scaler, "v2_cheater_scaler.pkl")

