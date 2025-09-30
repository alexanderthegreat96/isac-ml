# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# 2. Load the dataset
df = pd.read_csv("isac-ml-training-dataset.csv")  # adjust path if needed

# 3. Drop non-numeric / non-predictive columns
X = df.drop(columns=["username", "identifier", "isCheater"])
y = df["isCheater"]

# 4. Optionally add a rule-based feature
X["suspiciousHeadshotRate"] = (df["headshotsPerHour"] > 750).astype(int)

# 5. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Scale features (StandardScaler: mean=0, std=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Train a classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# 8. Evaluate
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# 9. (Optional) Save model and scaler
joblib.dump(model, "cheater_model.pkl")
joblib.dump(scaler, "scaler.pkl")

