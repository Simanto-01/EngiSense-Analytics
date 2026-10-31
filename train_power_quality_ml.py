from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


project = Path(__file__).resolve().parent

csv_file = (
    project / "data" / "processed"
    / "power_quality_training.csv"
)

data = pd.read_csv(csv_file)


X = data.drop(columns=["signal_type"])
y = data["signal_type"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n--- EngiSense ML Training ---")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print(f"Test accuracy: {accuracy * 100:.2f}%")


model_folder = project / "models"
model_folder.mkdir(parents=True, exist_ok=True)

model_file = model_folder / "power_quality_rf.joblib"

if model_file.exists():
    print("Model file already exists; not overwriting.")
else:
    joblib.dump(model, model_file)
    print("Model saved to:", model_file)
