from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay
)

project = Path(__file__).resolve().parent


data = pd.read_csv(
    project / "data/processed/power_quality_training.csv"
)

model = joblib.load(
    project / "models/power_quality_rf.joblib"
)

X = data.drop(columns=["signal_type"])
y = data["signal_type"]


_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

predictions = model.predict(X_test)

print("\n--- EngiSense Classification Report ---")
print(classification_report(y_test, predictions))


fig, ax = plt.subplots(figsize=(9, 7))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    labels=model.classes_,
    cmap="Blues",
    ax=ax,
    xticks_rotation=30
)

ax.set_title("EngiSense - ML Confusion Matrix")

plt.tight_layout()

output_file = (
    project / "reports/ml_confusion_matrix.png"
)

fig.savefig(output_file, dpi=300, bbox_inches="tight")

print("\nConfusion matrix saved to:", output_file)

plt.show()
