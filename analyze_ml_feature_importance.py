from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt

project = Path(__file__).resolve().parent

model = joblib.load(
    project / "models/power_quality_rf.joblib"
)

features = pd.Series(
    model.feature_importances_,
    index=model.feature_names_in_
).sort_values(ascending=False)

print("\n--- EngiSense Feature Importance ---")
print(features.round(4))

fig, ax = plt.subplots(figsize=(10, 6))

features.sort_values().plot.barh(ax=ax)

ax.set_title("EngiSense - Random Forest Feature Importance")
ax.set_xlabel("Feature Importance")
ax.set_ylabel("Electrical Signal Features")

plt.tight_layout()

output_file = (
    project / "reports/ml_feature_importance.png"
)

if output_file.exists():
    print("\nPlot already exists; not overwriting.")
else:
    fig.savefig(output_file, dpi=300)
    print("\nSaved to:", output_file)

plt.show()
