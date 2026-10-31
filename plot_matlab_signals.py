
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


project_folder = Path(__file__).resolve().parent

csv_file = (
    project_folder
    / "data"
    / "raw"
    / "matlab"
    / "power_quality_signals.csv"
)


data = pd.read_csv(csv_file)


signals = [
    ("normal_voltage_v", "Normal AC Voltage"),
    ("sag_voltage_v", "Voltage Sag"),
    ("swell_voltage_v", "Voltage Swell"),
    ("interruption_voltage_v", "Voltage Interruption"),
    ("transient_voltage_v", "Electrical Transient"),
]

fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)

for ax, (column, title) in zip(axes, signals):
    ax.plot(data["time_s"], data[column])
    ax.set_title(title)
    ax.set_ylabel("Voltage (V)")
    ax.grid(True)

axes[-1].set_xlabel("Time (seconds)")

fig.suptitle(
    "EngiSense Analytics - MATLAB Signals in Python",
    fontsize=15
)


plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2.0)


reports_folder = project_folder / "reports"
reports_folder.mkdir(parents=True, exist_ok=True)


output_file = reports_folder / "py_power_quality_comparison.png"

fig.savefig(output_file, dpi=300, bbox_inches="tight")

print("Plot saved to:", output_file)

plt.show()
