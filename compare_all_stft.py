from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import stft

project_folder = Path(__file__).resolve().parent

csv_file = (
    project_folder / "data" / "raw" /
    "matlab" / "power_quality_signals.csv"
)

data = pd.read_csv(csv_file)

time = data["time_s"].to_numpy(dtype=float)
sampling_rate = 1 / np.mean(np.diff(time))

signals = [
    ("normal_voltage_v", "Normal Voltage"),
    ("sag_voltage_v", "Voltage Sag"),
    ("swell_voltage_v", "Voltage Swell"),
    ("interruption_voltage_v", "Voltage Interruption"),
    ("transient_voltage_v", "Electrical Transient"),
]

fig, axes = plt.subplots(
    5, 1, figsize=(11, 15), sharex=True, sharey=True
)

for ax, (column, title) in zip(axes, signals):

    values = data[column].to_numpy(dtype=float)

    frequencies, times, Zxx = stft(
        values,
        fs=sampling_rate,
        window="hann",
        nperseg=100,
        noverlap=90
    )

    magnitude_db = 20 * np.log10(
        np.abs(Zxx) + 1e-10
    )

    mesh = ax.pcolormesh(
        times,
        frequencies,
        magnitude_db,
        shading="auto",
        cmap="viridis",
        vmin=-60,
        vmax=45
    )

    ax.set_title(title)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(0, 1500)
    ax.grid(False)

axes[-1].set_xlabel("Time (seconds)")
axes[-1].set_xlim(0, 0.1)

fig.suptitle(
    "EngiSense - Power Quality STFT Comparison",
    fontsize=15
)

fig.subplots_adjust(
    top=0.94,
    bottom=0.06,
    left=0.10,
    right=0.87,
    hspace=0.40
)

fig.colorbar(
    mesh,
    ax=axes,
    label="Magnitude (dB)",
    shrink=0.75
)

reports_folder = project_folder / "reports"
reports_folder.mkdir(parents=True, exist_ok=True)

output_file = reports_folder / "all_power_quality_stft.png"

fig.savefig(output_file, dpi=300, bbox_inches="tight")

print("All five STFT plots saved successfully!")
print("Saved to:", output_file)

plt.show()
