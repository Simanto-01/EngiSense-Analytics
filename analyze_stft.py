from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import stft


project_folder = Path(__file__).resolve().parent

csv_file = (
    project_folder
    / "data"
    / "raw"
    / "matlab"
    / "power_quality_signals.csv"
)

data = pd.read_csv(csv_file)


time = data["time_s"].to_numpy(dtype=float)
voltage = data["transient_voltage_v"].to_numpy(dtype=float)


dt = np.mean(np.diff(time))
sampling_rate = 1 / dt


frequencies, times, Zxx = stft(
    voltage,
    fs=sampling_rate,
    window="hann",
    nperseg=100,
    noverlap=90
)


magnitude_db = 20 * np.log10(
    np.abs(Zxx) + 1e-10
)


fig, ax = plt.subplots(figsize=(11, 6))

mesh = ax.pcolormesh(
    times,
    frequencies,
    magnitude_db,
    shading="auto",
    cmap="viridis"
)

ax.set_title("EngiSense - Electrical Transient STFT")
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Frequency (Hz)")
ax.set_ylim(0, 1500)
ax.set_xlim(0, 0.1)

fig.colorbar(mesh, ax=ax, label="Magnitude (dB)")

plt.tight_layout()


reports_folder = project_folder / "reports"
reports_folder.mkdir(parents=True, exist_ok=True)

output_file = reports_folder / "electrical_transient_stft.png"

fig.savefig(output_file, dpi=300, bbox_inches="tight")

print("STFT plot saved successfully!")
print("Saved to:", output_file)

plt.show()



normal_voltage = data["normal_voltage_v"].to_numpy(dtype=float)
transient_voltage = data["transient_voltage_v"].to_numpy(dtype=float)

signals = [
    ("Normal AC Voltage", normal_voltage),
    ("Electrical Transient", transient_voltage),
]

fig, axes = plt.subplots(
    2, 1, figsize=(11, 9), sharex=True, sharey=True
)

for ax, (title, signal) in zip(axes, signals):

    f, t_stft, Z = stft(
        signal,
        fs=sampling_rate,
        window="hann",
        nperseg=100,
        noverlap=90
    )

    magnitude_db = 20 * np.log10(np.abs(Z) + 1e-10)

    mesh = ax.pcolormesh(
        t_stft,
        f,
        magnitude_db,
        shading="auto",
        cmap="viridis",
        vmin=-60,
        vmax=45
    )

    ax.set_title(title)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(0, 1500)
    ax.axvline(
        0.04,
        color="red",
        linestyle="--",
        label="Transient start"
    )
    ax.legend(loc="upper right")

axes[-1].set_xlabel("Time (seconds)")
axes[-1].set_xlim(0, 0.1)

fig.suptitle(
    "EngiSense - Normal vs Transient STFT",
    fontsize=15
)

fig.colorbar(
    mesh,
    ax=axes,
    label="Magnitude (dB)",
    shrink=0.8
)

fig.subplots_adjust(
    top=0.91,
    bottom=0.09,
    left=0.10,
    right=0.88,
    hspace=0.32
)

output_file = (
    reports_folder / "normal_vs_transient_stft.png"
)

fig.savefig(output_file, dpi=300, bbox_inches="tight")

print("\nSTFT comparison saved successfully!")
print("Saved to:", output_file)

plt.show()
