from pathlib import Path
import numpy as np
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


time = data["time_s"].to_numpy(dtype=float)
dt = np.mean(np.diff(time))
sampling_rate = 1 / dt


signals = [
    ("normal_voltage_v", "Normal AC Voltage"),
    ("sag_voltage_v", "Voltage Sag"),
    ("swell_voltage_v", "Voltage Swell"),
    ("interruption_voltage_v", "Voltage Interruption"),
    ("transient_voltage_v", "Electrical Transient"),
]


fig, axes = plt.subplots(
    5, 1, figsize=(11, 13), sharex=True
)

print("\n--- EngiSense FFT Comparison ---")

for ax, (column, title) in zip(axes, signals):

    voltage = data[column].to_numpy(dtype=float)
    N = len(voltage)


    fft_values = np.fft.rfft(voltage)
    frequencies = np.fft.rfftfreq(N, d=dt)

    amplitudes = np.abs(fft_values) / N


    if N % 2 == 0:
        amplitudes[1:-1] *= 2
    else:
        amplitudes[1:] *= 2


    dominant_index = np.argmax(amplitudes[1:]) + 1

    print(
        f"{title}: "
        f"{frequencies[dominant_index]:.1f} Hz, "
        f"{amplitudes[dominant_index]:.2f} V"
    )


    ax.plot(frequencies, amplitudes)
    ax.set_title(title)
    ax.set_ylabel("Amplitude (V)")
    ax.set_xlim(0, 1500)
    ax.grid(True)

axes[-1].set_xlabel("Frequency (Hz)")

fig.suptitle(
    "EngiSense Analytics - FFT Comparison",
    fontsize=15
)


plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2.0)


reports_folder = project_folder / "reports"
reports_folder.mkdir(parents=True, exist_ok=True)

output_file = reports_folder / "power_quality_fft_comparison.png"

fig.savefig(output_file, dpi=300, bbox_inches="tight")

print("\nFFT comparison saved to:", output_file)

plt.show()
