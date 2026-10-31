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
voltage = data["normal_voltage_v"].to_numpy(dtype=float)


sampling_interval = np.mean(np.diff(time))
sampling_rate = 1 / sampling_interval


N = len(voltage)

fft_values = np.fft.rfft(voltage)
frequencies = np.fft.rfftfreq(N, d=sampling_interval)


amplitudes = np.abs(fft_values) / N
amplitudes[1:-1] *= 2


dominant_index = np.argmax(amplitudes[1:]) + 1
dominant_frequency = frequencies[dominant_index]

print("\n--- EngiSense Frequency Analysis ---")
print(f"Sampling Frequency: {sampling_rate:.2f} Hz")
print(f"Dominant Frequency: {dominant_frequency:.2f} Hz")
print(f"Peak FFT Amplitude: {amplitudes[dominant_index]:.2f} V")


plt.figure(figsize=(10, 5))
plt.plot(frequencies, amplitudes)
plt.title("EngiSense - Normal Voltage FFT")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Peak Amplitude (V)")
plt.xlim(0, 500)
plt.grid(True)
plt.tight_layout()


reports_folder = project_folder / "reports"
reports_folder.mkdir(parents=True, exist_ok=True)

output_file = reports_folder / "python_normal_voltage_fft.png"
plt.savefig(output_file, dpi=300)

print("FFT plot saved to:", output_file)

plt.show()
