from pathlib import Path
import pandas as pd
import streamlit as st
import numpy as np
from scipy.signal import stft

st.set_page_config(
    page_title="EngiSense Analytics",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
<style>
/* Bold, uppercase headings */
h1, h2, h3, h4, h5, h6 {
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

/* Bold, uppercase tab labels */
button[data-baseweb="tab"] {
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

/* Bold, uppercase metric labels */
[data-testid="stMetricLabel"] {
    font-weight: 800 !important;
    text-transform: uppercase !important;
}

/* Bold, uppercase chart and input labels */
[data-testid="stWidgetLabel"] {
    font-weight: 700 !important;
    text-transform: uppercase !important;
}
</style>
""", unsafe_allow_html=True)


project = Path(__file__).resolve().parent

st.title("⚡ ENGISENSE ANALYTICS")
st.caption("Electrical Engineering Monitoring and Intelligent Diagnosis")

# Navigation
tab1, tab2 = st.tabs([
    "ESP32 Sensor Monitoring",
    "MATLAB + Machine Learning"
])


with tab1:
    st.header("ESP32 Sensor Monitoring")

    esp32_file = (
        project / "data/unified/esp32/esp32_integrated.csv"
    )

    if esp32_file.exists():
        esp32 = pd.read_csv(esp32_file)

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Temperature",
            f"{esp32['temperature_c'].iloc[-1]:.1f} °C"
        )

        col2.metric(
            "Humidity",
            f"{esp32['humidity_pct'].iloc[-1]:.1f} %"
        )

        col3.metric(
            "ADC Voltage",
            f"{esp32['voltage_v'].iloc[-1]:.2f} V"
        )

        st.subheader("ESP32 Voltage Measurements")

        st.line_chart(
            esp32.set_index("time_s")["voltage_v"]
        )

        st.subheader("Sensor Data")
        st.dataframe(esp32, use_container_width=True)

    else:
        st.warning("ESP32 integrated data not found.")


with tab2:
    st.header("MATLAB Power Quality + ML")

    matlab_file = (
        project / "data/raw/matlab/power_quality_signals.csv"
    )

    predictions_file = (
        project / "reports/matlab_ml_predictions.csv"
    )

    if matlab_file.exists():
        matlab = pd.read_csv(matlab_file)

        signal_columns = {
            "Normal": "normal_voltage_v",
            "Sag": "sag_voltage_v",
            "Swell": "swell_voltage_v",
            "Interruption": "interruption_voltage_v",
            "Transient": "transient_voltage_v"
        }

        selected = st.selectbox(
            "Select electrical signal",
            list(signal_columns.keys())
        )

        st.subheader(f"{selected} Voltage Waveform")

        st.line_chart(
            matlab.set_index("time_s")[
                signal_columns[selected]
            ]
        )


        st.subheader("Frequency Spectrum (FFT)")

        time = matlab["time_s"].to_numpy(dtype=float)
        voltage = matlab[
            signal_columns[selected]
        ].to_numpy(dtype=float)

        sampling_rate = 1 / np.mean(np.diff(time))

        frequencies = np.fft.rfftfreq(
            len(voltage),
            d=1 / sampling_rate
        )

        amplitudes = (
            2 * np.abs(np.fft.rfft(voltage))
            / len(voltage)
        )


        amplitudes[0] /= 2
        if len(voltage) % 2 == 0:
            amplitudes[-1] /= 2

        fft_data = pd.DataFrame({
            "Frequency (Hz)": frequencies,
            "Amplitude (V)": amplitudes
        })

        fft_data = fft_data[
            fft_data["Frequency (Hz)"] <= 1500
        ]

        st.line_chart(
            fft_data.set_index("Frequency (Hz)")
        )


        st.subheader("Transient Event Detection")

        f_stft, t_stft, Z = stft(
            voltage,
            fs=sampling_rate,
            window="hann",
            nperseg=100,
            noverlap=90
        )

        hf_band = (
            (f_stft >= 700)
            & (f_stft <= 1500)
        )

        hf_energy = np.sum(
            np.abs(Z[hf_band, :]) ** 2,
            axis=0
        )

        peak_energy = float(np.max(hf_energy))

        concentration = peak_energy / (
            float(np.mean(hf_energy)) + 1e-12
        )

        transient_detected = (
            peak_energy > 100
            and concentration > 5
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Peak HF Energy",
            f"{peak_energy:.1f}"
        )

        col2.metric(
            "Energy Concentration",
            f"{concentration:.2f}"
        )

        col3.metric(
            "Transient Detected",
            "YES" if transient_detected else "NO"
        )

        if transient_detected:
            st.warning(
                "Localized high-frequency transient detected."
            )
        else:
            st.success(
                "No transient detected by the current thresholds."
            )


    else:
        st.warning("MATLAB data not found.")

    if predictions_file.exists():
        predictions = pd.read_csv(predictions_file)

        st.subheader("Machine Learning Predictions")
        st.dataframe(predictions, use_container_width=True)

        accuracy = predictions["correct"].mean() * 100

        st.metric(
            "MATLAB Signal Classification Accuracy",
            f"{accuracy:.1f}%"
        )

    else:
        st.warning("ML predictions file not found.")
