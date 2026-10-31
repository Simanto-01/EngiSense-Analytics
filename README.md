# ******_EngiSense Analytics_******

### Electrical Engineering Monitoring, Signal Processing & Intelligent Diagnostics

**From embedded sensor measurements to power-quality analysis and machine-learning-based fault identification.**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![MATLAB](https://img.shields.io/badge/MATLAB-Signal%20Simulation-orange)
![ESP32](https://img.shields.io/badge/ESP32-Wokwi%20Simulation-00979D)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-V1%20Prototype-brightgreen)

---

## Overview

**EngiSense Analytics** is an electrical engineering project that brings together embedded systems, MATLAB simulations, digital signal processing, and machine learning in a single analytical workflow.

The project started with two practical questions: how can electrical and sensor measurements be analyzed more effectively, and how reliably can an intelligent system distinguish different electrical disturbances?

To explore these questions, I developed an ESP32-based sensor simulation in Wokwi, generated power-quality waveforms in MATLAB, and built a Python pipeline for signal analysis and disturbance classification. The results are brought together in an interactive Streamlit dashboard.

The work goes beyond building a classifier. After testing the machine-learning model under different noise conditions, I investigated why its predictions deteriorated and developed an additional STFT-based transient detector to address one of the observed weaknesses.

EngiSense Analytics is currently a **simulation-based engineering prototype**. It is designed for experimentation, learning, and further development rather than direct deployment on an electrical power network.

### What the project covers

- ESP32 temperature, humidity, and ADC voltage monitoring
- MATLAB simulation of normal and disturbed AC voltage waveforms
- Time-domain electrical measurements and feature extraction
- Fast Fourier Transform (FFT) analysis
- Short-Time Fourier Transform (STFT) analysis
- Random Forest power-quality classification
- Machine-learning noise-robustness experiments
- Physics-informed electrical transient detection
- Interactive visualization through Streamlit

---

## 1. System Architecture

EngiSense Analytics combines two independent sources of simulated measurements.

The first is an ESP32 sensor circuit developed in Wokwi. It provides temperature, humidity, and ADC voltage readings. The second is a MATLAB-based electrical simulation that produces high-sampling-rate AC voltage waveforms.

Both datasets are exported to CSV and processed in Python. The electrical signals are then analyzed using signal-processing methods and machine-learning techniques.

```text
                         ENGISENSE ANALYTICS
                                  |
                 +----------------+----------------+
                 |                                 |
            ESP32 / WOKWI                        MATLAB
           Sensor Simulation              Power-Quality Simulation
                 |                                 |
          DHT22 Measurements                  Normal Voltage
          ADC Voltage Input                    Voltage Sag
                 |                             Voltage Swell
                 |                         Voltage Interruption
                 |                         Electrical Transient
                 |                         Harmonic Distortion
                 |                                 |
                 v                                 v
            CSV Export                        CSV Export
                 |                                 |
                 +----------------+----------------+
                                  |
                                  v
                            PYTHON ANALYTICS
                                  |
                   +--------------+--------------+
                   |                             |
             Data Processing              Signal Processing
                   |                             |
             Data Validation                 RMS / FFT / STFT
             Feature Extraction              Spectral Analysis
                   |                             |
                   +--------------+--------------+
                                  |
                                  v
                         INTELLIGENT ANALYSIS
                                  |
                         Random Forest Model
                         Noise Robustness Tests
                         Transient Event Detector
                                  |
                                  v
                          STREAMLIT DASHBOARD
                                  |
                         Sensor Measurements
                         Electrical Waveforms
                         Frequency Spectra
                         Classification Results
                         Transient Detection
```

The distinction between the two data sources is important. The ESP32 simulation demonstrates embedded sensor acquisition, while the MATLAB dataset is used for electrical power-quality analysis. The low-rate ESP32 data is **not** used to measure 50 Hz mains waveforms.

---

## 2. ESP32 Sensor Monitoring

### 2.1 Circuit Design and Wokwi Simulation

The embedded monitoring system was developed using an ESP32 microcontroller in Wokwi.

The circuit consists of a DHT22 temperature and humidity sensor and a potentiometer connected to an ESP32 analog input. The potentiometer provides a variable voltage that can be measured through the microcontroller's ADC.

This setup provides a straightforward way to explore sensor interfacing, analog measurements, serial data acquisition, and the transfer of embedded measurements into Python.

![ESP32 Circuit in Wokwi](reports/Esp32_Con.png)

*Figure 1. ESP32 circuit developed in Wokwi, showing the DHT22 sensor and potentiometer connected to the microcontroller.*

The Wokwi project files are retained in `firmware/wokwi_esp32/`, including the Arduino sketch, circuit diagram, and library configuration.

### 2.2 Data Acquisition

A total of **52 sensor readings** were collected during the simulation.

The exported dataset includes:

| Field | Description |
|---|---|
| `time_ms` | Timestamp in milliseconds |
| `temperature_c` | Temperature in °C |
| `humidity_pct` | Relative humidity in % |
| `adc_raw` | Raw ADC reading |
| `voltage_v` | Calculated ADC voltage in V |

The recorded measurements produced the following approximate values:

| Parameter | Result |
|---|---:|
| Mean temperature | 24 °C |
| Mean humidity | 40% |
| Mean ADC voltage | 1.48 V |
| Sampling frequency | 0.498 Hz |
| Number of samples | 52 |

The ADC measurements cover a simulated voltage range of approximately 0–3.3 V.

### 2.3 ADC Voltage Analysis

The potentiometer voltage was recorded over time and visualized in Python.

![ESP32 ADC Voltage Measurements](reports/wokwi_voltage_graph.png)

*Figure 2. ADC voltage measurements obtained from the Wokwi ESP32 simulation.*

The plot shows how the measured voltage changes as the potentiometer input varies. It also demonstrates the complete path from a simulated analog input to a dataset that can be analyzed and displayed in Python.

The original data is stored at:

`data/raw/wokwi/esp32_sensor_data.csv`

The integrated dataset is stored at:

`data/unified/esp32/esp32_integrated.csv`

The integration process preserves the original measurements and adds a time value in seconds and a source identifier.

---

## 3. MATLAB Power-Quality Simulation

### 3.1 Reference AC Signal

MATLAB was used to generate a reference 50 Hz AC voltage waveform and several representative power-quality disturbances.

The reference signal is defined by:

\[
v(t)=V_m\sin(2\pi ft)
\]

The simulation uses the following parameters:

| Parameter | Value |
|---|---:|
| Fundamental frequency | 50 Hz |
| Peak voltage | 325 V |
| Sampling frequency | 5,000 Hz |
| Simulation duration | 0.1 s |
| Number of samples | 500 |
| Reference RMS voltage | Approximately 230 V |

The 0.1-second observation window contains five cycles of the fundamental waveform.

### 3.2 Simulated Electrical Conditions

Five operating conditions were prepared for the Python analysis and machine-learning workflow.

**Normal Voltage**

A sinusoidal AC waveform representing the reference operating condition.

**Voltage Sag**

A temporary reduction in voltage amplitude to 50% of its normal value between 0.04 s and 0.08 s.

**Voltage Swell**

A temporary increase in voltage amplitude to 150% of its normal value over the same interval.

**Voltage Interruption**

A temporary reduction of the voltage waveform to zero between 0.04 s and 0.08 s.

**Electrical Transient**

A damped, high-frequency oscillation superimposed on the fundamental waveform, beginning at approximately 0.04 s. The reference transient contains a 1,000 Hz oscillatory component.

### 3.3 Waveform Comparison

![MATLAB Power Quality Waveforms](reports/py_power_quality_comparison.png)

*Figure 3. Python visualization of the five MATLAB-generated electrical waveforms: Normal, Sag, Swell, Interruption, and Transient.*

The differences are visible in the time domain. Sags and swells change the waveform amplitude, interruptions temporarily remove the signal, and the transient introduces a short burst of high-frequency oscillation.

These waveform characteristics provide the starting point for the signal-processing and classification experiments.

### 3.4 Harmonic Distortion

A separate MATLAB experiment investigates harmonic distortion by adding higher-order frequency components to the fundamental waveform.

The distorted waveform contains harmonics from the second through the tenth order, producing a Total Harmonic Distortion (THD) of approximately **21.02%**.

The harmonic experiment is used to demonstrate spectral analysis. It is not included as a sixth class in the current machine-learning model.

The MATLAB source and exported electrical data are stored at:

```text
data/raw/matlab/
├── engisense_ac_signal.m
└── power_quality_signals.csv
```

The exported CSV contains the time vector and the five waveforms used in the classification workflow.

The original MATLAB plots are also available in `reports/matlab_plots/`.

---

## 4. Digital Signal Processing

Signal processing forms an important part of EngiSense Analytics. Before training a classifier, the electrical waveforms are examined in both the time and frequency domains.

### 4.1 Time-Domain Analysis

Several numerical characteristics are calculated from each waveform, including:

- RMS voltage
- Peak voltage
- Mean voltage
- Standard deviation
- Crest factor
- Peak-to-peak voltage

RMS voltage is calculated as:

\[
V_{\mathrm{RMS}}=\sqrt{\frac{1}{N}\sum_{n=0}^{N-1}v[n]^2}
\]

Crest factor is defined as:

\[
\mathrm{Crest\ Factor}=\frac{V_{\mathrm{peak}}}{V_{\mathrm{RMS}}}
\]

These measurements help describe the overall magnitude and shape of a waveform.

### 4.2 RMS Comparison

The calculated RMS values for the five MATLAB reference signals are:

| Signal | RMS voltage |
|---|---:|
| Normal | 229.81 V |
| Sag | 192.27 V |
| Swell | 281.46 V |
| Interruption | 178.01 V |
| Transient | 230.42 V |

These values were calculated over the **complete 0.1-second records**, rather than only the intervals containing the disturbances.

This explains why, for example, the full-record RMS voltage of the sag waveform is higher than half the nominal RMS voltage.

### 4.3 Fast Fourier Transform

The Fast Fourier Transform (FFT) is used to examine the frequency components present in each waveform.

A normal sinusoidal signal is dominated by its 50 Hz fundamental frequency. Disturbances can introduce additional spectral content, although the appearance of that content depends on the event type and observation window.

![Power Quality FFT Comparison](reports/power_quality_fft_comparison.png)

*Figure 4. FFT comparison of the five MATLAB-generated electrical signals.*

The FFT plots show the dominant fundamental component and the additional frequency content associated with the simulated disturbances.

For the electrical transient, a smaller component is visible near 1,000 Hz.

### 4.4 Short-Time Fourier Transform

FFT analysis shows which frequencies are present, but it does not directly identify when those components occur.

To investigate the timing of electrical disturbances, the project also uses the Short-Time Fourier Transform (STFT).

STFT divides a waveform into overlapping windows and examines the frequency content within each window. The result is a time-frequency representation of the signal.

![Power Quality STFT Comparison](reports/all_power_quality_stft.png)

*Figure 5. STFT comparison of Normal, Sag, Swell, Interruption, and Transient waveforms.*

The transient is particularly interesting because its high-frequency content appears over a short interval rather than throughout the entire signal.

The spectrograms also contain some energy near the record boundaries and disturbance transitions. These effects must be considered when interpreting short-window spectral analysis.

### 4.5 Normal Versus Transient STFT

![Normal vs Transient STFT](reports/normal_vs_transient_stft.png)

*Figure 6. STFT comparison of normal AC voltage and an electrical transient, with the event beginning near 0.04 s.*

The normal waveform is dominated by low-frequency content, while the transient introduces a localized component around 1,000 Hz.

This observation later became central to the noise-robustness investigation.

---

## 5. Machine-Learning-Based Power-Quality Classification

### 5.1 Dataset Preparation

A synthetic training dataset was generated to develop a classifier for five electrical signal categories:

1. Normal
2. Sag
3. Swell
4. Interruption
5. Transient

The dataset contains **1,000 examples**, with 200 examples from each category.

Rather than using identical copies of the MATLAB waveforms, the training generator varies signal properties such as phase, amplitude, disturbance timing, and transient characteristics. A small amount of Gaussian noise is also included.

Seven numerical features are extracted from each generated waveform:

| Feature | Purpose |
|---|---|
| RMS voltage | Measures overall signal magnitude |
| Peak voltage | Captures maximum absolute voltage |
| Crest factor | Relates peak magnitude to RMS |
| Mean voltage | Measures average signal value |
| Standard deviation | Describes signal variability |
| Peak-to-peak voltage | Measures total voltage excursion |
| High-frequency energy | Captures higher-frequency spectral content |

The resulting feature dataset is saved at:

`data/processed/power_quality_training.csv`

### 5.2 Random Forest Model

A Random Forest classifier was trained using:

| Configuration | Value |
|---|---|
| Algorithm | Random Forest |
| Number of trees | 100 |
| Training split | 80% |
| Test split | 20% |
| Stratification | By signal class |
| Random seed | 42 |

The model learns relationships between the extracted numerical features and the corresponding disturbance labels.

### 5.3 Classification Results

The classifier achieved **99.0% accuracy** on the held-out synthetic test set.

It also correctly classified all five original MATLAB reference signals.

These are encouraging results, but they represent different evaluation conditions. The synthetic test split measures performance within the generated dataset, while the MATLAB evaluation uses only five reference examples.

Neither result should be interpreted as proof of equivalent performance on real-world power-quality measurements.

### 5.4 Confusion Matrix

![Random Forest Confusion Matrix](reports/ml_confusion_matrix.png)

*Figure 7. Confusion matrix showing the Random Forest classification results on the synthetic test dataset.*

### 5.5 Feature Importance

![Random Forest Feature Importance](reports/ml_feature_importance.png)

*Figure 8. Feature-importance results from the trained Random Forest model.*

The results indicate that high-frequency energy and voltage-magnitude-related features play important roles in classification.

Several features are correlated, particularly RMS voltage and standard deviation. The importance scores are therefore useful for understanding the model, but they should not be treated as independent causal explanations.

The trained classifier is saved at:

`models/power_quality_rf.joblib`

---

## 6. Investigating Machine-Learning Robustness

### 6.1 Why Noise Matters

High classification accuracy under controlled conditions does not necessarily mean that a model will remain reliable when measurements become noisy.

To examine this, Gaussian noise was added to the five MATLAB reference waveforms at several levels.

The noise standard deviation was expressed as a percentage of the 325 V reference peak amplitude.

### 6.2 Noise Robustness Results

| Added noise | Classification accuracy |
|---|---:|
| 0% | 100% |
| 1% | 91% |
| 3% | 77% |
| 5% | 60% |
| 10% | 60% |
| 15% | 59% |
| 20% | 48% |

Each noise level was evaluated using 100 predictions across the five signal categories.

The model performed well on the clean references, but its accuracy decreased considerably as noise increased.

This revealed a limitation that was not apparent from the original 99% synthetic test accuracy.

### 6.3 Error Analysis

A more detailed experiment was performed at 5% added noise, using 100 noisy realizations of each reference waveform.

| Actual signal | Correct predictions |
|---|---:|
| Normal | 0/100 |
| Sag | 0/100 |
| Swell | 100/100 |
| Interruption | 100/100 |
| Transient | 100/100 |

The noisy Normal examples were consistently classified as Transient. Most noisy Sag examples were classified as Interruption.

The results suggested that some of the features used by the model were sensitive to noise.

In particular, broadband noise increased the measured high-frequency energy, which could make an otherwise normal waveform resemble a transient according to the existing feature representation.

### 6.4 Investigating Filtering

A fourth-order Butterworth low-pass filter with a 1,500 Hz cutoff frequency was tested as a possible way to reduce the noise-related classification errors.

Although filtering reduced the measured high-frequency energy in the tested noisy Normal waveform, the Random Forest still classified it as Transient.

This result indicated that a simple low-pass filter was not sufficient for the observed failure.

A different approach was needed—one that considered not only the amount of high-frequency energy, but also how that energy was distributed over time.

---

## 7. Physics-Informed Transient Detection

### 7.1 From Noise Sensitivity to Energy Localization

A genuine electrical transient and broadband measurement noise can both introduce high-frequency components into a voltage waveform.

The key difference explored in this project is their temporal behavior.

Broadband noise tends to distribute energy across the observation window. A short electrical transient, by contrast, can create a concentrated burst of high-frequency energy around the event time.

STFT analysis makes this difference easier to investigate.

### 7.2 High-Frequency Energy Localization

The transient analysis uses an STFT with the following configuration:

| Parameter | Value |
|---|---|
| Sampling frequency | 5,000 Hz |
| Window | Hann |
| Window length | 100 samples |
| Overlap | 90 samples |
| High-frequency band | 700–1,500 Hz |

The high-frequency energy for each STFT window is calculated by summing the squared magnitudes of the spectral coefficients within the selected frequency band.

![High-Frequency Energy Localization](reports/transient_energy_localization.png)

*Figure 9. High-frequency STFT energy for clean Normal, noisy Normal (5%), and genuine Transient waveforms.*

The genuine transient produces a strong, localized energy peak near 0.042 s.

The noisy Normal signal shows lower, more distributed energy throughout the record.

This difference provides a useful basis for an additional detection method.

### 7.3 Energy Concentration

To quantify the localization, an energy concentration ratio is calculated:

\[
C=\frac{\max(E(t))}{\operatorname{mean}(E(t))+\varepsilon}
\]

where:

- \(E(t)\) represents the high-frequency energy in each STFT window.
- \(C\) is the concentration ratio.
- \(\varepsilon\) is a small numerical constant used to avoid division by zero.

The initial comparison produced:

| Signal | Concentration ratio |
|---|---:|
| Clean Normal | 10.71 |
| Noisy Normal (5%) | 2.22 |
| Genuine Transient | 11.32 |

One important observation is that the clean Normal signal also has a high concentration ratio, even though its absolute high-frequency energy is extremely small.

For that reason, the concentration ratio cannot be used reliably on its own.

### 7.4 Detection Method

The proposed detector combines two conditions:

1. The maximum high-frequency energy must exceed an absolute threshold.
2. The energy concentration ratio must exceed a localization threshold.

The current exploratory implementation uses:

```python
transient_detected = (
    peak_energy > 100
    and concentration > 5
)
```

These values were selected during the experiments. They are not standardized power-quality thresholds and would require further validation before use outside the simulated conditions.

### 7.5 Preliminary Detection Results

The detector correctly rejected the clean Normal reference waveform and detected the genuine MATLAB transient.

It was also evaluated against noisy Normal waveforms using 50 realizations at each of six noise levels.

| Noise level | False alarms |
|---|---:|
| 1% | 0/50 |
| 3% | 0/50 |
| 5% | 0/50 |
| 10% | 0/50 |
| 15% | 0/50 |
| 20% | 0/50 |
| **Total** | **0/300** |

No false alarms were observed in this controlled experiment.

This is a promising result, although the tests use a limited waveform setup and the thresholds were informed by earlier observations. The result should therefore be treated as preliminary rather than a general false-alarm guarantee.

### 7.6 Transient Amplitude Sensitivity

A separate experiment examined how transient amplitude affects detection.

With the original peak-energy threshold of 300, the results were:

| Transient amplitude | Peak energy | Detected |
|---|---:|---|
| 20 V | 39.6 | No |
| 40 V | 158.6 | No |
| 60 V | 356.8 | Yes |
| 80 V | 634.4 | Yes |
| 100 V | 991.2 | Yes |
| 150 V | 2,230.2 | Yes |
| 200 V | 3,964.8 | Yes |

The energy threshold was subsequently reduced to 100, and the noisy-Normal false-alarm experiment was repeated successfully.

The revised threshold has not yet undergone a comprehensive sensitivity study across different transient amplitudes, durations, frequencies, and noise levels.

### 7.7 Engineering Significance

The transient detector is not a replacement for the Random Forest classifier.

Instead, it provides a complementary method based on the time-localized behavior of high-frequency energy.

The main lesson from this investigation is that **a signal's temporal structure can provide diagnostic information that aggregate statistical features may miss**.

That is a useful direction for future work on more robust electrical disturbance identification.

---

## 8. Streamlit Dashboard

The Streamlit dashboard brings the project's measurements and analytical results into a single interface.

It is organized into two main sections.

### 8.1 ESP32 Sensor Monitoring

The ESP32 tab displays:

- Latest temperature measurement
- Latest relative humidity measurement
- Latest ADC voltage measurement
- ADC voltage history
- Integrated sensor dataset

### 8.2 MATLAB and Machine Learning

The electrical analysis tab allows the user to select one of the five MATLAB power-quality signals.

For the selected waveform, the dashboard displays:

- Time-domain voltage waveform
- FFT frequency spectrum
- STFT-based transient detection metrics
- Transient detection status

The tab also displays the saved machine-learning predictions and classification accuracy for the five MATLAB reference signals.

The dashboard uses previously generated datasets and results. It does not retrain the Random Forest or regenerate the MATLAB simulations when opened.

### Running the Dashboard

```bash
streamlit run engisense_dashboard.py
```

The application is available locally at:

`http://localhost:8501`

### Public Deployment

EngiSense Analytics is being prepared for deployment using Streamlit Community Cloud.

The intended public address is:

`https://engisense-analytics.streamlit.app`

The public URL will become active only after successful deployment and confirmation that the requested subdomain is available.

---

## 9. Project Structure

```text
EngiSense-Analytics/
│
├── .streamlit/
│   └── config.toml
│
├── data/
│   ├── raw/
│   │   ├── matlab/
│   │   │   ├── engisense_ac_signal.m
│   │   │   └── power_quality_signals.csv
│   │   └── wokwi/
│   │       └── esp32_sensor_data.csv
│   │
│   ├── processed/
│   │   ├── power_quality_features.csv
│   │   └── power_quality_training.csv
│   │
│   └── unified/
│       └── esp32/
│           └── esp32_integrated.csv
│
├── firmware/
│   └── wokwi_esp32/
│       ├── sketch.ino
│       ├── diagram.json
│       └── libraries.txt
│
├── models/
│   └── power_quality_rf.joblib
│
├── reports/
│   ├── matlab_plots/
│   ├── Esp32_Con.png
│   ├── wokwi_voltage_graph.png
│   ├── py_power_quality_comparison.png
│   ├── power_quality_fft_comparison.png
│   ├── all_power_quality_stft.png
│   ├── normal_vs_transient_stft.png
│   ├── transient_energy_localization.png
│   ├── ml_confusion_matrix.png
│   ├── ml_feature_importance.png
│   ├── ml_noise_robustness.csv
│   └── ml_noise_class_accuracy.csv
│
├── src/
│   └── engisense/
│       ├── __init__.py
│       ├── statistics.py
│       ├── validation.py
│       └── visualization.py
│
├── engisense_dashboard.py
├── analyze_power_quality.py
├── analyze_frequency.py
├── analyze_stft.py
├── analyze_transient_localization.py
├── generate_training_data.py
├── train_power_quality_ml.py
├── evaluate_power_quality_ml.py
├── predict_matlab_signals.py
├── test_ml_robustness.py
├── evaluate_transient_detector.py
├── validate_transient_amplitude.py
├── requirements.txt
├── .gitignore
└── README.md
```

Other analysis, visualization, validation, and data-integration scripts are also included in the repository.

---

## 10. Installation and Setup

### Prerequisites

- Python 3
- Git
- Python dependencies listed in `requirements.txt`

MATLAB is required to rerun the original electrical simulations. Wokwi is required to rerun the embedded simulation. Neither is required to view the already-exported datasets in the Streamlit dashboard.

### Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/EngiSense-Analytics.git
cd EngiSense-Analytics
```

Replace `YOUR-USERNAME` with the repository owner's GitHub username.

### Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch the Application

```bash
streamlit run engisense_dashboard.py
```

Streamlit will display the local application address in the terminal.

The dashboard loads the included CSV datasets and saved ML predictions, so retraining is not necessary for normal dashboard use.

---

## 11. Reproducibility and Data Organization

The repository separates raw measurements, processed datasets, trained models, and generated results.

This organization helps preserve the relationship between the original simulations and the later analysis.

The main data flow is:

**Simulation → CSV Export → Python Processing → Feature Extraction → Analysis and Classification → Dashboard**

The original MATLAB and Wokwi datasets are retained separately from processed outputs.

Several experiments use fixed random seeds to support repeatable results. Exact numerical outputs may still depend on software versions and numerical-library behavior.

The saved Random Forest model and evaluation reports are included so that the project can be inspected without repeating every experiment.

---

## 12. Key Findings

The work produced several findings worth carrying forward.

**Electrical disturbances can be characterized using relatively simple signal measurements.** RMS voltage, peak magnitude, waveform shape, and spectral content reveal meaningful differences between the simulated operating conditions.

**A Random Forest can classify the generated disturbance examples accurately under controlled conditions.** The model achieved 99% accuracy on its held-out synthetic test set.

**Noise robustness requires separate evaluation.** The classifier's performance deteriorated substantially when additional noise was applied to the MATLAB reference signals, despite its strong original test accuracy.

**The timing of spectral energy matters.** STFT analysis showed a useful distinction between broadly distributed noise energy and a localized electrical transient.

**Physics-informed methods can complement machine learning.** Combining absolute high-frequency energy with an energy concentration measure produced encouraging preliminary transient-detection results.

These findings provide a foundation for future experiments, but they do not establish real-world diagnostic reliability.

---

## 13. Current Limitations

EngiSense Analytics V1 remains a simulation-based prototype.

Its main limitations are:

- The ESP32 measurements were generated in Wokwi rather than acquired from physical hardware.
- The electrical waveforms were generated in MATLAB rather than measured from a live electrical network.
- The machine-learning training dataset is synthetic.
- The five MATLAB reference waveforms provide only a limited additional evaluation set.
- The noise experiments use repeated noisy realizations of those reference waveforms.
- The transient detector uses exploratory thresholds selected from controlled experiments.
- The detector has not been comprehensively evaluated against weak, overlapping, or differently shaped disturbances.
- The dashboard does not currently acquire live mains voltage or current waveforms.
- The system is not intended for electrical protection, safety-critical decisions, or certified power-quality monitoring.

Recognizing these limitations is part of the engineering process. They also help identify which improvements would be most valuable in a future version.

---

## 14. Development Roadmap

### Version 1 — Engineering Prototype

**Current implementation**

- ESP32 sensor monitoring in Wokwi
- MATLAB power-quality waveform simulation
- Time-domain and frequency-domain analysis
- FFT and STFT visualization
- Random Forest disturbance classification
- Noise-robustness experiments
- Physics-informed transient detection
- Streamlit dashboard
- GitHub documentation and deployment preparation

### Version 2 — Advanced Engineering Analysis

Possible next steps include:

- Independent power-quality datasets
- More realistic and varied disturbance simulations
- Simulink-based electrical models
- Improved noise-resistant feature extraction
- Expanded model comparison and validation
- More comprehensive transient detection testing
- Additional harmonic analysis and classification

### Version 3 — Hardware and Real-Time Monitoring

Longer-term possibilities include:

- Physical ESP32-based sensor acquisition
- Appropriately isolated electrical voltage and current measurement hardware
- Higher-sampling-rate data acquisition
- Real-time signal processing
- Live event detection and logging
- Monitoring alerts and deployment testing

Versions 2 and 3 are future development directions, not completed features.

---

## 15. Project Status

**EngiSense Analytics V1 — Engineering Prototype**

The core simulation, data integration, signal-processing, machine-learning, and dashboard components have been developed.

The remaining release activities are GitHub publication and public Streamlit deployment.

The project is intended to serve as a foundation for further electrical engineering experimentation, particularly in power-quality analysis and intelligent disturbance detection.

---

### ****EngiSense Analytics****

*Turning electrical measurements into meaningful engineering insight.*