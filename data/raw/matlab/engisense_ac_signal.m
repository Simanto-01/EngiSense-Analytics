% ============================================================
% EngiSense Analytics
% Electrical Power Quality Signal Simulation and Analysis
% ============================================================

clc;
clear;
close all;

%% 1. Basic AC Signal Parameters

frequency = 50;          % Fundamental frequency (Hz)
amplitude = 325;          % Peak voltage (V)
sampling_rate = 5000;     % Sampling frequency (Hz)
duration = 0.1;           % Signal duration (seconds)

t = 0:1/sampling_rate:duration-1/sampling_rate;

% Generate normal 50 Hz AC voltage
voltage = amplitude * sin(2*pi*frequency*t);

%% 2. Normal AC Voltage Waveform

figure;
plot(t, voltage, 'b', 'LineWidth', 1.2);
title('EngiSense - 50 Hz AC Voltage');
xlabel('Time (seconds)');
ylabel('Voltage (V)');
grid on;

%% 3. RMS Voltage Analysis

rms_voltage = sqrt(mean(voltage.^2));
theoretical_rms = amplitude / sqrt(2);

fprintf('\n--- EngiSense RMS Analysis ---\n');
fprintf('Calculated RMS: %.2f V\n', rms_voltage);
fprintf('Theoretical RMS: %.2f V\n', theoretical_rms);

%% 4. FFT of Normal AC Voltage

N = length(voltage);

Y = fft(voltage);

P2 = abs(Y/N);
P1 = P2(1:N/2+1);
P1(2:end-1) = 2*P1(2:end-1);

f = sampling_rate*(0:N/2)/N;

figure;
plot(f, P1, 'b', 'LineWidth', 1.2);
title('EngiSense - FFT Frequency Spectrum');
xlabel('Frequency (Hz)');
ylabel('Peak Amplitude (V)');
grid on;
xlim([0 500]);

%% 5. Generate Harmonic Distortion (Up to 10th Order)

distorted_voltage = voltage;

% Harmonic percentages for orders 2 through 10
harmonic_percent = [5 15 3 10 2 7 2 5 1];

for h = 2:10

    harmonic_amplitude = ...
        amplitude * harmonic_percent(h-1)/100;

    distorted_voltage = distorted_voltage + ...
        harmonic_amplitude * ...
        sin(2*pi*h*frequency*t);

end

figure;
plot(t, distorted_voltage, 'r', 'LineWidth', 1.2);
title('EngiSense - Voltage with Harmonics Up to 10th Order');
xlabel('Time (seconds)');
ylabel('Voltage (V)');
grid on;

%% 6. Harmonic FFT and THD Analysis

N = length(distorted_voltage);

Y = fft(distorted_voltage);

P2 = abs(Y/N);
P1 = P2(1:N/2+1);
P1(2:end-1) = 2*P1(2:end-1);

f = sampling_rate*(0:N/2)/N;

harmonic_amplitudes = zeros(1,10);

for h = 1:10

    harmonic_frequency = h * frequency;

    [~, index] = min(abs(f - harmonic_frequency));

    harmonic_amplitudes(h) = P1(index);

end

THD = sqrt(sum(harmonic_amplitudes(2:10).^2)) ...
      / harmonic_amplitudes(1) * 100;

fprintf('\n--- EngiSense Harmonic Analysis ---\n');

for h = 1:10

    fprintf('Harmonic %d (%d Hz): %.2f V\n', ...
        h, h*frequency, harmonic_amplitudes(h));

end

fprintf('Total Harmonic Distortion: %.2f %%\n', THD);

figure;
bar((1:10)*frequency, harmonic_amplitudes);
title('EngiSense - Harmonic Spectrum');
xlabel('Frequency (Hz)');
ylabel('Peak Amplitude (V)');
grid on;

%% 7. Voltage Sag Simulation

sag_voltage = voltage;

sag_index = (t >= 0.04) & (t < 0.08);

% Voltage reduced to 50%
sag_voltage(sag_index) = ...
    0.5 * voltage(sag_index);

figure;
plot(t, sag_voltage, 'b', 'LineWidth', 1.2);
title('EngiSense - Voltage Sag');
xlabel('Time (seconds)');
ylabel('Voltage (V)');
grid on;

%% 8. Voltage Swell Simulation

swell_voltage = voltage;

swell_index = (t >= 0.04) & (t < 0.08);

% Voltage increased to 150%
swell_voltage(swell_index) = ...
    1.5 * voltage(swell_index);

figure;
plot(t, swell_voltage, 'r', 'LineWidth', 1.2);
title('EngiSense - Voltage Swell');
xlabel('Time (seconds)');
ylabel('Voltage (V)');
grid on;

%% 9. Voltage Interruption Simulation

interruption_voltage = voltage;

interruption_index = (t >= 0.04) & (t < 0.08);

% Complete interruption
interruption_voltage(interruption_index) = 0;

figure;
plot(t, interruption_voltage, 'k', 'LineWidth', 1.2);
title('EngiSense - Voltage Interruption');
xlabel('Time (seconds)');
ylabel('Voltage (V)');
grid on;

%% 10. Electrical Transient Simulation

transient_voltage = voltage;

start_time = 0.04;
transient_frequency = 1000;
transient_amplitude = 150;
decay_rate = 200;

transient_time = t - start_time;

transient = zeros(size(t));

transient_index = t >= start_time;

transient(transient_index) = ...
    transient_amplitude * ...
    exp(-decay_rate * transient_time(transient_index)) .* ...
    sin(2*pi*transient_frequency * ...
    transient_time(transient_index));

transient_voltage = voltage + transient;

figure;
plot(t, transient_voltage, 'm', 'LineWidth', 1.2);
title('EngiSense - Electrical Transient');
xlabel('Time (seconds)');
ylabel('Voltage (V)');
grid on;

%% 11. Comparison of All Power Quality Disturbances

figure;

subplot(5,1,1);
plot(t, voltage, 'b');
title('Normal AC Voltage');
ylabel('Voltage (V)');
grid on;

subplot(5,1,2);
plot(t, sag_voltage, 'b');
title('Voltage Sag');
ylabel('Voltage (V)');
grid on;

subplot(5,1,3);
plot(t, swell_voltage, 'r');
title('Voltage Swell');
ylabel('Voltage (V)');
grid on;

subplot(5,1,4);
plot(t, interruption_voltage, 'k');
title('Voltage Interruption');
ylabel('Voltage (V)');
grid on;

subplot(5,1,5);
plot(t, transient_voltage, 'm');
title('Electrical Transient');
xlabel('Time (seconds)');
ylabel('Voltage (V)');
grid on;

sgtitle('EngiSense Analytics - Power Quality Disturbances');

%% 12. Final Summary

fprintf('\n--- EngiSense Simulation Summary ---\n');
fprintf('Fundamental Frequency: %d Hz\n', frequency);
fprintf('Sampling Rate: %d Hz\n', sampling_rate);
fprintf('Number of Samples: %d\n', length(t));
fprintf('Signal Duration: %.2f seconds\n', duration);
fprintf('Calculated RMS Voltage: %.2f V\n', rms_voltage);
fprintf('THD: %.2f %%\n', THD);
fprintf('All simulations completed successfully!\n');
power_quality_data = table( ...
    t(:), voltage(:), sag_voltage(:), ...
    swell_voltage(:), interruption_voltage(:), ...
    transient_voltage(:), ...
    'VariableNames', { ...
    'time_s', 'normal_voltage_v', 'sag_voltage_v', ...
    'swell_voltage_v', 'interruption_voltage_v', ...
    'transient_voltage_v'});
openvar('power_quality_data')
