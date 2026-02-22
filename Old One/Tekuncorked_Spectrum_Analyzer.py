import numpy as np
import matplotlib.pyplot as plt

Fs = 10000

t = np.arange(0, 1, 1 / Fs)

component_1 = np.sin(2 * np.pi * 50 * t)
component_2 = 0.2 * np.sin(2 * np.pi * 150 * t + np.pi / 3)
component_3 = 0.1 * np.sin(2 * np.pi * 250 * t + np.pi / 4)

signal = component_1 + component_2 + component_3
fft_result = np.fft.fft(signal)

N = len(signal)
df = Fs / N

frequencies = np.arange(0, N) * df

magnitude = np.abs(fft_result)

plt.plot(frequencies, magnitude)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Spectrum Analyzer')
plt.show()
