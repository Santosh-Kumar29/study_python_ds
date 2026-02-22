import numpy as np
import matplotlib.pyplot as plt

# setting the sample frequency and time vector
Fs = 10000

# Here FS represent the sample frequency (number of sample per second), and t is a time vector representating
# the time point at which the signal is evaluated

# in this case the time vector range from 0 to 1 seconds with step size of 1/Fs which corresponds to the sampling
# interval
t = np.arange(0, 1, 1 / Fs)

#  it will be caluculate sinuosoidal compenent with 50hz, 150hz, 250hz
component_1 = np.sin(2 * np.pi * 50 * t)
component_2 = 0.2 * np.sin(2 * np.pi * 150 * t + np.pi / 3)
component_3 = 0.1 * np.sin(2 * np.pi * 250 * t + np.pi / 4)

# it will create a composite signal
signal = component_1 + component_2 + component_3

#  it will find the fourier transform
fft_result = np.fft.fft(signal)

# it is a function used to compute the discrete fourier transform of the signal


N = len(signal)
# n represent the number of sample in the signal


df = Fs / N
# df is the frequency resolution, calculate the sampling frequency divide the number of samples

frequencies = np.arange(0, N) * df
# np.arrange is used to create an array of frequency from 0 to n-1

#  it will calculate the magnitude spectrum
magnitude = np.abs(fft_result)

#  finally it will plot in the frequency spectrum
plt.plot(frequencies, magnitude)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Spectrum Analyzer')
plt.show()
