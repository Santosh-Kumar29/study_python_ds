import numpy as np
import matplotlib.pyplot as pt

frequency = int(input("enter a range: "))

t = np.arange(0, 1, 1 / frequency)
signal = np.sin(2 * np.pi * 50 * t)

fourier_transform = np.fft.fft(signal)

number_of_sample_per_signal = len(signal)

frequency_resolution = fourier_transform / number_of_sample_per_signal

frequency_resolution = np.arange(0, number_of_sample_per_signal) * frequency_resolution

magnitude = np.abs(fourier_transform)

pt.plot(frequency_resolution, magnitude)
pt.xlabel('freq')
pt.ylabel('magnitude')
pt.show()
