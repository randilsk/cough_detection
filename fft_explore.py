import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

#sf.read gives two things at ones, audio samples and the sample rate
#datah holds the audio numbers and the samplerate holdsthe number like 16000
data,samplerate = sf.read("Recording.wav") 



data =data[:,0] #if stereo, take only one channel
print(data.shape)

fft_result = np.fft.rfft(data)
magnitude = np.abs(fft_result)
print(fft_result.shape)
print(magnitude.shape)
print(fft_result[0:5])
print(magnitude[0:5])

freqs = np.fft.rfftfreq(len(data), d=1/samplerate)
print(freqs[0:5])
print(freqs[-5:])