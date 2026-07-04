import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# librosa.load automatically converts stereo->mono and resamples for you
y, sr = librosa.load("Recording.wav", sr=16000, mono=True)

print(y.shape)
print(sr)

mel_spec = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_fft=1024,       # window size in samples (~64ms at 16kHz)
    hop_length=512,   # how far to slide the window each step (~32ms, 50% overlap)
    n_mels=32,        # number of mel bins
    power=2.0,
)

log_mel = np.log(mel_spec + 1e-6)  # +1e-6 avoids log(0) which is undefined

print(log_mel.shape)

plt.figure(figsize=(10, 4))
librosa.display.specshow(log_mel, sr=sr, hop_length=512, x_axis="time", y_axis="mel")
plt.colorbar(label="log energy")
plt.title("Log-Mel Spectrogram")
plt.tight_layout()
plt.savefig("mel_spectrogram.png")
plt.show()