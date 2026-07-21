import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf

SR = 16000
DURATION = 2  # seconds to record

print("Get ready to cough...")
sd.wait()
input("Press Enter, then cough within 2 seconds...")

# record from the default microphone
recording = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype="float32")
sd.wait()  # waits until recording is finished
audio = recording.flatten()  # sd.rec gives a 2D array, flatten to 1D like before

print("Recorded", audio.shape)

# --- same 1-second energy window extraction as before ---
WINDOW_SAMPLES = SR * 1
hop = SR // 10
best_start = 0
best_energy = -1
for start in range(0, len(audio) - WINDOW_SAMPLES, hop):
    chunk = audio[start:start + WINDOW_SAMPLES]
    energy = np.sum(chunk ** 2)
    if energy > best_energy:
        best_energy = energy
        best_start = start
windowed = audio[best_start:best_start + WINDOW_SAMPLES]

# --- same mel spectrogram computation as before ---
mel = librosa.feature.melspectrogram(
    y=windowed, sr=SR, n_fft=1024, hop_length=512, n_mels=32, power=2.0
)
feat = np.log(mel + 1e-6)

# --- same normalization as training (must match exactly) ---
data = np.load("features_cache.npz", allow_pickle=True)
X = data["X"]
classes = data["classes"]
mean = X.mean()
std = X.std()
feat_norm = (feat - mean) / (std + 1e-6)
feat_norm = feat_norm[np.newaxis, ..., np.newaxis].astype(np.float32)

# --- load the float model (simpler than the quantized one for this quick test) ---
model = tf.keras.models.load_model("experiments/model_v2_earlystop.keras")
prediction = model.predict(feat_norm)

predicted_class = classes[np.argmax(prediction)]
print("Prediction:", predicted_class)
print("Confidence per class:")
for cls, prob in zip(classes, prediction[0]):
    print(f"  {cls}: {prob:.3f}")