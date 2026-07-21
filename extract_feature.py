import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm

SR = 16000  #sample rate
WINDOW_SAMPLES = SR * 1  # 1 second

#function for extracting highest energy window
def extract_1sec_window(y):
    if len(y) <= WINDOW_SAMPLES:
        # pad short clips with zeros at the end
        return np.pad(y, (0, WINDOW_SAMPLES - len(y)))
    #what above code does is if the clip is 1s or shorter pad it with zeros at the end to reach exactly 16000 samples

    # find the loudest 1-second window using RMS energy
    hop = SR // 10  # check every 0.1s
    #floor division, round the value to nearest lower number //
    best_start = 0
    best_energy = -1
    for start in range(0, len(y) - WINDOW_SAMPLES, hop):
        chunk = y[start:start + WINDOW_SAMPLES]
        energy = np.sum(chunk ** 2)
        if energy > best_energy:
            best_energy = energy
            best_start = start
    return y[best_start:best_start + WINDOW_SAMPLES]

#function for computing melspectrogram
def compute_logmel(y):
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=1024, hop_length=512, n_mels=32, power=2.0
    )
    return np.log(mel + 1e-6)

manifest = pd.read_csv("manifest.csv")

X = []
y_labels = []

for _, row in tqdm(manifest.iterrows(), total=len(manifest)):
    try:
        audio, _ = librosa.load(row['filepath'], sr=SR, mono=True)
        windowed = extract_1sec_window(audio)
        feat = compute_logmel(windowed)
        X.append(feat)
        y_labels.append(row['label'])
    except Exception as e:
        print(f"skipped {row['filepath']}: {e}")

X = np.array(X, dtype=np.float32)
y_labels = np.array(y_labels)

print(X.shape)
print(y_labels.shape)

classes = sorted(set(y_labels))
label_to_idx = {c: i for i, c in enumerate(classes)}
y_int = np.array([label_to_idx[l] for l in y_labels])

np.savez("features_cache.npz", X=X, y=y_int, classes=classes)
print("saved features_cache.npz")
print("classes:", classes)